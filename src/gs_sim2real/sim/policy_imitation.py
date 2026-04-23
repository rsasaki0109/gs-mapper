"""Reference imitation policy for route policy replay batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, cast

from .gym_adapter import RoutePolicyAction, RoutePolicyGymAdapter
from .policy_dataset import RoutePolicyCallable, RoutePolicyGoal, RoutePolicyTransitionRecord
from .policy_quality import (
    RoutePolicyBaselineEvaluation,
    RoutePolicyQualityThresholds,
    evaluate_route_policy_baselines,
)
from .policy_replay import (
    RoutePolicyReplayBatch,
    RoutePolicyReplayFeatureSchema,
    RoutePolicyReplaySample,
    RoutePolicyReplaySource,
    build_route_policy_replay_batch,
)


ROUTE_POLICY_IMITATION_VERSION = "gs-mapper-route-policy-imitation/v1"

_TARGET_KEY_CANDIDATES = (
    ("payload.target.position.0", "payload.target.position.1", "payload.target.position.2"),
    ("payload.target.x", "payload.target.y", "payload.target.z"),
    ("target.position.0", "target.position.1", "target.position.2"),
    ("target.x", "target.y", "target.z"),
    ("payload.position.0", "payload.position.1", "payload.position.2"),
    ("payload.0.position.0", "payload.0.position.1", "payload.0.position.2"),
    ("payload.0", "payload.1", "payload.2"),
    ("0", "1", "2"),
)


RoutePolicyImitationSource = RoutePolicyReplayBatch | Sequence[RoutePolicyReplayBatch] | RoutePolicyReplaySource


@dataclass(frozen=True, slots=True)
class RoutePolicyActionDecoderConfig:
    """Configuration for decoding replay action vectors back into route actions."""

    target_keys: tuple[str, str, str] | None = None
    route_id_prefix: str = "imitation-route"
    target_frame_id: str | None = None

    def __post_init__(self) -> None:
        if self.target_keys is not None:
            object.__setattr__(
                self,
                "target_keys",
                _target_key_tuple(self.target_keys, "target_keys"),
            )
        if not str(self.route_id_prefix):
            raise ValueError("route_id_prefix must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "targetKeys": None if self.target_keys is None else list(self.target_keys),
            "routeIdPrefix": self.route_id_prefix,
            "targetFrameId": self.target_frame_id,
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyImitationFitConfig:
    """Training configuration for the dependency-free reference imitation model."""

    neighbor_count: int = 1
    distance_epsilon: float = 1e-9
    action_decoder: RoutePolicyActionDecoderConfig = field(default_factory=RoutePolicyActionDecoderConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "neighbor_count", _positive_int(self.neighbor_count, "neighbor_count"))
        object.__setattr__(self, "distance_epsilon", _positive_float(self.distance_epsilon, "distance_epsilon"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "neighborCount": self.neighbor_count,
            "distanceEpsilon": self.distance_epsilon,
            "actionDecoder": self.action_decoder.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyImitationModel:
    """Nearest-neighbor route policy fitted from schema-bound replay samples."""

    schema: RoutePolicyReplayFeatureSchema
    samples: tuple[RoutePolicyReplaySample, ...]
    config: RoutePolicyImitationFitConfig = field(default_factory=RoutePolicyImitationFitConfig)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_IMITATION_VERSION

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("samples must contain at least one replay sample")
        if self.schema.action_feature_count <= 0:
            raise ValueError("schema must contain at least one action key")
        for index, sample in enumerate(self.samples):
            _sample_dimensions(sample, self.schema, index)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def observation_matrix(self) -> tuple[tuple[float, ...], ...]:
        return tuple(sample.observation_vector for sample in self.samples)

    @property
    def action_matrix(self) -> tuple[tuple[float, ...], ...]:
        return tuple(sample.action_vector for sample in self.samples)

    def predict_action_vector(self, observation: Mapping[str, float]) -> tuple[float, ...]:
        """Predict one replay action vector from live adapter observation features."""

        observation_vector = _feature_vector(observation, self.schema.observation_keys)
        return self.predict_action_vector_from_features(observation_vector)

    def predict_action_vector_from_features(self, observation_vector: Sequence[float]) -> tuple[float, ...]:
        """Predict one replay action vector from an already ordered observation vector."""

        normalized_observation = _finite_vector(observation_vector, "observation_vector")
        if len(normalized_observation) != self.schema.observation_feature_count:
            raise ValueError("observation_vector length must match schema observation feature count")

        distances = sorted(
            (
                (_squared_distance(normalized_observation, sample.observation_vector), index, sample)
                for index, sample in enumerate(self.samples)
            ),
            key=lambda item: (item[0], item[1]),
        )
        neighbors = distances[: min(self.config.neighbor_count, len(distances))]
        if len(neighbors) == 1:
            return neighbors[0][2].action_vector

        weights = tuple(1.0 / (math.sqrt(distance) + self.config.distance_epsilon) for distance, _, _ in neighbors)
        total_weight = sum(weights)
        if total_weight <= 0.0 or not math.isfinite(total_weight):
            raise ValueError("neighbor weights must produce a finite positive sum")

        action_values: list[float] = []
        for action_index in range(self.schema.action_feature_count):
            weighted_value = sum(
                weight * sample.action_vector[action_index]
                for weight, (_, _, sample) in zip(weights, neighbors, strict=True)
            )
            action_values.append(_finite_float(weighted_value / total_weight, "predicted_action"))
        return tuple(action_values)

    def predict_action(self, observation: Mapping[str, float], info: Mapping[str, Any]) -> RoutePolicyAction:
        """Predict a route policy action accepted by ``RoutePolicyGymAdapter.step``."""

        return decode_route_policy_action_vector(
            self.predict_action_vector(observation),
            self.schema,
            info,
            config=self.config.action_decoder,
        )

    def as_policy(self) -> RoutePolicyCallable:
        """Return this fitted model as a route policy callable."""

        def policy(observation: Mapping[str, float], info: Mapping[str, Any]) -> RoutePolicyAction:
            return self.predict_action(observation, info)

        return policy

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-imitation-model",
            "version": self.version,
            "sampleCount": self.sample_count,
            "schema": self.schema.to_dict(),
            "config": self.config.to_dict(),
            "trainingSampleRefs": [_sample_ref(sample) for sample in self.samples],
            "metadata": _json_mapping(self.metadata),
        }


def fit_route_policy_imitation_model(
    source: RoutePolicyImitationSource,
    *,
    schema: RoutePolicyReplayFeatureSchema | None = None,
    config: RoutePolicyImitationFitConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyImitationModel:
    """Fit a deterministic k-nearest-neighbor route policy from replay batches."""

    batches = _replay_batches_from_source(source, schema=schema)
    if not batches:
        raise ValueError("source must contain at least one replay batch")
    resolved_schema = schema or batches[0].schema
    samples: list[RoutePolicyReplaySample] = []
    for index, batch in enumerate(batches):
        if not _schemas_compatible(batch.schema, resolved_schema):
            raise ValueError(f"batch {index} schema does not match the imitation fit schema")
        samples.extend(batch.samples)
    if not samples:
        raise ValueError("source must contain at least one replay sample")
    return RoutePolicyImitationModel(
        schema=resolved_schema,
        samples=tuple(samples),
        config=config or RoutePolicyImitationFitConfig(),
        metadata={
            "sourceBatchCount": len(batches),
            **_json_mapping(metadata or {}),
        },
    )


def decode_route_policy_action_vector(
    action_vector: Sequence[float],
    schema: RoutePolicyReplayFeatureSchema,
    info: Mapping[str, Any] | None = None,
    *,
    config: RoutePolicyActionDecoderConfig | None = None,
) -> dict[str, Any]:
    """Decode one replay action vector into a route action mapping."""

    resolved_config = config or RoutePolicyActionDecoderConfig()
    values = _finite_vector(action_vector, "action_vector")
    if len(values) != schema.action_feature_count:
        raise ValueError("action_vector length must match schema action feature count")

    action_features = dict(zip(schema.action_keys, values, strict=True))
    target_keys = resolved_config.target_keys or _infer_target_keys(schema.action_keys, action_features)
    missing_keys = tuple(key for key in target_keys if key not in action_features)
    if missing_keys:
        raise ValueError(f"action vector is missing target keys: {', '.join(missing_keys)}")

    x, y, z = (_finite_float(action_features[key], key) for key in target_keys)
    goal = _mapping_or_empty((info or {}).get("goal"))
    orientation = _orientation_from_goal(goal)
    target = {
        "position": [x, y, z],
        "orientationXyzw": orientation,
        "frameId": resolved_config.target_frame_id or _frame_id_from_goal(goal),
    }
    return {
        "routeId": _decoded_route_id(resolved_config, info or {}),
        "target": target,
    }


def evaluate_route_policy_imitation_model(
    adapters: Sequence[RoutePolicyGymAdapter],
    model: RoutePolicyImitationModel,
    *,
    episode_count: int,
    policy_name: str = "imitation",
    evaluation_id: str = "route-policy-imitation",
    seed_start: int = 0,
    goals: Sequence[RoutePolicyGoal] | None = None,
    max_steps: int | None = None,
    thresholds: RoutePolicyQualityThresholds | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyBaselineEvaluation:
    """Evaluate a fitted imitation model through the standard baseline scorer."""

    return evaluate_route_policy_baselines(
        adapters,
        {policy_name: model.as_policy()},
        episode_count=episode_count,
        evaluation_id=evaluation_id,
        seed_start=seed_start,
        goals=goals,
        max_steps=max_steps,
        thresholds=thresholds,
        metadata={
            "imitationModel": {
                "version": model.version,
                "sampleCount": model.sample_count,
                "config": model.config.to_dict(),
                "schema": model.schema.to_dict(),
            },
            **_json_mapping(metadata or {}),
        },
    )


def _replay_batches_from_source(
    source: RoutePolicyImitationSource,
    *,
    schema: RoutePolicyReplayFeatureSchema | None,
) -> tuple[RoutePolicyReplayBatch, ...]:
    if isinstance(source, RoutePolicyReplayBatch):
        return (source,)
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        items = tuple(source)
        if not items:
            raise ValueError("source must contain at least one replay batch or transition")
        if all(isinstance(item, RoutePolicyReplayBatch) for item in items):
            return cast(tuple[RoutePolicyReplayBatch, ...], items)
        return (build_route_policy_replay_batch(_transition_records(items), schema=schema),)
    return (build_route_policy_replay_batch(source, schema=schema),)


def _transition_records(items: Sequence[Any]) -> tuple[RoutePolicyTransitionRecord, ...]:
    if all(isinstance(item, RoutePolicyTransitionRecord) for item in items):
        return tuple(items)
    raise TypeError("source sequence must contain replay batches or transition records")


def _schemas_compatible(left: RoutePolicyReplayFeatureSchema, right: RoutePolicyReplayFeatureSchema) -> bool:
    return (
        left.observation_keys == right.observation_keys
        and left.action_keys == right.action_keys
        and left.next_observation_keys == right.next_observation_keys
    )


def _sample_dimensions(sample: RoutePolicyReplaySample, schema: RoutePolicyReplayFeatureSchema, index: int) -> None:
    if len(sample.observation_vector) != schema.observation_feature_count:
        raise ValueError(f"sample {index} observation vector length does not match schema")
    if len(sample.action_vector) != schema.action_feature_count:
        raise ValueError(f"sample {index} action vector length does not match schema")
    if len(sample.next_observation_vector) != schema.next_observation_feature_count:
        raise ValueError(f"sample {index} next observation vector length does not match schema")


def _feature_vector(features: Mapping[str, Any], keys: Sequence[str]) -> tuple[float, ...]:
    return tuple(_finite_float(features.get(key, 0.0), key) for key in keys)


def _finite_vector(values: Sequence[float], field_name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a numeric sequence")
    return tuple(_finite_float(value, field_name) for value in values)


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((a - b) * (a - b) for a, b in zip(left, right, strict=True))


def _infer_target_keys(action_keys: Sequence[str], action_features: Mapping[str, float]) -> tuple[str, str, str]:
    action_key_set = set(action_keys)
    for candidate in _TARGET_KEY_CANDIDATES:
        if set(candidate).issubset(action_key_set):
            return candidate
    if len(action_keys) == 3:
        return _target_key_tuple(tuple(action_keys), "action_keys")
    if len(action_features) == 3:
        return _target_key_tuple(tuple(action_features.keys()), "action_features")
    raise ValueError("could not infer target x/y/z action keys; configure target_keys")


def _decoded_route_id(config: RoutePolicyActionDecoderConfig, info: Mapping[str, Any]) -> str:
    step_index = _optional_int(info.get("stepIndex", info.get("step_index")))
    episode_index = _optional_int(info.get("episodeIndex", info.get("episode_index")))
    if episode_index is None and step_index is None:
        return config.route_id_prefix
    if episode_index is None:
        return f"{config.route_id_prefix}-{step_index}"
    if step_index is None:
        return f"{config.route_id_prefix}-{episode_index}"
    return f"{config.route_id_prefix}-{episode_index}-{step_index}"


def _orientation_from_goal(goal: Mapping[str, Any]) -> list[float]:
    orientation = goal.get("orientationXyzw", goal.get("orientation_xyzw"))
    if orientation is None:
        return [0.0, 0.0, 0.0, 1.0]
    values = _finite_vector(_sequence(orientation, "orientationXyzw"), "orientationXyzw")
    if len(values) != 4:
        raise ValueError("goal orientationXyzw must contain four values")
    return list(values)


def _frame_id_from_goal(goal: Mapping[str, Any]) -> str:
    return str(goal.get("frameId", goal.get("frame_id", "generic_world")))


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise TypeError(f"{field_name} must be a sequence")


def _target_key_tuple(keys: Sequence[str], field_name: str) -> tuple[str, str, str]:
    normalized = tuple(str(key) for key in keys)
    if len(normalized) != 3:
        raise ValueError(f"{field_name} must contain exactly three keys")
    if len(set(normalized)) != 3:
        raise ValueError(f"{field_name} must not contain duplicate keys")
    return cast(tuple[str, str, str], normalized)


def _sample_ref(sample: RoutePolicyReplaySample) -> dict[str, Any]:
    return {
        "datasetId": sample.dataset_id,
        "episodeId": sample.episode_id,
        "sceneId": sample.scene_id,
        "episodeIndex": sample.episode_index,
        "stepIndex": sample.step_index,
    }


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _finite_float(value, "json-value")
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _finite_float(value: Any, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _positive_float(value: float, field_name: str) -> float:
    normalized = _finite_float(value, field_name)
    if normalized <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return normalized


def _positive_int(value: int, field_name: str) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be positive")
    return normalized
