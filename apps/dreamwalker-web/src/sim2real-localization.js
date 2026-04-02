import {
  absoluteAngleDifferenceDegrees,
  deriveEstimateLabelFromFileName,
  normalizeLocalizationEstimate,
  normalizePoseSample,
  normalizeSignedDegrees,
  parseTextTrajectory,
  readNonEmptyString,
  vectorNorm,
  vectorSubtract
} from './sim2real-localization-core.js'
import { importLiveLocalizationStreamMessage } from './sim2real-live-localization-import.js'

export const localizationMonitorDefaultUrl = 'ws://127.0.0.1:8782/localization'

export function normalizeLocalizationMonitorUrl(candidate) {
  try {
    const normalized = new URL(candidate || localizationMonitorDefaultUrl).toString()
    return normalized.replace(/\/$/, '')
  } catch {
    return localizationMonitorDefaultUrl
  }
}

export { normalizeLocalizationEstimate }

function buildRelativeTimestampTimeline(poses) {
  const timedPoses = poses
    .filter((pose) => Number.isFinite(pose.timestampSeconds))
    .sort((left, right) => {
      if (left.timestampSeconds !== right.timestampSeconds) {
        return left.timestampSeconds - right.timestampSeconds
      }

      return left.index - right.index
    })

  if (timedPoses.length === 0) {
    return []
  }

  const baseTimestampSeconds = timedPoses[0].timestampSeconds

  return timedPoses.map((pose) => ({
    ...pose,
    relativeTimestampSeconds: pose.timestampSeconds - baseTimestampSeconds
  }))
}

function interpolateLinear(a, b, t) {
  return a + (b - a) * t
}

function interpolateAngleDegrees(a, b, t) {
  return normalizeSignedDegrees(a + normalizeSignedDegrees(b - a) * t)
}

function buildInterpolatedPose(lowerPose, upperPose, targetRelativeTimestampSeconds, interpolationFactor) {
  return {
    index: lowerPose.index,
    label: `interp:${lowerPose.index + 1}-${upperPose.index + 1}`,
    position: [
      interpolateLinear(lowerPose.position[0], upperPose.position[0], interpolationFactor),
      interpolateLinear(lowerPose.position[1], upperPose.position[1], interpolationFactor),
      interpolateLinear(lowerPose.position[2], upperPose.position[2], interpolationFactor)
    ],
    yawDegrees: interpolateAngleDegrees(
      lowerPose.yawDegrees,
      upperPose.yawDegrees,
      interpolationFactor
    ),
    timestampSeconds: interpolateLinear(
      lowerPose.timestampSeconds,
      upperPose.timestampSeconds,
      interpolationFactor
    ),
    relativeTimestampSeconds: targetRelativeTimestampSeconds,
    interpolationFactor,
    interpolationKind: 'linear'
  }
}

function buildIndexAlignedPairs(groundTruthPoses, estimatePoses) {
  const matchedCount = Math.min(groundTruthPoses.length, estimatePoses.length)

  return Array.from({ length: matchedCount }, (_, pairIndex) => {
    const groundTruthPose = groundTruthPoses[pairIndex]
    const estimatePose = estimatePoses[pairIndex]
    const timeDeltaSeconds =
      Number.isFinite(groundTruthPose.timestampSeconds) &&
      Number.isFinite(estimatePose.timestampSeconds)
        ? Math.abs(estimatePose.timestampSeconds - groundTruthPose.timestampSeconds)
        : null

    return {
      pairIndex,
      groundTruth: groundTruthPose,
      estimate: estimatePose,
      timeDeltaSeconds
    }
  })
}

function buildTimestampAlignedPairs(groundTruthPoses, estimatePoses) {
  const groundTruthTimeline = buildRelativeTimestampTimeline(groundTruthPoses)
  const estimateTimeline = buildRelativeTimestampTimeline(estimatePoses)

  if (groundTruthTimeline.length === 0 || estimateTimeline.length === 0) {
    throw new Error('timestamp alignment requires timestamps in both ground truth and estimate')
  }

  const pairs = []
  let estimateCursor = 0

  for (let groundTruthIndex = 0; groundTruthIndex < groundTruthTimeline.length; groundTruthIndex += 1) {
    const groundTruthPose = groundTruthTimeline[groundTruthIndex]
    const targetRelativeTimestampSeconds = groundTruthPose.relativeTimestampSeconds

    if (estimateTimeline.length === 1) {
      pairs.push({
        pairIndex: pairs.length,
        groundTruth: groundTruthPose,
        estimate: estimateTimeline[0],
        timeDeltaSeconds: Math.abs(
          estimateTimeline[0].relativeTimestampSeconds - targetRelativeTimestampSeconds
        ),
        interpolationKind: 'single-sample'
      })
      continue
    }

    while (
      estimateCursor + 1 < estimateTimeline.length &&
      estimateTimeline[estimateCursor + 1].relativeTimestampSeconds < targetRelativeTimestampSeconds
    ) {
      estimateCursor += 1
    }

    const firstEstimatePose = estimateTimeline[0]
    const lastEstimatePose = estimateTimeline[estimateTimeline.length - 1]

    if (
      Math.abs(firstEstimatePose.relativeTimestampSeconds - targetRelativeTimestampSeconds) < 0.0001
    ) {
      pairs.push({
        pairIndex: pairs.length,
        groundTruth: groundTruthPose,
        estimate: firstEstimatePose,
        timeDeltaSeconds: 0,
        interpolationKind: 'exact'
      })
      continue
    }

    if (
      Math.abs(lastEstimatePose.relativeTimestampSeconds - targetRelativeTimestampSeconds) < 0.0001
    ) {
      pairs.push({
        pairIndex: pairs.length,
        groundTruth: groundTruthPose,
        estimate: lastEstimatePose,
        timeDeltaSeconds: 0,
        interpolationKind: 'exact'
      })
      continue
    }

    if (targetRelativeTimestampSeconds < firstEstimatePose.relativeTimestampSeconds) {
      pairs.push({
        pairIndex: pairs.length,
        groundTruth: groundTruthPose,
        estimate: firstEstimatePose,
        timeDeltaSeconds: Math.abs(
          firstEstimatePose.relativeTimestampSeconds - targetRelativeTimestampSeconds
        ),
        interpolationKind: 'clamped-start'
      })
      continue
    }

    if (targetRelativeTimestampSeconds > lastEstimatePose.relativeTimestampSeconds) {
      pairs.push({
        pairIndex: pairs.length,
        groundTruth: groundTruthPose,
        estimate: lastEstimatePose,
        timeDeltaSeconds: Math.abs(
          lastEstimatePose.relativeTimestampSeconds - targetRelativeTimestampSeconds
        ),
        interpolationKind: 'clamped-end'
      })
      continue
    }

    const lowerEstimatePose = estimateTimeline[estimateCursor]
    const upperEstimatePose = estimateTimeline[estimateCursor + 1]

    if (
      Math.abs(lowerEstimatePose.relativeTimestampSeconds - targetRelativeTimestampSeconds) < 0.0001
    ) {
      pairs.push({
        pairIndex: pairs.length,
        groundTruth: groundTruthPose,
        estimate: lowerEstimatePose,
        timeDeltaSeconds: 0,
        interpolationKind: 'exact'
      })
      continue
    }

    if (
      Math.abs(upperEstimatePose.relativeTimestampSeconds - targetRelativeTimestampSeconds) < 0.0001
    ) {
      pairs.push({
        pairIndex: pairs.length,
        groundTruth: groundTruthPose,
        estimate: upperEstimatePose,
        timeDeltaSeconds: 0,
        interpolationKind: 'exact'
      })
      continue
    }

    const interpolationFactor =
      (targetRelativeTimestampSeconds - lowerEstimatePose.relativeTimestampSeconds) /
      Math.max(
        0.000001,
        upperEstimatePose.relativeTimestampSeconds - lowerEstimatePose.relativeTimestampSeconds
      )

    pairs.push({
      pairIndex: pairs.length,
      groundTruth: groundTruthPose,
      estimate: buildInterpolatedPose(
        lowerEstimatePose,
        upperEstimatePose,
        targetRelativeTimestampSeconds,
        interpolationFactor
      ),
      timeDeltaSeconds: 0,
      interpolationKind: 'linear'
    })
  }

  return pairs
}

function buildStatistics(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return null
  }

  const sorted = [...values].sort((a, b) => a - b)
  const sum = values.reduce((total, value) => total + value, 0)
  const squaredSum = values.reduce((total, value) => total + value * value, 0)
  const middle = Math.floor(sorted.length / 2)
  const median =
    sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]

  return {
    min: sorted[0],
    max: sorted[sorted.length - 1],
    mean: sum / values.length,
    median,
    rmse: Math.sqrt(squaredSum / values.length)
  }
}

function extractGroundTruthPoses(bundleLike) {
  const bundle = bundleLike && typeof bundleLike === 'object' ? bundleLike : {}

  if (bundle.type !== 'route-capture-bundle') {
    throw new Error('ground truth must be a route-capture-bundle')
  }

  if (Array.isArray(bundle.captures) && bundle.captures.length > 0) {
    return bundle.captures.map((capture, index) => normalizePoseSample(capture, index))
  }

  if (Array.isArray(bundle.route) && bundle.route.length > 0) {
    return bundle.route.map((pose, index) => normalizePoseSample(pose, index))
  }

  throw new Error('ground truth bundle must include capture poses or route poses')
}

export function parseLocalizationEstimateDocument(rawText, options = {}) {
  const text = String(rawText ?? '').trim()

  if (!text) {
    throw new Error('localization estimate file is empty')
  }

  if (text.startsWith('{') || text.startsWith('[')) {
    try {
      const parsedJson = JSON.parse(text)
      const normalizedJson = normalizeLocalizationEstimate(parsedJson)

      return {
        ...normalizedJson,
        label:
          normalizedJson.label === 'Localization Estimate'
            ? deriveEstimateLabelFromFileName(options.fileName)
            : normalizedJson.label
      }
    } catch (error) {
      const jsonError = error instanceof Error ? error.message : String(error)

      try {
        return parseTextTrajectory(text, options)
      } catch {
        throw new Error(`failed to parse localization estimate JSON: ${jsonError}`)
      }
    }
  }

  return parseTextTrajectory(text, options)
}

export function applyLocalizationEstimateStreamMessage(
  previousEstimate,
  rawMessage,
  options = {}
) {
  return importLiveLocalizationStreamMessage(previousEstimate, rawMessage, options)
}

export function buildLocalizationBenchmarkReport({
  alignment = 'auto',
  groundTruthBundle,
  groundTruthLabel,
  estimateInput,
  estimateLabel
}) {
  const groundTruthPoses = extractGroundTruthPoses(groundTruthBundle)
  const normalizedEstimate = normalizeLocalizationEstimate(estimateInput)
  const requestedAlignment = readNonEmptyString(alignment) || 'auto'
  const groundTruthTimestampedCount = groundTruthPoses.filter((pose) =>
    Number.isFinite(pose.timestampSeconds)
  ).length
  const estimateTimestampedCount = normalizedEstimate.poses.filter((pose) =>
    Number.isFinite(pose.timestampSeconds)
  ).length
  const resolvedAlignment =
    requestedAlignment === 'auto'
      ? groundTruthTimestampedCount > 0 && estimateTimestampedCount > 0
        ? 'timestamp'
        : 'index'
      : requestedAlignment
  const matchedPairs =
    resolvedAlignment === 'timestamp'
      ? buildTimestampAlignedPairs(groundTruthPoses, normalizedEstimate.poses)
      : buildIndexAlignedPairs(groundTruthPoses, normalizedEstimate.poses)
  const matchedCount = matchedPairs.length
  const interpolatedCount = matchedPairs.filter(
    (pair) => pair.interpolationKind === 'linear'
  ).length
  const clampedCount = matchedPairs.filter(
    (pair) => pair.interpolationKind === 'clamped-start' || pair.interpolationKind === 'clamped-end'
  ).length

  if (matchedCount === 0) {
    throw new Error('localization benchmark requires at least one matched pose')
  }

  const samples = []
  const translationErrors = []
  const yawErrors = []
  const timeDeltaErrors = []
  const rpeTranslationErrors = []
  const rpeYawErrors = []

  for (let index = 0; index < matchedPairs.length; index += 1) {
    const pair = matchedPairs[index]
    const groundTruthPose = pair.groundTruth
    const estimatePose = pair.estimate
    const translationErrorMeters = vectorNorm(
      vectorSubtract(estimatePose.position, groundTruthPose.position)
    )
    const yawErrorDegrees = absoluteAngleDifferenceDegrees(
      estimatePose.yawDegrees,
      groundTruthPose.yawDegrees
    )

    translationErrors.push(translationErrorMeters)
    yawErrors.push(yawErrorDegrees)
    if (Number.isFinite(pair.timeDeltaSeconds)) {
      timeDeltaErrors.push(pair.timeDeltaSeconds)
    }
    samples.push({
      index,
      pairIndex: pair.pairIndex,
      interpolationKind: pair.interpolationKind ?? 'none',
      timeDeltaSeconds: pair.timeDeltaSeconds,
      translationErrorMeters,
      yawErrorDegrees,
      groundTruth: groundTruthPose,
      estimate: estimatePose
    })

    if (index > 0) {
      const previousPair = matchedPairs[index - 1]
      const previousGroundTruthPose = previousPair.groundTruth
      const previousEstimatePose = previousPair.estimate
      const groundTruthDelta = vectorSubtract(
        groundTruthPose.position,
        previousGroundTruthPose.position
      )
      const estimateDelta = vectorSubtract(estimatePose.position, previousEstimatePose.position)
      const rpeTranslationErrorMeters = vectorNorm(
        vectorSubtract(estimateDelta, groundTruthDelta)
      )
      const groundTruthYawDelta = normalizeSignedDegrees(
        groundTruthPose.yawDegrees - previousGroundTruthPose.yawDegrees
      )
      const estimateYawDelta = normalizeSignedDegrees(
        estimatePose.yawDegrees - previousEstimatePose.yawDegrees
      )

      rpeTranslationErrors.push(rpeTranslationErrorMeters)
      rpeYawErrors.push(absoluteAngleDifferenceDegrees(estimateYawDelta, groundTruthYawDelta))
    }
  }

  const translationStats = buildStatistics(translationErrors)
  const yawStats = buildStatistics(yawErrors)
  const timeDeltaStats = buildStatistics(timeDeltaErrors)
  const rpeTranslationStats = buildStatistics(rpeTranslationErrors)
  const rpeYawStats = buildStatistics(rpeYawErrors)

  return {
    protocol: 'dreamwalker-localization-benchmark/v1',
    type: 'localization-benchmark-report',
    createdAt: new Date().toISOString(),
    alignment: resolvedAlignment,
    requestedAlignment,
    groundTruth: {
      label:
        readNonEmptyString(groundTruthLabel) ||
        readNonEmptyString(groundTruthBundle.fragmentLabel) ||
        readNonEmptyString(groundTruthBundle.fragmentId) ||
        'Ground Truth Capture',
      fragmentId: readNonEmptyString(groundTruthBundle.fragmentId),
      fragmentLabel: readNonEmptyString(groundTruthBundle.fragmentLabel),
      capturedAt: readNonEmptyString(groundTruthBundle.capturedAt),
      poseCount: groundTruthPoses.length,
      timestampedPoseCount: groundTruthTimestampedCount
    },
    estimate: {
      label: readNonEmptyString(estimateLabel) || normalizedEstimate.label,
      protocol: normalizedEstimate.protocol,
      sourceType: normalizedEstimate.sourceType,
      poseCount: normalizedEstimate.poses.length,
      timestampedPoseCount: estimateTimestampedCount,
      interpolationMode: resolvedAlignment === 'timestamp' ? 'linear' : 'none'
    },
    matching: {
      matchedCount,
      groundTruthCount: groundTruthPoses.length,
      estimateCount: normalizedEstimate.poses.length,
      groundTruthRemainderCount: Math.max(0, groundTruthPoses.length - matchedCount),
      estimateRemainderCount: Math.max(0, normalizedEstimate.poses.length - matchedCount),
      timeAligned: resolvedAlignment === 'timestamp',
      interpolatedCount,
      clampedCount
    },
    metrics: {
      translation: translationStats,
      yaw: yawStats,
      timeDelta: timeDeltaStats,
      ateRmseMeters: translationStats?.rmse ?? null,
      rpeTranslation: rpeTranslationStats,
      rpeYaw: rpeYawStats,
      rpeTranslationRmseMeters: rpeTranslationStats?.rmse ?? null,
      rpeYawRmseDegrees: rpeYawStats?.rmse ?? null
    },
    samples
  }
}
