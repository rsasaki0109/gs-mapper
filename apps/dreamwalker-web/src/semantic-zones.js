function clampZoneCost(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 0;
  }

  return Math.max(0, Math.min(100, Math.round(numeric)));
}

function parseWorldXZ(positionLike) {
  if (!Array.isArray(positionLike)) {
    throw new Error('zone center must be an array');
  }

  if (positionLike.length === 2) {
    return [Number(positionLike[0]), Number(positionLike[1])];
  }

  if (positionLike.length >= 3) {
    return [Number(positionLike[0]), Number(positionLike[2])];
  }

  throw new Error('zone center must contain at least X and Z');
}

function ensureFinitePair(values, fieldName) {
  if (!values.every((value) => Number.isFinite(value))) {
    throw new Error(`${fieldName} must contain finite numbers`);
  }
}

function resolveSemanticZoneAccent(zone) {
  const tags = new Set(zone.tags);

  if (zone.cost >= 80 || tags.has('hazard') || tags.has('void')) {
    return '#ff9e72';
  }

  if (zone.cost >= 45 || tags.has('ledge') || tags.has('gate')) {
    return '#f4ca72';
  }

  if (tags.has('safe') || tags.has('stream')) {
    return '#93ffd4';
  }

  return '#85e3e1';
}

export function buildSemanticZoneMap(payload) {
  const bounds = payload?.bounds ?? {};
  const resolution = Number(payload?.resolution ?? 0.5);
  const minX = Number(bounds.minX);
  const maxX = Number(bounds.maxX);
  const minZ = Number(bounds.minZ);
  const maxZ = Number(bounds.maxZ);

  if (!Number.isFinite(resolution) || resolution <= 0) {
    throw new Error('semantic zone resolution must be > 0');
  }

  if (
    !Number.isFinite(minX) ||
    !Number.isFinite(maxX) ||
    !Number.isFinite(minZ) ||
    !Number.isFinite(maxZ) ||
    maxX <= minX ||
    maxZ <= minZ
  ) {
    throw new Error('semantic zone bounds must define a positive X/Z area');
  }

  const zones = (payload?.zones ?? []).map((zoneLike, index) => {
    const id = String(zoneLike?.id ?? `zone-${index}`);
    const label = String(zoneLike?.label ?? id);
    const shape = String(zoneLike?.shape ?? 'rect').trim().toLowerCase();
    const [centerX, centerZ] = parseWorldXZ(zoneLike?.center ?? [0, 0, 0]);
    ensureFinitePair([centerX, centerZ], `${id} center`);
    const cost = clampZoneCost(zoneLike?.cost ?? 100);
    const tags = Array.isArray(zoneLike?.tags)
      ? zoneLike.tags.filter((tag) => typeof tag === 'string' && tag.trim()).map((tag) => tag.trim())
      : [];

    if (shape === 'rect') {
      const size = Array.isArray(zoneLike?.size) ? zoneLike.size : [1, 1];
      const sizeX = Number(size[0]);
      const sizeZ = Number(size[1]);

      if (!Number.isFinite(sizeX) || !Number.isFinite(sizeZ) || sizeX <= 0 || sizeZ <= 0) {
        throw new Error(`rect zone '${id}' size must contain positive numbers`);
      }

      return {
        id,
        label,
        shape,
        centerX,
        centerZ,
        sizeX,
        sizeZ,
        radius: 0,
        cost,
        tags,
        accentColor: resolveSemanticZoneAccent({ cost, tags })
      };
    }

    if (shape === 'circle') {
      const radius = Number(zoneLike?.radius ?? 0);
      if (!Number.isFinite(radius) || radius <= 0) {
        throw new Error(`circle zone '${id}' radius must be > 0`);
      }

      return {
        id,
        label,
        shape,
        centerX,
        centerZ,
        sizeX: 0,
        sizeZ: 0,
        radius,
        cost,
        tags,
        accentColor: resolveSemanticZoneAccent({ cost, tags })
      };
    }

    throw new Error(`unsupported semantic zone shape: ${shape}`);
  });

  return {
    frameId: String(payload?.frameId ?? 'dreamwalker_map'),
    resolution,
    minX,
    maxX,
    minZ,
    maxZ,
    defaultCost: clampZoneCost(payload?.defaultCost ?? 0),
    zones
  };
}

export function serializeSemanticZoneMap(zoneMap) {
  if (!zoneMap) {
    return null;
  }

  return {
    frameId: zoneMap.frameId,
    resolution: zoneMap.resolution,
    defaultCost: zoneMap.defaultCost,
    bounds: {
      minX: zoneMap.minX,
      maxX: zoneMap.maxX,
      minZ: zoneMap.minZ,
      maxZ: zoneMap.maxZ
    },
    zones: zoneMap.zones.map((zone) => ({
      id: zone.id,
      label: zone.label,
      shape: zone.shape,
      center: [zone.centerX, 0, zone.centerZ],
      size: zone.shape === 'rect' ? [zone.sizeX, zone.sizeZ] : undefined,
      radius: zone.shape === 'circle' ? zone.radius : undefined,
      cost: zone.cost,
      tags: [...zone.tags]
    }))
  };
}

export function pointInSemanticZone(zone, x, z) {
  if (zone.shape === 'rect') {
    return (
      Math.abs(x - zone.centerX) <= zone.sizeX / 2 &&
      Math.abs(z - zone.centerZ) <= zone.sizeZ / 2
    );
  }

  if (zone.shape === 'circle') {
    return Math.hypot(x - zone.centerX, z - zone.centerZ) <= zone.radius;
  }

  return false;
}

export function findSemanticZoneHits(zoneMap, x, z) {
  if (!zoneMap) {
    return [];
  }

  return zoneMap.zones.filter((zone) => pointInSemanticZone(zone, x, z));
}

export function buildSemanticZoneProjectionPoints(zoneMap, activeZoneIds = [], anchorY = 0.4) {
  if (!zoneMap) {
    return [];
  }

  const activeIds = new Set(activeZoneIds);

  return zoneMap.zones.map((zone) => ({
    id: zone.id,
    kind: 'semantic-zone',
    label: zone.label,
    accentColor: zone.accentColor,
    isActive: activeIds.has(zone.id),
    cost: zone.cost,
    tags: zone.tags,
    position: [zone.centerX, anchorY, zone.centerZ]
  }));
}

export function buildSemanticZoneSurfacePoints(zoneMap, surfaceY = 0.05, circleSegments = 14) {
  if (!zoneMap) {
    return [];
  }

  return zoneMap.zones.flatMap((zone) => {
    if (zone.shape === 'rect') {
      const halfX = zone.sizeX / 2;
      const halfZ = zone.sizeZ / 2;
      const corners = [
        [zone.centerX - halfX, zone.centerZ - halfZ],
        [zone.centerX + halfX, zone.centerZ - halfZ],
        [zone.centerX + halfX, zone.centerZ + halfZ],
        [zone.centerX - halfX, zone.centerZ + halfZ]
      ];

      return corners.map(([x, z], index) => ({
        id: `${zone.id}-surface-${index}`,
        zoneId: zone.id,
        zoneShape: zone.shape,
        order: index,
        position: [x, surfaceY, z]
      }));
    }

    return Array.from({ length: circleSegments }, (_, index) => {
      const angle = (Math.PI * 2 * index) / circleSegments;
      return {
        id: `${zone.id}-surface-${index}`,
        zoneId: zone.id,
        zoneShape: zone.shape,
        order: index,
        position: [
          zone.centerX + Math.cos(angle) * zone.radius,
          surfaceY,
          zone.centerZ + Math.sin(angle) * zone.radius
        ]
      };
    });
  });
}

export function summarizeSemanticZoneHits(zoneHits) {
  if (!zoneHits.length) {
    return {
      label: 'Outside Map',
      maxCost: null,
      tags: []
    };
  }

  const uniqueTags = [...new Set(zoneHits.flatMap((zone) => zone.tags))];
  const labels = zoneHits.map((zone) => zone.label).join(' / ');
  const maxCost = Math.max(...zoneHits.map((zone) => zone.cost));

  return {
    label: labels,
    maxCost,
    tags: uniqueTags
  };
}
