import { findSemanticZoneHits } from './semantic-zones.js';

function isHazardZone(zone) {
  return (
    zone.cost >= 80 ||
    zone.tags.includes('hazard') ||
    zone.tags.includes('void')
  );
}

function buildBoundsFromPositions(positions, padding = 0) {
  if (!Array.isArray(positions) || positions.length === 0) {
    return null;
  }

  const finitePadding = Number.isFinite(Number(padding)) ? Math.max(0, Number(padding)) : 0;
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;

  positions.forEach((position) => {
    if (!Array.isArray(position) || position.length < 3) {
      return;
    }

    const x = Number(position[0]);
    const z = Number(position[2]);

    if (!Number.isFinite(x) || !Number.isFinite(z)) {
      return;
    }

    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minZ = Math.min(minZ, z);
    maxZ = Math.max(maxZ, z);
  });

  if (!Number.isFinite(minX) || !Number.isFinite(maxX) || !Number.isFinite(minZ) || !Number.isFinite(maxZ)) {
    return null;
  }

  return {
    minX: minX - finitePadding,
    maxX: maxX + finitePadding,
    minZ: minZ - finitePadding,
    maxZ: maxZ + finitePadding,
    center: [
      (minX + maxX) / 2,
      0,
      (minZ + maxZ) / 2
    ],
    size: [
      Math.max(0.25, maxX - minX + finitePadding * 2),
      Math.max(0.25, maxZ - minZ + finitePadding * 2)
    ],
    padding: finitePadding
  };
}

function buildIssueSegments(nodes, predicate) {
  const segments = [];
  let startIndex = -1;
  let collectedNodes = [];

  const flushSegment = () => {
    if (startIndex < 0 || collectedNodes.length === 0) {
      startIndex = -1;
      collectedNodes = [];
      return;
    }

    const positions = collectedNodes
      .map((node) => (Array.isArray(node.position) ? node.position : null))
      .filter(Boolean);

    segments.push({
      startIndex,
      endIndex: collectedNodes[collectedNodes.length - 1].index,
      nodeCount: collectedNodes.length,
      positions
    });
    startIndex = -1;
    collectedNodes = [];
  };

  nodes.forEach((node) => {
    if (predicate(node)) {
      if (startIndex < 0) {
        startIndex = node.index;
      }
      collectedNodes.push(node);
      return;
    }

    flushSegment();
  });

  flushSegment();
  return segments;
}

export function isRouteNodeOutsideBounds(zoneMap, position) {
  if (!zoneMap || !Array.isArray(position) || position.length < 3) {
    return true;
  }

  const x = Number(position[0]);
  const z = Number(position[2]);

  if (!Number.isFinite(x) || !Number.isFinite(z)) {
    return true;
  }

  return x < zoneMap.minX || x > zoneMap.maxX || z < zoneMap.minZ || z > zoneMap.maxZ;
}

export function analyzeRouteAgainstZoneMap(route, zoneMap) {
  const positions = Array.isArray(route?.route) ? route.route : [];
  const labels = new Set();
  let hitNodeCount = 0;
  let outsideBoundsCount = 0;
  let hazardNodeCount = 0;
  let maxCost = Number(zoneMap?.defaultCost ?? 0) || 0;

  const nodes = positions.map((position, index) => {
    const outsideBounds = isRouteNodeOutsideBounds(zoneMap, position);

    if (outsideBounds) {
      outsideBoundsCount += 1;
      return {
        index,
        position: Array.isArray(position) ? [...position] : null,
        outsideBounds: true,
        zoneLabels: [],
        tags: [],
        maxCost: null,
        hazard: false
      };
    }

    const x = Number(position[0]);
    const z = Number(position[2]);
    const hits = findSemanticZoneHits(zoneMap, x, z);
    const pointCost = hits.length
      ? Math.max(...hits.map((zone) => zone.cost))
      : zoneMap.defaultCost;
    const zoneLabels = hits.map((zone) => zone.label);
    const tags = [...new Set(hits.flatMap((zone) => zone.tags))];
    const hazard = hits.some((zone) => isHazardZone(zone));

    if (hits.length > 0) {
      hitNodeCount += 1;
      hits.forEach((zone) => labels.add(zone.label));
    }

    if (hazard) {
      hazardNodeCount += 1;
    }

    maxCost = Math.max(maxCost, pointCost);

    return {
      index,
      position: [...position],
      outsideBounds: false,
      zoneLabels,
      tags,
      maxCost: pointCost,
      hazard
    };
  });

  return {
    nodeCount: positions.length,
    hitNodeCount,
    outsideBoundsCount,
    hazardNodeCount,
    maxCost,
    labels: [...labels],
    nodes
  };
}

export function buildRouteTuningDiagnostics(route, zoneMap, analysis, options = {}) {
  const corridorPadding = Number.isFinite(Number(options.corridorPadding))
    ? Math.max(0, Number(options.corridorPadding))
    : 0.75;
  const boundsPadding = Number.isFinite(Number(options.boundsPadding))
    ? Math.max(0, Number(options.boundsPadding))
    : corridorPadding;
  const routePositions = Array.isArray(route?.route) ? route.route : [];
  const coveredNodes = analysis.nodes.filter((node) => node.zoneLabels.length > 0 && !node.outsideBounds);
  const uncoveredNodes = analysis.nodes.filter(
    (node) => !node.outsideBounds && node.zoneLabels.length === 0
  );
  const hazardNodes = analysis.nodes.filter((node) => node.hazard);
  const outsideBoundsNodes = analysis.nodes.filter((node) => node.outsideBounds);
  const uncoveredSegments = buildIssueSegments(
    analysis.nodes,
    (node) => !node.outsideBounds && node.zoneLabels.length === 0
  );
  const hazardSegments = buildIssueSegments(analysis.nodes, (node) => node.hazard);
  const outsideBoundsSegments = buildIssueSegments(analysis.nodes, (node) => node.outsideBounds);
  const recommendations = [];

  if (route?.frameId && zoneMap?.frameId && route.frameId !== zoneMap.frameId) {
    recommendations.push({
      severity: 'warning',
      kind: 'frame-drift',
      message: `frame drift: route=${route.frameId} / zone=${zoneMap.frameId}`,
      routeFrameId: route.frameId,
      zoneFrameId: zoneMap.frameId
    });
  }

  outsideBoundsSegments.forEach((segment) => {
    recommendations.push({
      severity: 'warning',
      kind: 'outside-bounds',
      message: `route node ${segment.startIndex}-${segment.endIndex} が zone bounds の外です`,
      nodeRange: [segment.startIndex, segment.endIndex],
      suggestedBounds: buildBoundsFromPositions(segment.positions, boundsPadding)
    });
  });

  uncoveredSegments.forEach((segment) => {
    recommendations.push({
      severity: 'warning',
      kind: 'uncovered-corridor',
      message: `route node ${segment.startIndex}-${segment.endIndex} を覆う safe corridor zone が不足しています`,
      nodeRange: [segment.startIndex, segment.endIndex],
      suggestedRect: buildBoundsFromPositions(segment.positions, corridorPadding)
    });
  });

  hazardSegments.forEach((segment) => {
    recommendations.push({
      severity: 'warning',
      kind: 'hazard-overlap',
      message: `route node ${segment.startIndex}-${segment.endIndex} が hazard zone と重なっています`,
      nodeRange: [segment.startIndex, segment.endIndex],
      suggestedRect: buildBoundsFromPositions(segment.positions, corridorPadding / 2)
    });
  });

  return {
    uncoveredNodeCount: uncoveredNodes.length,
    coveredNodeCount: coveredNodes.length,
    routeBounds: buildBoundsFromPositions(routePositions, 0),
    routeBoundsWithPadding: buildBoundsFromPositions(routePositions, boundsPadding),
    coveredBounds: buildBoundsFromPositions(
      coveredNodes.map((node) => node.position),
      0
    ),
    uncoveredBounds: buildBoundsFromPositions(
      uncoveredNodes.map((node) => node.position),
      0
    ),
    hazardBounds: buildBoundsFromPositions(
      hazardNodes.map((node) => node.position),
      0
    ),
    outsideBoundsNodeCount: outsideBoundsNodes.length,
    uncoveredSegments,
    hazardSegments,
    outsideBoundsSegments,
    recommendations
  };
}

export function summarizeRouteZoneCoverage(route, zoneMap) {
  const analysis = analyzeRouteAgainstZoneMap(route, zoneMap);

  return {
    nodeCount: analysis.nodeCount,
    hitNodeCount: analysis.hitNodeCount,
    outsideBoundsCount: analysis.outsideBoundsCount,
    hazardNodeCount: analysis.hazardNodeCount,
    maxCost: analysis.maxCost,
    labels: analysis.labels
  };
}
