export function readNonEmptyString(value) {
  return typeof value === 'string' ? value.trim() : ''
}

export function deriveEstimateLabelFromFileName(fileName) {
  const normalized = readNonEmptyString(fileName).replace(/\.[^./\\]+$/u, '')

  return normalized || 'Localization Estimate'
}

export function isRecord(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

export function parseTimestampSecondsCandidate(value) {
  if (value === null || value === undefined) {
    return null
  }

  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null
  }

  const text = readNonEmptyString(value)

  if (!text) {
    return null
  }

  const numericValue = Number(text)

  if (Number.isFinite(numericValue)) {
    return numericValue
  }

  const parsedDateMs = Date.parse(text)
  return Number.isFinite(parsedDateMs) ? parsedDateMs / 1000 : null
}

export function toFiniteNumber(value) {
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

export function normalizeSignedDegrees(value) {
  const normalized = value % 360

  if (normalized <= -180) {
    return normalized + 360
  }

  if (normalized > 180) {
    return normalized - 360
  }

  return normalized
}

export function absoluteAngleDifferenceDegrees(a, b) {
  return Math.abs(normalizeSignedDegrees(a - b))
}

export function vectorSubtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

export function vectorNorm(vector) {
  return Math.hypot(vector[0], vector[1], vector[2])
}

export function normalizeQuaternionLike(value) {
  if (Array.isArray(value) && value.length >= 4) {
    const quaternion = value.slice(0, 4).map((item) => Number(item))
    return quaternion.every((item) => Number.isFinite(item)) ? quaternion : null
  }

  if (value && typeof value === 'object') {
    const quaternion = [value.x, value.y, value.z, value.w].map((item) => Number(item))
    return quaternion.every((item) => Number.isFinite(item)) ? quaternion : null
  }

  return null
}

export function quaternionToYawDegrees(quaternionLike) {
  const quaternion = normalizeQuaternionLike(quaternionLike)

  if (!quaternion) {
    return null
  }

  const [x, y, z, w] = quaternion
  const sinYaw = 2 * (w * y + x * z)
  const cosYaw = 1 - 2 * (y * y + z * z)
  return (Math.atan2(sinYaw, cosYaw) * 180) / Math.PI
}

export function resolvePositionVector(sample) {
  if (Array.isArray(sample.position) && sample.position.length >= 3) {
    const position = sample.position.slice(0, 3).map((value) => Number(value))
    return position.every((value) => Number.isFinite(value)) ? position : null
  }

  if (Array.isArray(sample.translation) && sample.translation.length >= 3) {
    const position = sample.translation.slice(0, 3).map((value) => Number(value))
    return position.every((value) => Number.isFinite(value)) ? position : null
  }

  const x = toFiniteNumber(sample.x)
  const y = toFiniteNumber(sample.y)
  const z = toFiniteNumber(sample.z)

  if (x !== null && y !== null && z !== null) {
    return [x, y, z]
  }

  return null
}

export function resolveYawDegrees(sample) {
  const yawDegrees = toFiniteNumber(sample.yawDegrees)

  if (yawDegrees !== null) {
    return yawDegrees
  }

  const yawRadians = toFiniteNumber(sample.yawRadians)

  if (yawRadians !== null) {
    return (yawRadians * 180) / Math.PI
  }

  return quaternionToYawDegrees(sample.orientation ?? sample.rotation ?? sample.quaternion) ?? 0
}

export function resolveTimestampSeconds(container, sample) {
  const candidates = [
    container.relativeTimeSeconds,
    sample.relativeTimeSeconds,
    container.timestampSeconds,
    sample.timestampSeconds,
    container.timestamp,
    sample.timestamp,
    container.timeSeconds,
    sample.timeSeconds,
    container.time,
    sample.time,
    container.capturedAt,
    sample.capturedAt,
    container.response?.relativeTimeSeconds,
    container.response?.timestampSeconds,
    container.response?.timestamp,
    container.response?.capturedAt
  ]

  for (const candidate of candidates) {
    const parsed = parseTimestampSecondsCandidate(candidate)

    if (parsed !== null) {
      return parsed
    }
  }

  return null
}

export function normalizePoseSample(sampleLike, index) {
  const container = sampleLike && typeof sampleLike === 'object' ? sampleLike : {}
  const sample = container.pose && typeof container.pose === 'object' ? container.pose : container
  const position = resolvePositionVector(sample)

  if (!position) {
    throw new Error(`pose sample ${index + 1} must include a finite 3D position`)
  }

  return {
    index,
    label: readNonEmptyString(container.label) || `pose:${index + 1}`,
    position,
    yawDegrees: resolveYawDegrees(sample),
    timestampSeconds: resolveTimestampSeconds(container, sample)
  }
}

function parseTextTrajectoryLine(line, lineIndex) {
  const tokens = line.split(/[\s,]+/u).filter(Boolean)

  if (tokens.length < 7) {
    return null
  }

  const numericTokens = tokens.map((token) => Number(token))

  if (tokens.length >= 8 && numericTokens.slice(0, 8).every((value) => Number.isFinite(value))) {
    return {
      lineIndex,
      timestamp: numericTokens[0],
      position: numericTokens.slice(1, 4),
      orientation: numericTokens.slice(4, 8)
    }
  }

  if (numericTokens.slice(0, 7).every((value) => Number.isFinite(value))) {
    return {
      lineIndex,
      timestamp: null,
      position: numericTokens.slice(0, 3),
      orientation: numericTokens.slice(3, 7)
    }
  }

  return null
}

export function parseTextTrajectory(rawText, options = {}) {
  const candidateLines = String(rawText)
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith('#') && !line.startsWith('//') && !line.startsWith('%'))
  const poses = []

  for (let index = 0; index < candidateLines.length; index += 1) {
    const parsedLine = parseTextTrajectoryLine(candidateLines[index], index)

    if (!parsedLine) {
      continue
    }

    poses.push({
      index: poses.length,
      label: `pose:${poses.length + 1}`,
      timestamp: parsedLine.timestamp,
      position: parsedLine.position,
      orientation: parsedLine.orientation
    })
  }

  if (poses.length === 0) {
    throw new Error('text trajectory must contain lines like: timestamp tx ty tz qx qy qz qw')
  }

  return {
    protocol: 'tum-trajectory-text/v1',
    type: 'localization-estimate',
    sourceType: 'tum-trajectory-text',
    label: deriveEstimateLabelFromFileName(options.fileName),
    poses
  }
}

function extractEstimatePoseList(inputLike) {
  if (Array.isArray(inputLike)) {
    return {
      poses: inputLike,
      sourceType: 'array'
    }
  }

  const input = inputLike && typeof inputLike === 'object' ? inputLike : {}

  if (input.type === 'route-capture-bundle' && Array.isArray(input.captures)) {
    return {
      poses: input.captures,
      sourceType: 'route-capture-bundle'
    }
  }

  const poseCollections = ['poses', 'trajectory', 'route', 'samples', 'estimates', 'captures']

  for (const key of poseCollections) {
    if (Array.isArray(input[key])) {
      return {
        poses: input[key],
        sourceType: key
      }
    }
  }

  if (input.pose && typeof input.pose === 'object') {
    return {
      poses: [input.pose],
      sourceType: 'single-pose'
    }
  }

  return {
    poses: [],
    sourceType: readNonEmptyString(input.type) || 'object'
  }
}

export function normalizeLocalizationEstimate(inputLike) {
  const input = inputLike && typeof inputLike === 'object' ? inputLike : {}

  if (input.type === 'localization-estimate' && Array.isArray(input.poses) && input.poses.length > 0) {
    return {
      protocol: readNonEmptyString(input.protocol),
      type: 'localization-estimate',
      sourceType: readNonEmptyString(input.sourceType) || 'poses',
      label:
        readNonEmptyString(input.label) ||
        readNonEmptyString(input.name) ||
        readNonEmptyString(input.runLabel) ||
        readNonEmptyString(input.fragmentLabel) ||
        readNonEmptyString(input.fragmentId) ||
        'Localization Estimate',
      poses: input.poses.map((pose, index) => normalizePoseSample(pose, index))
    }
  }

  const extracted = extractEstimatePoseList(inputLike)

  if (!Array.isArray(extracted.poses) || extracted.poses.length === 0) {
    throw new Error('localization estimate must include at least one pose')
  }

  const poses = extracted.poses.map((pose, index) => normalizePoseSample(pose, index))

  return {
    protocol: readNonEmptyString(input.protocol),
    type: 'localization-estimate',
    sourceType: extracted.sourceType,
    label:
      readNonEmptyString(input.label) ||
      readNonEmptyString(input.name) ||
      readNonEmptyString(input.runLabel) ||
      readNonEmptyString(input.fragmentLabel) ||
      readNonEmptyString(input.fragmentId) ||
      readNonEmptyString(input.type) ||
      'Localization Estimate',
    poses
  }
}
