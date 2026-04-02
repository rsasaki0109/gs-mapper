import {
  isRecord,
  normalizeLocalizationEstimate,
  normalizePoseSample,
  readNonEmptyString
} from './sim2real-localization-core.js'

function parseRawMessage(rawMessage) {
  return typeof rawMessage === 'string' || rawMessage instanceof String
    ? JSON.parse(String(rawMessage))
    : rawMessage
}

function resolveMaxPoses(options) {
  return Math.max(1, Math.round(Number(options.maxPoses ?? 240)))
}

function resolveNextLabel(parsedMessage, previousEstimate, options) {
  return (
    readNonEmptyString(parsedMessage.label) ||
    readNonEmptyString(previousEstimate?.label) ||
    readNonEmptyString(options.defaultLabel) ||
    'Live Localization Estimate'
  )
}

function buildClearResult() {
  return {
    kind: 'clear',
    estimate: null
  }
}

function buildSnapshotResult(snapshotPayload, parsedMessage, nextLabel) {
  const snapshotBody = Array.isArray(snapshotPayload)
    ? { poses: snapshotPayload }
    : isRecord(snapshotPayload)
      ? snapshotPayload
      : {}
  const normalizedEstimate = normalizeLocalizationEstimate({
    ...snapshotBody,
    protocol:
      readNonEmptyString(snapshotBody.protocol) || readNonEmptyString(parsedMessage.protocol),
    type: 'localization-estimate',
    label: readNonEmptyString(snapshotBody.label) || nextLabel,
    sourceType: 'live-stream'
  })

  return {
    kind: 'snapshot',
    estimate: {
      ...normalizedEstimate,
      sourceType: 'live-stream'
    }
  }
}

function buildAppendResult({
  previousEstimate,
  parsedMessage,
  nextLabel,
  nextPoseCandidate,
  shouldReset,
  options
}) {
  const existingPoses =
    shouldReset || !Array.isArray(previousEstimate?.poses) ? [] : previousEstimate.poses
  const nextPose = normalizePoseSample(nextPoseCandidate, existingPoses.length)
  const nextPoses = [...existingPoses, nextPose]
    .slice(-resolveMaxPoses(options))
    .map((pose, poseIndex) => ({
      ...pose,
      index: poseIndex
    }))

  return {
    kind: shouldReset ? 'reset' : 'append',
    estimate: {
      protocol: readNonEmptyString(parsedMessage.protocol),
      type: 'localization-estimate',
      sourceType: 'live-stream',
      label: nextLabel,
      poses: nextPoses
    }
  }
}

function hasTopLevelPosePayload(parsedMessage) {
  return (
    Array.isArray(parsedMessage.position) ||
    Array.isArray(parsedMessage.translation) ||
    Number.isFinite(Number(parsedMessage.x)) ||
    Number.isFinite(Number(parsedMessage.y)) ||
    Number.isFinite(Number(parsedMessage.z))
  )
}

class StrictCanonicalLiveLocalizationStreamImportPolicy {
  name = 'strict_canonical'
  label = 'Strict Canonical'
  style = 'exact-contract'
  tier = 'experiment'
  capabilities = {
    supportsCanonicalMessages: true,
    supportsWrapperAliases: false,
    supportsTopLevelPoseShortcuts: false,
    supportsMessageAliases: false
  }

  importMessage(previousEstimate, rawMessage, options = {}) {
    const parsedMessage = parseRawMessage(rawMessage)

    if (!isRecord(parsedMessage)) {
      throw new Error('live localization message must be a JSON object')
    }

    const messageType = readNonEmptyString(parsedMessage.type) || 'pose-estimate'
    const nextLabel = resolveNextLabel(parsedMessage, previousEstimate, options)

    if (messageType === 'clear' || messageType === 'localization-estimate-clear') {
      return buildClearResult()
    }

    if (messageType === 'localization-estimate') {
      return buildSnapshotResult(parsedMessage, parsedMessage, nextLabel)
    }

    const shouldReset = parsedMessage.reset === true || messageType === 'reset'
    if (!isRecord(parsedMessage.pose)) {
      if (shouldReset) {
        return buildClearResult()
      }
      throw new Error('live localization pose messages must include a pose object')
    }

    return buildAppendResult({
      previousEstimate,
      parsedMessage,
      nextLabel,
      nextPoseCandidate: {
        ...parsedMessage,
        ...parsedMessage.pose,
        label: readNonEmptyString(parsedMessage.pose.label) || nextLabel
      },
      shouldReset,
      options
    })
  }
}

class WrappedPoseLiveLocalizationStreamImportPolicy {
  name = 'wrapped_pose'
  label = 'Wrapped Pose'
  style = 'wrapper-oriented'
  tier = 'experiment'
  capabilities = {
    supportsCanonicalMessages: true,
    supportsWrapperAliases: true,
    supportsTopLevelPoseShortcuts: false,
    supportsMessageAliases: false
  }

  importMessage(previousEstimate, rawMessage, options = {}) {
    const parsedMessage = parseRawMessage(rawMessage)

    if (!isRecord(parsedMessage)) {
      throw new Error('live localization message must be a JSON object')
    }

    const messageType = readNonEmptyString(parsedMessage.type) || 'pose-estimate'
    const nextLabel = resolveNextLabel(parsedMessage, previousEstimate, options)

    if (messageType === 'clear' || messageType === 'localization-estimate-clear') {
      return buildClearResult()
    }

    if (
      messageType === 'localization-estimate' ||
      Array.isArray(parsedMessage.poses) ||
      Array.isArray(parsedMessage.trajectory) ||
      Array.isArray(parsedMessage.route) ||
      Array.isArray(parsedMessage.samples)
    ) {
      return buildSnapshotResult(parsedMessage, parsedMessage, nextLabel)
    }

    const shouldReset = parsedMessage.reset === true || messageType === 'reset'
    const poseWrapper = isRecord(parsedMessage.pose)
      ? parsedMessage.pose
      : isRecord(parsedMessage.cameraPose)
        ? parsedMessage.cameraPose
        : null

    if (!poseWrapper) {
      if (shouldReset) {
        return buildClearResult()
      }
      throw new Error('live localization pose messages must include pose or cameraPose')
    }

    return buildAppendResult({
      previousEstimate,
      parsedMessage,
      nextLabel,
      nextPoseCandidate: {
        ...parsedMessage,
        ...poseWrapper,
        label: readNonEmptyString(poseWrapper.label) || nextLabel
      },
      shouldReset,
      options
    })
  }
}

class AliasFriendlyLiveLocalizationStreamImportPolicy {
  name = 'alias_friendly'
  label = 'Alias Friendly'
  style = 'compatibility-first'
  tier = 'core'
  capabilities = {
    supportsCanonicalMessages: true,
    supportsWrapperAliases: true,
    supportsTopLevelPoseShortcuts: true,
    supportsMessageAliases: true
  }

  importMessage(previousEstimate, rawMessage, options = {}) {
    const parsedMessage = parseRawMessage(rawMessage)

    if (!isRecord(parsedMessage)) {
      throw new Error('live localization message must be a JSON object')
    }

    const messageTypeCandidate = readNonEmptyString(parsedMessage.type) || 'pose-estimate'
    const nextLabel = resolveNextLabel(parsedMessage, previousEstimate, options)
    const normalizedMessageType =
      {
        append: 'pose-estimate',
        pose: 'pose-estimate',
        sample: 'pose-estimate',
        estimate: 'localization-estimate',
        'stream-reset': 'reset',
        'clear-stream': 'clear',
        'stream-clear': 'clear'
      }[messageTypeCandidate] || messageTypeCandidate

    if (normalizedMessageType === 'clear' || normalizedMessageType === 'localization-estimate-clear') {
      return buildClearResult()
    }

    const explicitSnapshot =
      parsedMessage.estimateTrajectory ??
      parsedMessage.estimate ??
      parsedMessage.snapshot ??
      parsedMessage.localizationEstimate ??
      null
    if (
      normalizedMessageType === 'localization-estimate' ||
      Array.isArray(parsedMessage.poses) ||
      Array.isArray(parsedMessage.trajectory) ||
      Array.isArray(parsedMessage.route) ||
      Array.isArray(parsedMessage.samples) ||
      explicitSnapshot !== null
    ) {
      return buildSnapshotResult(explicitSnapshot ?? parsedMessage, parsedMessage, nextLabel)
    }

    const shouldReset = parsedMessage.reset === true || normalizedMessageType === 'reset'
    const poseWrapper = isRecord(parsedMessage.pose)
      ? parsedMessage.pose
      : isRecord(parsedMessage.cameraPose)
        ? parsedMessage.cameraPose
        : isRecord(parsedMessage.sample)
          ? parsedMessage.sample
          : null
    const hasPosePayload = Boolean(poseWrapper) || hasTopLevelPosePayload(parsedMessage)

    if (shouldReset && !hasPosePayload) {
      return buildClearResult()
    }

    const nextPoseCandidate = poseWrapper
      ? {
          ...parsedMessage,
          ...poseWrapper,
          label: readNonEmptyString(poseWrapper.label) || nextLabel
        }
      : {
          ...parsedMessage,
          label: nextLabel
        }

    return buildAppendResult({
      previousEstimate,
      parsedMessage,
      nextLabel,
      nextPoseCandidate,
      shouldReset,
      options
    })
  }
}

const corePolicies = {
  alias_friendly: new AliasFriendlyLiveLocalizationStreamImportPolicy()
}

export const CORE_LIVE_LOCALIZATION_STREAM_IMPORT_POLICIES = corePolicies

export function importLiveLocalizationStreamMessage(
  previousEstimate,
  rawMessage,
  options = {},
  policy = 'alias_friendly'
) {
  const policies = {
    strict_canonical: new StrictCanonicalLiveLocalizationStreamImportPolicy(),
    wrapped_pose: new WrappedPoseLiveLocalizationStreamImportPolicy(),
    ...corePolicies
  }

  if (!policies[policy]) {
    throw new Error(
      `unsupported live localization stream import policy: ${policy}. Expected one of ${Object.keys(
        policies
      )
        .sort()
        .join(', ')}`
    )
  }

  return policies[policy].importMessage(previousEstimate, rawMessage, options)
}

export const EXPERIMENT_LIVE_LOCALIZATION_STREAM_IMPORT_POLICIES = Object.freeze([
  new StrictCanonicalLiveLocalizationStreamImportPolicy(),
  new WrappedPoseLiveLocalizationStreamImportPolicy(),
  corePolicies.alias_friendly
])
