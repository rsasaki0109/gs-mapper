export const sim2realQueryProtocolId = 'dreamwalker-sim2real-query/v1'
export const localizationImageBenchmarkProtocolId = 'dreamwalker-localization-image-benchmark/v1'

function isRecord(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function readNonEmptyString(value) {
  return typeof value === 'string' ? value.trim() : ''
}

function normalizeFiniteNumber(value, label) {
  const normalized = Number(value)

  if (!Number.isFinite(normalized)) {
    throw new Error(`${label} must be a finite number`)
  }

  return normalized
}

function normalizePositiveNumber(value, label) {
  const normalized = normalizeFiniteNumber(value, label)

  if (normalized <= 0) {
    throw new Error(`${label} must be positive`)
  }

  return normalized
}

function normalizePositiveInteger(value, label) {
  const normalized = normalizePositiveNumber(value, label)
  const rounded = Math.round(normalized)

  if (Math.abs(rounded - normalized) > 0.0001) {
    throw new Error(`${label} must be an integer`)
  }

  return rounded
}

function normalizeVector(value, expectedLength, label) {
  if (
    !Array.isArray(value) ||
    value.length !== expectedLength ||
    value.some((item) => !Number.isFinite(Number(item)))
  ) {
    throw new Error(`${label} must be an array of ${expectedLength} finite numbers`)
  }

  return value.map((item) => Number(item))
}

function normalizeStringArray(value, label) {
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array`)
  }

  const normalized = value
    .map((item) => readNonEmptyString(item))
    .filter(Boolean)

  if (!normalized.length) {
    throw new Error(`${label} must include at least one string`)
  }

  return normalized
}

function parseRawMessage(rawMessage) {
  return typeof rawMessage === 'string' || rawMessage instanceof String
    ? JSON.parse(String(rawMessage))
    : rawMessage
}

function requireRecord(value, label) {
  if (!isRecord(value)) {
    throw new Error(`${label} must be a JSON object`)
  }

  return value
}

function firstMapping(record, keys) {
  for (const key of keys) {
    if (isRecord(record?.[key])) {
      return record[key]
    }
  }

  return null
}

function firstValue(record, keys) {
  for (const key of keys) {
    if (record?.[key] !== undefined && record[key] !== null) {
      return record[key]
    }
  }

  return undefined
}

function resolveProtocol(protocolCandidate, defaultProtocol) {
  const protocol = readNonEmptyString(protocolCandidate)

  if (
    protocol &&
    protocol !== sim2realQueryProtocolId &&
    protocol !== localizationImageBenchmarkProtocolId
  ) {
    throw new Error(`Unsupported sim2real protocol: ${protocol}`)
  }

  return protocol || defaultProtocol
}

function normalizeQueryDefaults(defaultsLike, aliases = false) {
  const defaults = requireRecord(defaultsLike, 'query-ready defaults')

  return {
    width: normalizePositiveInteger(
      firstValue(defaults, aliases ? ['width', 'imageWidth'] : ['width']),
      'query-ready defaults.width'
    ),
    height: normalizePositiveInteger(
      firstValue(defaults, aliases ? ['height', 'imageHeight'] : ['height']),
      'query-ready defaults.height'
    ),
    fovDegrees: normalizePositiveNumber(
      firstValue(defaults, aliases ? ['fovDegrees', 'fov'] : ['fovDegrees']),
      'query-ready defaults.fovDegrees'
    ),
    nearClip: normalizePositiveNumber(
      firstValue(defaults, aliases ? ['nearClip', 'near'] : ['nearClip']),
      'query-ready defaults.nearClip'
    ),
    farClip: normalizePositiveNumber(
      firstValue(defaults, aliases ? ['farClip', 'far'] : ['farClip']),
      'query-ready defaults.farClip'
    ),
    pointRadius: Math.max(
      0,
      normalizePositiveInteger(
        firstValue(defaults, aliases ? ['pointRadius', 'radius'] : ['pointRadius']),
        'query-ready defaults.pointRadius'
      )
    )
  }
}

function normalizePose(poseLike, aliases = false) {
  const pose = requireRecord(poseLike, 'render-result pose')

  return {
    position: normalizeVector(
      firstValue(pose, aliases ? ['position', 'translation'] : ['position']),
      3,
      'render-result pose.position'
    ),
    orientation: normalizeVector(
      firstValue(pose, aliases ? ['orientation', 'quaternion'] : ['orientation']),
      4,
      'render-result pose.orientation'
    )
  }
}

function normalizeQueryReadyMessage(bodyLike, protocolCandidate, aliases = false) {
  const body = requireRecord(bodyLike, 'query-ready message')
  const defaults = firstMapping(body, aliases ? ['defaults', 'queryDefaults', 'renderDefaults'] : ['defaults'])

  return {
    protocol: resolveProtocol(protocolCandidate ?? body.protocol, sim2realQueryProtocolId),
    type: 'query-ready',
    transport: readNonEmptyString(
      firstValue(body, aliases ? ['transport', 'queryTransport'] : ['transport'])
    ),
    endpoint: readNonEmptyString(
      firstValue(body, aliases ? ['endpoint', 'url'] : ['endpoint'])
    ),
    frameId: readNonEmptyString(
      firstValue(body, aliases ? ['frameId', 'mapFrameId'] : ['frameId'])
    ),
    renderer: readNonEmptyString(
      firstValue(body, aliases ? ['renderer', 'backend'] : ['renderer'])
    ),
    rendererReason: readNonEmptyString(
      firstValue(body, aliases ? ['rendererReason', 'backendReason'] : ['rendererReason'])
    ),
    requestTypes: normalizeStringArray(
      firstValue(body, aliases ? ['requestTypes', 'supportedRequests'] : ['requestTypes']),
      'query-ready requestTypes'
    ),
    defaults: normalizeQueryDefaults(defaults, aliases)
  }
}

function normalizeRenderResultMessage(bodyLike, protocolCandidate, aliases = false) {
  const body = requireRecord(bodyLike, 'render-result message')
  const pose =
    firstMapping(body, aliases ? ['pose', 'cameraPose'] : ['pose']) || body.pose

  const colorJpegBase64 = readNonEmptyString(
    firstValue(body, aliases ? ['colorJpegBase64', 'jpegBase64', 'colorBase64'] : ['colorJpegBase64'])
  )
  const depthBase64 = readNonEmptyString(
    firstValue(body, aliases ? ['depthBase64', 'depthFloat32Base64'] : ['depthBase64'])
  )

  if (!colorJpegBase64) {
    throw new Error('render-result colorJpegBase64 is required')
  }

  if (!depthBase64) {
    throw new Error('render-result depthBase64 is required')
  }

  return {
    protocol: resolveProtocol(protocolCandidate ?? body.protocol, sim2realQueryProtocolId),
    type: 'render-result',
    frameId: readNonEmptyString(body.frameId),
    width: normalizePositiveInteger(
      firstValue(body, aliases ? ['width', 'imageWidth'] : ['width']),
      'render-result width'
    ),
    height: normalizePositiveInteger(
      firstValue(body, aliases ? ['height', 'imageHeight'] : ['height']),
      'render-result height'
    ),
    fovDegrees: normalizePositiveNumber(
      firstValue(body, aliases ? ['fovDegrees', 'fov'] : ['fovDegrees']),
      'render-result fovDegrees'
    ),
    nearClip: normalizePositiveNumber(
      firstValue(body, aliases ? ['nearClip', 'near'] : ['nearClip']),
      'render-result nearClip'
    ),
    farClip: normalizePositiveNumber(
      firstValue(body, aliases ? ['farClip', 'far'] : ['farClip']),
      'render-result farClip'
    ),
    pointRadius: Math.max(
      0,
      normalizePositiveInteger(
        firstValue(body, aliases ? ['pointRadius', 'radius'] : ['pointRadius']),
        'render-result pointRadius'
      )
    ),
    pose: normalizePose(pose, aliases),
    cameraInfo: isRecord(body.cameraInfo) ? body.cameraInfo : {},
    colorJpegBase64,
    depthBase64
  }
}

function normalizeBenchmarkReportMessage(bodyLike, protocolCandidate) {
  const body = requireRecord(bodyLike, 'localization image benchmark report')

  return {
    ...body,
    protocol: resolveProtocol(protocolCandidate ?? body.protocol, localizationImageBenchmarkProtocolId),
    type: 'localization-image-benchmark-report'
  }
}

function normalizeErrorMessage(bodyLike, protocolCandidate, aliases = false) {
  const body = requireRecord(bodyLike, 'error message')
  const error = readNonEmptyString(
    firstValue(body, aliases ? ['error', 'message', 'detail'] : ['error'])
  )

  if (!error) {
    throw new Error('sim2real error message is required')
  }

  return {
    protocol: resolveProtocol(protocolCandidate ?? body.protocol, sim2realQueryProtocolId),
    type: 'error',
    error
  }
}

function normalizeStrictType(typeCandidate) {
  const normalized = readNonEmptyString(typeCandidate)

  if (
    normalized !== 'query-ready' &&
    normalized !== 'render-result' &&
    normalized !== 'localization-image-benchmark-report' &&
    normalized !== 'error'
  ) {
    throw new Error(`Unsupported sim2real message type: ${normalized || '(empty)'}`)
  }

  return normalized
}

function normalizeAliasFriendlyType(typeCandidate) {
  const normalized = readNonEmptyString(typeCandidate)

  return (
    {
      ready: 'query-ready',
      hello: 'query-ready',
      queryReady: 'query-ready',
      result: 'render-result',
      renderResult: 'render-result',
      frame: 'render-result',
      'benchmark-report': 'localization-image-benchmark-report',
      benchmarkReport: 'localization-image-benchmark-report',
      localizationImageBenchmarkReport: 'localization-image-benchmark-report',
      err: 'error',
      failure: 'error',
      queryError: 'error'
    }[normalized] || normalizeStrictType(normalized)
  )
}

class StrictCanonicalSim2realWebsocketMessagePolicy {
  name = 'strict_canonical'
  label = 'Strict Canonical'
  style = 'exact-contract'
  tier = 'experiment'
  capabilities = {
    supportsCanonicalEnvelope: true,
    supportsNestedWrappers: false,
    supportsTypeAliases: false,
    supportsFieldAliases: false
  }

  importMessage(rawMessage) {
    const parsed = requireRecord(parseRawMessage(rawMessage), 'sim2real message')
    const messageType = normalizeStrictType(parsed.type)

    if (messageType === 'query-ready') {
      return normalizeQueryReadyMessage(parsed, parsed.protocol, false)
    }
    if (messageType === 'render-result') {
      return normalizeRenderResultMessage(parsed, parsed.protocol, false)
    }
    if (messageType === 'localization-image-benchmark-report') {
      return normalizeBenchmarkReportMessage(parsed, parsed.protocol)
    }

    return normalizeErrorMessage(parsed, parsed.protocol, false)
  }
}

class EnvelopeFirstSim2realWebsocketMessagePolicy {
  name = 'envelope_first'
  label = 'Envelope First'
  style = 'wrapper-oriented'
  tier = 'experiment'
  capabilities = {
    supportsCanonicalEnvelope: true,
    supportsNestedWrappers: true,
    supportsTypeAliases: false,
    supportsFieldAliases: false
  }

  importMessage(rawMessage) {
    const parsed = requireRecord(parseRawMessage(rawMessage), 'sim2real message')
    const messageType = normalizeStrictType(
      parsed.type ??
        parsed.messageType ??
        parsed.kind ??
        firstMapping(parsed, ['payload', 'body', 'server', 'result', 'report', 'errorBody'])?.type
    )
    const body =
      firstMapping(parsed, ['payload', 'body', 'server', 'result', 'report', 'errorBody']) || parsed

    if (messageType === 'query-ready') {
      return normalizeQueryReadyMessage(body, parsed.protocol ?? body.protocol, false)
    }
    if (messageType === 'render-result') {
      return normalizeRenderResultMessage(body, parsed.protocol ?? body.protocol, false)
    }
    if (messageType === 'localization-image-benchmark-report') {
      return normalizeBenchmarkReportMessage(body, parsed.protocol ?? body.protocol)
    }

    return normalizeErrorMessage(body, parsed.protocol ?? body.protocol, false)
  }
}

class AliasFriendlySim2realWebsocketMessagePolicy {
  name = 'alias_friendly'
  label = 'Alias Friendly'
  style = 'compatibility-first'
  tier = 'core'
  capabilities = {
    supportsCanonicalEnvelope: true,
    supportsNestedWrappers: true,
    supportsTypeAliases: true,
    supportsFieldAliases: true
  }

  importMessage(rawMessage) {
    const parsed = requireRecord(parseRawMessage(rawMessage), 'sim2real message')
    const body =
      firstMapping(parsed, [
        'payload',
        'body',
        'server',
        'ready',
        'result',
        'report',
        'response',
        'errorBody'
      ]) || parsed
    const messageType = normalizeAliasFriendlyType(
      parsed.type ??
        parsed.messageType ??
        parsed.kind ??
        body.type ??
        body.messageType ??
        body.kind
    )

    if (messageType === 'query-ready') {
      return normalizeQueryReadyMessage(body, parsed.protocol ?? body.protocol, true)
    }
    if (messageType === 'render-result') {
      return normalizeRenderResultMessage(body, parsed.protocol ?? body.protocol, true)
    }
    if (messageType === 'localization-image-benchmark-report') {
      return normalizeBenchmarkReportMessage(body, parsed.protocol ?? body.protocol)
    }

    return normalizeErrorMessage(body, parsed.protocol ?? body.protocol, true)
  }
}

export const EXPERIMENT_SIM2REAL_WEBSOCKET_MESSAGE_POLICIES = [
  new StrictCanonicalSim2realWebsocketMessagePolicy(),
  new EnvelopeFirstSim2realWebsocketMessagePolicy(),
  new AliasFriendlySim2realWebsocketMessagePolicy()
]

export function importSim2realWebsocketMessage(rawMessage, policy = 'alias_friendly') {
  const policies = {
    strict_canonical: EXPERIMENT_SIM2REAL_WEBSOCKET_MESSAGE_POLICIES[0],
    envelope_first: EXPERIMENT_SIM2REAL_WEBSOCKET_MESSAGE_POLICIES[1],
    alias_friendly: EXPERIMENT_SIM2REAL_WEBSOCKET_MESSAGE_POLICIES[2]
  }
  const selectedPolicy = policies[policy]

  if (!selectedPolicy) {
    throw new Error(
      `Unsupported sim2real websocket message policy: ${policy}. Expected one of ${Object.keys(
        policies
      ).join(', ')}`
    )
  }

  return selectedPolicy.importMessage(rawMessage)
}
