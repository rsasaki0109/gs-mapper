import {
  importSim2realWebsocketMessage,
  localizationImageBenchmarkProtocolId,
  sim2realQueryProtocolId
} from './sim2real-websocket-protocol.js'

export const sim2realDefaultUrl = 'ws://127.0.0.1:8781/sim2real'
export { localizationImageBenchmarkProtocolId, sim2realQueryProtocolId }

function isRecord(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function normalizeFiniteNumber(value, label) {
  if (!Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number`)
  }

  return Number(value)
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
    value.some((item) => !Number.isFinite(item))
  ) {
    throw new Error(`${label} must be an array of ${expectedLength} finite numbers`)
  }

  return value.map((item) => Number(item))
}

function decodeBase64ToUint8Array(base64Value) {
  const binary = atob(base64Value)
  const bytes = new Uint8Array(binary.length)

  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }

  return bytes
}

function createPreviewCanvas(width, height) {
  if (typeof document === 'undefined') {
    return typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(width, height) : null
  }

  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  return canvas
}

function normalizeDepthSampleRange(depthSamples, farClip) {
  let minimum = Number.POSITIVE_INFINITY
  let maximum = Number.NEGATIVE_INFINITY

  depthSamples.forEach((sample) => {
    if (!Number.isFinite(sample) || sample >= farClip) {
      return
    }

    minimum = Math.min(minimum, sample)
    maximum = Math.max(maximum, sample)
  })

  if (!Number.isFinite(minimum) || !Number.isFinite(maximum)) {
    return {
      min: null,
      max: null
    }
  }

  if (Math.abs(maximum - minimum) < 0.0001) {
    maximum = minimum + 1
  }

  return {
    min: minimum,
    max: maximum
  }
}

export function normalizeSim2realUrl(candidate) {
  try {
    const normalized = new URL(candidate || sim2realDefaultUrl).toString()
    return normalized.replace(/\/$/, '')
  } catch {
    return sim2realDefaultUrl
  }
}

export function parseSim2realConfigFromSearch() {
  if (typeof window === 'undefined') {
    return {
      enabled: false,
      url: sim2realDefaultUrl
    }
  }

  const searchParams = new URLSearchParams(window.location.search)
  const sim2realParam = searchParams.get('sim2real')?.trim() ?? ''
  const sim2realUrlParam = searchParams.get('sim2realUrl')?.trim() ?? ''
  const sim2realParamLooksLikeUrl = /^wss?:\/\//i.test(sim2realParam)
  const enabled =
    Boolean(sim2realUrlParam) ||
    sim2realParam === '1' ||
    sim2realParam === 'true' ||
    sim2realParamLooksLikeUrl
  const explicitUrl = sim2realUrlParam || (sim2realParamLooksLikeUrl ? sim2realParam : '')

  return {
    enabled,
    url: normalizeSim2realUrl(explicitUrl || sim2realDefaultUrl)
  }
}

export function quaternionFromYawDegrees(yawDegrees) {
  const radians = (normalizeFiniteNumber(yawDegrees, 'yawDegrees') * Math.PI) / 180
  const halfYaw = radians * 0.5

  return [0, 0, Math.sin(halfYaw), Math.cos(halfYaw)]
}

export function buildSim2realRenderRequest(poseLike, options = {}) {
  const pose = isRecord(poseLike) ? poseLike : {}
  const position = normalizeVector(pose.position, 3, 'pose.position')
  const orientation = Array.isArray(pose.orientation)
    ? normalizeVector(pose.orientation, 4, 'pose.orientation')
    : quaternionFromYawDegrees(pose.yawDegrees ?? 0)

  return {
    protocol: sim2realQueryProtocolId,
    type: 'render',
    pose: {
      position,
      orientation
    },
    width: normalizePositiveInteger(options.width ?? 640, 'width'),
    height: normalizePositiveInteger(options.height ?? 480, 'height'),
    fovDegrees: normalizePositiveNumber(options.fovDegrees ?? 60, 'fovDegrees'),
    nearClip: normalizePositiveNumber(options.nearClip ?? 0.05, 'nearClip'),
    farClip: normalizePositiveNumber(options.farClip ?? 50, 'farClip'),
    pointRadius: Math.max(0, Math.round(Number(options.pointRadius ?? 1)))
  }
}

export function buildSim2realImageBenchmarkRequest({
  groundTruthBundle,
  estimate,
  alignment = 'auto',
  metrics = ['psnr', 'ssim', 'lpips'],
  maxFrames = null,
  lpipsNet = 'alex',
  device = 'cpu',
  responseTimeoutSeconds = 90
}) {
  if (!isRecord(groundTruthBundle)) {
    throw new Error('groundTruthBundle must be a JSON object')
  }

  if (!estimate || (typeof estimate !== 'object' && !Array.isArray(estimate))) {
    throw new Error('estimate must be a pose collection or localization estimate object')
  }

  if (!Array.isArray(metrics) || metrics.length === 0) {
    throw new Error('metrics must be a non-empty array')
  }

  const normalizedMetrics = metrics
    .map((metric) => (typeof metric === 'string' ? metric.trim().toLowerCase() : ''))
    .filter(Boolean)

  if (normalizedMetrics.length === 0) {
    throw new Error('metrics must include at least one metric name')
  }

  return {
    protocol: sim2realQueryProtocolId,
    type: 'localization-image-benchmark',
    groundTruthBundle,
    estimate,
    alignment: typeof alignment === 'string' && alignment.trim() ? alignment.trim() : 'auto',
    metrics: normalizedMetrics,
    ...(Number.isFinite(Number(maxFrames)) && Number(maxFrames) > 0
      ? { maxFrames: normalizePositiveInteger(Number(maxFrames), 'maxFrames') }
      : {}),
    lpipsNet: typeof lpipsNet === 'string' && lpipsNet.trim() ? lpipsNet.trim() : 'alex',
    device: typeof device === 'string' && device.trim() ? device.trim() : 'cpu',
    responseTimeoutSeconds: normalizePositiveNumber(responseTimeoutSeconds, 'responseTimeoutSeconds')
  }
}

export function parseSim2realMessage(rawMessage) {
  return importSim2realWebsocketMessage(rawMessage)
}

export function buildSim2realPreviewFromMessage(message) {
  const width = normalizePositiveInteger(message.width, 'render-result width')
  const height = normalizePositiveInteger(message.height, 'render-result height')
  const farClip = normalizePositiveNumber(message.farClip ?? 50, 'render-result farClip')
  const colorJpegBase64 =
    typeof message.colorJpegBase64 === 'string' ? message.colorJpegBase64.trim() : ''
  const depthBase64 =
    typeof message.depthBase64 === 'string' ? message.depthBase64.trim() : ''

  if (!colorJpegBase64) {
    throw new Error('render-result colorJpegBase64 is required')
  }

  if (!depthBase64) {
    throw new Error('render-result depthBase64 is required')
  }

  const depthBytes = decodeBase64ToUint8Array(depthBase64)
  const depth = new Float32Array(
    depthBytes.buffer,
    depthBytes.byteOffset,
    Math.floor(depthBytes.byteLength / 4)
  )
  const canvas = createPreviewCanvas(width, height)
  const depthRange = normalizeDepthSampleRange(depth, farClip)
  let depthSrc = ''

  if (canvas) {
    const context = canvas.getContext('2d')
    if (context) {
      const imageData = context.createImageData(width, height)

      for (let index = 0; index < width * height; index += 1) {
        const sample = depth[index]
        const pixelIndex = index * 4

        if (!Number.isFinite(sample) || sample >= farClip || depthRange.min === null || depthRange.max === null) {
          imageData.data[pixelIndex + 0] = 6
          imageData.data[pixelIndex + 1] = 18
          imageData.data[pixelIndex + 2] = 28
          imageData.data[pixelIndex + 3] = 255
          continue
        }

        const normalized = 1 - (sample - depthRange.min) / (depthRange.max - depthRange.min)
        imageData.data[pixelIndex + 0] = Math.round(26 + normalized * 84)
        imageData.data[pixelIndex + 1] = Math.round(72 + normalized * 136)
        imageData.data[pixelIndex + 2] = Math.round(112 + normalized * 118)
        imageData.data[pixelIndex + 3] = 255
      }

      context.putImageData(imageData, 0, 0)
      depthSrc =
        typeof canvas.convertToBlob === 'function'
          ? ''
          : canvas.toDataURL('image/png')
    }
  }

  return {
    colorSrc: `data:image/jpeg;base64,${colorJpegBase64}`,
    depthSrc,
    depthMin: depthRange.min,
    depthMax: depthRange.max,
    frameId: typeof message.frameId === 'string' ? message.frameId : '',
    width,
    height
  }
}
