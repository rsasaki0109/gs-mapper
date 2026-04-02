export const robotBridgeDefaultUrl = 'ws://127.0.0.1:8790/robotics';
export const robotBridgeProtocolId = 'dreamwalker-robotics/v1';
export const robotBridgeBrowserSource = 'dreamwalker-live';
export const robotBridgeServerSource = 'dreamwalker-bridge';
export const robotBridgeCliSource = 'dreamwalker-cli';
export const robotBridgeRosbridgeSource = 'dreamwalker-rosbridge-relay';
export const robotBridgeLegacySource = 'legacy-client';
const textEncoder = new TextEncoder();

function isRecord(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function normalizeVector(value, expectedLength, label) {
  if (
    !Array.isArray(value) ||
    value.length !== expectedLength ||
    value.some((item) => !Number.isFinite(item))
  ) {
    throw new Error(`${label} must be an array of ${expectedLength} finite numbers`);
  }

  return value.map((item) => Number(item));
}

function normalizeRequiredNumber(value, label) {
  if (!Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number`);
  }

  return Number(value);
}

export function normalizeRobotBridgeUrl(candidate) {
  try {
    const normalized = new URL(candidate || robotBridgeDefaultUrl).toString();
    return normalized.replace(/\/$/, '');
  } catch {
    return robotBridgeDefaultUrl;
  }
}

export function parseRobotBridgeConfigFromSearch() {
  if (typeof window === 'undefined') {
    return {
      enabled: false,
      url: robotBridgeDefaultUrl
    };
  }

  const searchParams = new URLSearchParams(window.location.search);
  const bridgeParam = searchParams.get('robotBridge')?.trim() ?? '';
  const bridgeUrlParam = searchParams.get('robotBridgeUrl')?.trim() ?? '';
  const bridgeParamLooksLikeUrl = /^wss?:\/\//i.test(bridgeParam);
  const enabled =
    Boolean(bridgeUrlParam) ||
    bridgeParam === '1' ||
    bridgeParam === 'true' ||
    bridgeParamLooksLikeUrl;
  const explicitUrl = bridgeUrlParam || (bridgeParamLooksLikeUrl ? bridgeParam : '');

  return {
    enabled,
    url: normalizeRobotBridgeUrl(explicitUrl || robotBridgeDefaultUrl)
  };
}

export function buildRobotBridgeMessage(type, fields = {}, options = {}) {
  const normalizedType = typeof type === 'string' ? type.trim() : '';

  if (!normalizedType) {
    throw new Error('Robot bridge message type is required');
  }

  if (!isRecord(fields)) {
    throw new Error('Robot bridge message fields must be an object');
  }

  const source =
    typeof options.source === 'string' && options.source.trim()
      ? options.source.trim()
      : robotBridgeBrowserSource;
  const sentAt =
    typeof options.sentAt === 'string' && options.sentAt.trim()
      ? options.sentAt
      : new Date().toISOString();

  return {
    protocol: robotBridgeProtocolId,
    type: normalizedType,
    source,
    sentAt,
    ...fields
  };
}

export function stringifyRobotBridgeMessage(type, fields = {}, options = {}) {
  return JSON.stringify(buildRobotBridgeMessage(type, fields, options));
}

export function parseRobotBridgeMessage(rawMessage, options = {}) {
  const { allowLegacy = true, requireProtocol = false } = options;
  const parsed =
    typeof rawMessage === 'string' || rawMessage instanceof String
      ? JSON.parse(String(rawMessage))
      : rawMessage;

  if (!isRecord(parsed)) {
    throw new Error('Robot bridge message must be a JSON object');
  }

  const messageType = typeof parsed.type === 'string' ? parsed.type.trim() : '';

  if (!messageType) {
    throw new Error('Robot bridge message type is required');
  }

  const protocol = typeof parsed.protocol === 'string' ? parsed.protocol.trim() : '';

  if (protocol && protocol !== robotBridgeProtocolId) {
    throw new Error(`Unsupported robot bridge protocol: ${protocol}`);
  }

  if (!protocol && (requireProtocol || !allowLegacy)) {
    throw new Error('Robot bridge protocol is required');
  }

  const source =
    typeof parsed.source === 'string' && parsed.source.trim()
      ? parsed.source.trim()
      : null;
  const sentAt =
    typeof parsed.sentAt === 'string' && parsed.sentAt.trim()
      ? parsed.sentAt
      : null;

  return {
    isLegacy: !protocol,
    messageType,
    message: {
      ...parsed,
      type: messageType,
      protocol: protocol || null,
      source,
      sentAt
    }
  };
}

export function normalizeRobotBridgeMessage(rawMessage, options = {}) {
  const { fallbackSource = robotBridgeLegacySource } = options;
  const { messageType, message } = parseRobotBridgeMessage(rawMessage, options);
  const { type, protocol, source, sentAt, ...fields } = message;

  return buildRobotBridgeMessage(messageType, fields, {
    source: source || fallbackSource,
    sentAt: sentAt || undefined
  });
}

function normalizeFrameMetadata(metadata, label) {
  if (!isRecord(metadata)) {
    throw new Error(`${label} metadata must be an object`);
  }

  const timestamp =
    typeof metadata.timestamp === 'string' && metadata.timestamp.trim()
      ? metadata.timestamp
      : new Date().toISOString();
  const cameraId =
    typeof metadata.cameraId === 'string' && metadata.cameraId.trim()
      ? metadata.cameraId.trim()
      : null;
  const pose = isRecord(metadata.pose) ? metadata.pose : null;

  if (!cameraId) {
    throw new Error(`${label} metadata cameraId is required`);
  }

  if (!pose) {
    throw new Error(`${label} metadata pose is required`);
  }

  return {
    timestamp,
    cameraId,
    width: normalizeRequiredNumber(metadata.width, `${label} metadata width`),
    height: normalizeRequiredNumber(metadata.height, `${label} metadata height`),
    fov: normalizeRequiredNumber(metadata.fov, `${label} metadata fov`),
    pose: {
      position: normalizeVector(
        pose.position,
        3,
        `${label} metadata pose.position`
      ),
      orientation: normalizeVector(
        pose.orientation,
        4,
        `${label} metadata pose.orientation`
      )
    }
  };
}

async function normalizeBinaryFramePayload(payload, label) {
  if (payload instanceof Blob) {
    return new Uint8Array(await payload.arrayBuffer());
  }

  if (payload instanceof ArrayBuffer) {
    return new Uint8Array(payload);
  }

  if (ArrayBuffer.isView(payload)) {
    return new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength);
  }

  throw new Error(`${label} payload must be a Blob, ArrayBuffer, or typed array`);
}

async function buildBinaryFrameMessage(header, payload, label) {
  const headerBytes = textEncoder.encode(JSON.stringify(header));
  const payloadBytes = await normalizeBinaryFramePayload(payload, label);
  const messageBytes = new Uint8Array(4 + headerBytes.length + payloadBytes.length);
  const view = new DataView(messageBytes.buffer);

  view.setUint32(0, headerBytes.length, true);
  messageBytes.set(headerBytes, 4);
  messageBytes.set(payloadBytes, 4 + headerBytes.length);

  return messageBytes.buffer;
}

export async function buildCameraFrameMessage(blob, metadata = {}) {
  const header = {
    type: 'camera-frame',
    ...normalizeFrameMetadata(metadata, 'Camera frame')
  };

  return buildBinaryFrameMessage(header, blob, 'Camera frame');
}

export async function buildDepthFrameMessage(payload, metadata = {}) {
  const header = {
    type: 'depth-frame',
    encoding: '32FC1',
    nearClip: normalizeRequiredNumber(
      metadata?.nearClip,
      'Depth frame metadata nearClip'
    ),
    farClip: normalizeRequiredNumber(
      metadata?.farClip,
      'Depth frame metadata farClip'
    ),
    ...normalizeFrameMetadata(metadata, 'Depth frame')
  };

  return buildBinaryFrameMessage(header, payload, 'Depth frame');
}
