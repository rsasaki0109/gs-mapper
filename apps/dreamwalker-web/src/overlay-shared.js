export const overlayStateKey = 'dreamwalker-live-overlay-state';
export const overlayRelayDefaultUrl = 'http://127.0.0.1:8787';

export function normalizeRelayUrl(candidate) {
  try {
    const normalized = new URL(candidate || overlayRelayDefaultUrl).toString();
    return normalized.endsWith('/') ? normalized.slice(0, -1) : normalized;
  } catch {
    return overlayRelayDefaultUrl;
  }
}

export function parseOverlayRelayConfigFromSearch() {
  if (typeof window === 'undefined') {
    return {
      enabled: false,
      url: overlayRelayDefaultUrl
    };
  }

  const searchParams = new URLSearchParams(window.location.search);
  const relayParam = searchParams.get('relay')?.trim() ?? '';
  const relayUrlParam = searchParams.get('relayUrl')?.trim() ?? '';
  const relayParamLooksLikeUrl = /^https?:\/\//i.test(relayParam);
  const enabled =
    Boolean(relayUrlParam) ||
    relayParam === '1' ||
    relayParam === 'true' ||
    relayParamLooksLikeUrl;
  const explicitUrl = relayUrlParam || (relayParamLooksLikeUrl ? relayParam : '');

  return {
    enabled,
    url: normalizeRelayUrl(explicitUrl || overlayRelayDefaultUrl)
  };
}

export function buildRelayEndpoint(relayUrl, pathname) {
  const relativePath = pathname.startsWith('/') ? pathname.slice(1) : pathname;
  return new URL(relativePath, `${normalizeRelayUrl(relayUrl)}/`).toString();
}

export function loadOverlayState() {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    const raw = window.localStorage.getItem(overlayStateKey);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}
