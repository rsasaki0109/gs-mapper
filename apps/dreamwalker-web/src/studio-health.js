import {
  dreamwalkerConfig,
  resolveDreamwalkerConfig,
  resolveWorldAssetBundle
} from './app-config.js';

function hasNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

export function normalizeLocalAssetPath(assetUrl) {
  if (!hasNonEmptyString(assetUrl)) {
    return '';
  }

  try {
    const parsed = new URL(assetUrl.trim(), 'https://dreamwalker.local');
    if (parsed.origin !== 'https://dreamwalker.local') {
      return '';
    }

    return parsed.pathname || '';
  } catch {
    return '';
  }
}

export function isLocalAssetUrl(assetUrl) {
  return Boolean(normalizeLocalAssetPath(assetUrl));
}

export function resolveStudioBundleTargetFragmentId(entryLike, bundleLike) {
  const entry =
    entryLike && typeof entryLike === 'object' ? entryLike : {};
  const bundle =
    bundleLike && typeof bundleLike === 'object' ? bundleLike : {};
  const state =
    bundle.state && typeof bundle.state === 'object' ? bundle.state : {};

  return (
    (hasNonEmptyString(entry.fragmentId) && entry.fragmentId.trim()) ||
    (hasNonEmptyString(state.fragmentId) && state.fragmentId.trim()) ||
    dreamwalkerConfig.defaultFragmentId
  );
}

export function buildWorldAssetHealth(assetBundle, options = {}) {
  const bundle =
    assetBundle && typeof assetBundle === 'object' ? assetBundle : {};
  const hasSplatCheck = Object.prototype.hasOwnProperty.call(options, 'splatExists');
  const hasColliderCheck = Object.prototype.hasOwnProperty.call(options, 'colliderExists');
  const splatExists = options.splatExists;
  const colliderExists = options.colliderExists;

  if (!hasNonEmptyString(bundle.splatUrl)) {
    return {
      status: 'error',
      label: 'Missing Splat',
      detail: 'splat asset が未設定です。world 表示を開始できません。'
    };
  }

  if (hasSplatCheck && splatExists === false) {
    return {
      status: 'error',
      label: 'Missing Splat File',
      detail: `splat file が見つかりません: ${bundle.splatUrl}`
    };
  }

  if (bundle.usesDemoFallback) {
    return {
      status: 'warning',
      label: 'Demo Fallback',
      detail: 'configured splat が無いため demo splat で動作しています。'
    };
  }

  if (!bundle.hasColliderMesh) {
    return {
      status: 'warning',
      label: 'Proxy Collider',
      detail: 'collider GLB が未設定のため walk は proxy floor を使います。'
    };
  }

  if (hasColliderCheck && colliderExists === false) {
    return {
      status: 'warning',
      label: 'Missing Collider File',
      detail: `collider GLB が見つかりません: ${bundle.colliderMeshUrl}`
    };
  }

  return {
    status: 'ready',
    label: 'Ready',
    detail: 'configured splat と collider GLB が揃っています。'
  };
}

export function resolveBundleWorldHealth(bundleLike, entryLike = null, options = {}) {
  const fragmentId = resolveStudioBundleTargetFragmentId(entryLike, bundleLike);
  const activeConfig = resolveDreamwalkerConfig(fragmentId);
  const assetBundle = resolveWorldAssetBundle(activeConfig, bundleLike?.assetWorkspace);
  const health = buildWorldAssetHealth(assetBundle, options);

  return {
    fragmentId,
    assetBundle,
    ...health
  };
}
