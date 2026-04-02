import { opendir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const appRoot = path.resolve(__dirname, '..');
const repoRoot = path.resolve(appRoot, '..', '..');

export const robotMissionArtifactPackProtocolId =
  'dreamwalker-robot-mission-artifact-pack/v1';

export function defaultDiscoveryRoots() {
  return Array.from(
    new Set(
      [
        repoRoot,
        path.join(repoRoot, 'raw_assets'),
        path.join(repoRoot, 'apps', 'dreamwalker-web', 'public'),
        path.join(os.homedir(), 'Downloads'),
        path.join(os.homedir(), '.claude', 'downloads'),
        path.join('/media', process.env.USER || '')
      ].filter(Boolean)
    )
  );
}

export function shouldIgnoreDirectory(dirName) {
  return new Set([
    '.git',
    'node_modules',
    'Library',
    'Temp',
    'Logs',
    'Obj',
    'Build',
    'Builds',
    'dist',
    '.pytest_cache',
    '.mypy_cache'
  ]).has(dirName);
}

export function classifyCandidate(filePath) {
  const lowerPath = filePath.toLowerCase();

  if (lowerPath.endsWith('.artifact-pack.json')) {
    return 'artifact-pack';
  }
  if (lowerPath.endsWith('.mission.json')) {
    return 'mission-manifest';
  }
  if (
    lowerPath.endsWith('.sog') ||
    lowerPath.endsWith('.spz') ||
    lowerPath.endsWith('.splat')
  ) {
    return 'splat';
  }
  if (lowerPath.endsWith('.glb')) {
    return 'collider';
  }

  return '';
}

export async function walkDirectory(rootPath, visitor) {
  let directory;
  try {
    directory = await opendir(rootPath);
  } catch {
    return;
  }

  for await (const entry of directory) {
    const entryPath = path.join(rootPath, entry.name);
    if (entry.isDirectory()) {
      if (shouldIgnoreDirectory(entry.name)) {
        continue;
      }
      await walkDirectory(entryPath, visitor);
      continue;
    }

    if (!entry.isFile()) {
      continue;
    }

    await visitor(entryPath);
  }
}

export async function isArtifactPack(filePath) {
  try {
    const parsed = JSON.parse(await readFile(filePath, 'utf8'));
    return parsed?.protocol === robotMissionArtifactPackProtocolId;
  } catch {
    return false;
  }
}

export async function discoverRobotBundles(roots) {
  const normalizedRoots = Array.from(
    new Set((roots?.length ? roots : defaultDiscoveryRoots()).map((entry) => path.resolve(entry)))
  );
  const found = {
    roots: normalizedRoots,
    artifactPacks: [],
    missionManifests: [],
    splats: [],
    colliders: []
  };

  for (const root of normalizedRoots) {
    await walkDirectory(root, async (filePath) => {
      const category = classifyCandidate(filePath);
      if (!category) {
        return;
      }

      if (category === 'artifact-pack') {
        if (await isArtifactPack(filePath)) {
          const artifactStats = await stat(filePath);
          found.artifactPacks.push({
            path: filePath,
            mtimeMs: artifactStats.mtimeMs
          });
        }
        return;
      }

      if (category === 'mission-manifest') {
        found.missionManifests.push(filePath);
        return;
      }

      if (category === 'splat') {
        found.splats.push(filePath);
        return;
      }

      if (category === 'collider') {
        found.colliders.push(filePath);
      }
    });
  }

  found.artifactPacks.sort((left, right) => right.mtimeMs - left.mtimeMs);
  found.missionManifests.sort();
  found.splats.sort();
  found.colliders.sort();

  return found;
}
