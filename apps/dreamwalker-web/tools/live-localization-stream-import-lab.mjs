import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

import { EXPERIMENT_LIVE_LOCALIZATION_STREAM_IMPORT_POLICIES } from '../src/sim2real-live-localization-import.js';

function meanOrNull(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return null;
  }

  return values.reduce((sum, value) => sum + Number(value), 0) / values.length;
}

function summarizeEstimate(result) {
  if (!result?.estimate) {
    return {
      kind: result?.kind ?? 'clear',
      poseCount: 0,
      label: '',
      sourceType: 'none',
      lastPosition: null
    };
  }

  const poses = Array.isArray(result.estimate.poses) ? result.estimate.poses : [];
  const lastPose = poses[poses.length - 1] ?? null;

  return {
    kind: result.kind,
    poseCount: poses.length,
    label: result.estimate.label,
    sourceType: result.estimate.sourceType,
    lastPosition: lastPose?.position ? [...lastPose.position] : null
  };
}

function buildFixtures() {
  const canonicalOrientation = [0, 0, 0, 1];
  return [
    {
      fixtureId: 'canonical-reset-and-append',
      label: 'Canonical Reset And Append',
      intent: 'Keep the current pose-estimate stream format stable for the live monitor.',
      messages: [
        { type: 'reset', label: 'ORB-SLAM3 Live' },
        {
          type: 'pose-estimate',
          label: 'ORB-SLAM3 Live',
          pose: {
            position: [0, 0, 0],
            orientation: canonicalOrientation,
            timestampSeconds: 0
          }
        },
        {
          type: 'pose-estimate',
          label: 'ORB-SLAM3 Live',
          pose: {
            position: [1, 0, 0],
            orientation: canonicalOrientation,
            timestampSeconds: 1
          }
        }
      ],
      expectedSummary: {
        kind: 'append',
        poseCount: 2,
        label: 'ORB-SLAM3 Live',
        sourceType: 'live-stream',
        lastPosition: [1, 0, 0]
      }
    },
    {
      fixtureId: 'snapshot-estimate',
      label: 'Snapshot Estimate',
      intent: 'Allow a full localization-estimate snapshot to replace the live trajectory in one message.',
      messages: [
        {
          type: 'localization-estimate',
          label: 'Snapshot Run',
          poses: [
            { position: [0, 0, 0], orientation: canonicalOrientation, timestampSeconds: 0 },
            { position: [2, 0, 0], orientation: canonicalOrientation, timestampSeconds: 2 }
          ]
        }
      ],
      expectedSummary: {
        kind: 'snapshot',
        poseCount: 2,
        label: 'Snapshot Run',
        sourceType: 'live-stream',
        lastPosition: [2, 0, 0]
      }
    },
    {
      fixtureId: 'wrapped-camera-pose',
      label: 'Wrapped CameraPose Alias',
      intent: 'Accept SDK-style pose wrappers without forcing clients to flatten cameraPose by hand.',
      messages: [
        {
          type: 'pose-estimate',
          label: 'SDK Stream',
          cameraPose: {
            position: [3, 1, 0],
            quaternion: canonicalOrientation,
            timestamp: '2026-04-02T00:00:00.000Z'
          }
        }
      ],
      expectedSummary: {
        kind: 'append',
        poseCount: 1,
        label: 'SDK Stream',
        sourceType: 'live-stream',
        lastPosition: [3, 1, 0]
      }
    },
    {
      fixtureId: 'top-level-shortcut',
      label: 'Top-Level Pose Shortcut',
      intent: 'Support quick local tools that send append messages without a nested pose object.',
      messages: [
        {
          type: 'append',
          label: 'CLI Stream',
          position: [4, 5, 6],
          orientation: canonicalOrientation,
          time: 3
        }
      ],
      expectedSummary: {
        kind: 'append',
        poseCount: 1,
        label: 'CLI Stream',
        sourceType: 'live-stream',
        lastPosition: [4, 5, 6]
      }
    },
    {
      fixtureId: 'clear-alias',
      label: 'Clear Alias',
      intent: 'Let live monitor tooling clear state with alias messages instead of one hard-coded string.',
      messages: [
        {
          type: 'pose-estimate',
          label: 'Alias Clear',
          pose: {
            position: [0, 0, 0],
            orientation: canonicalOrientation
          }
        },
        {
          type: 'clear-stream',
          label: 'Alias Clear'
        }
      ],
      expectedSummary: {
        kind: 'clear',
        poseCount: 0,
        label: '',
        sourceType: 'none',
        lastPosition: null
      }
    }
  ];
}

function evaluateFixture(policy, fixture) {
  const startedAt = performance.now();

  try {
    let estimate = null;
    let finalResult = null;

    for (const message of fixture.messages) {
      finalResult = policy.importMessage(estimate, message, {
        maxPoses: 240,
        defaultLabel: 'Live Localization Estimate'
      });
      estimate = finalResult.estimate;
    }

    const runtimeMs = performance.now() - startedAt;
    const summary = summarizeEstimate(finalResult);
    const expectedKeys = Object.keys(fixture.expectedSummary);
    const matchedKeys = expectedKeys.filter(
      (key) => JSON.stringify(summary[key]) === JSON.stringify(fixture.expectedSummary[key])
    );
    const matchScore = matchedKeys.length / Math.max(1, expectedKeys.length);

    return {
      fixtureId: fixture.fixtureId,
      label: fixture.label,
      intent: fixture.intent,
      status: 'ok',
      kind: summary.kind,
      matchScore,
      exactMatch: matchScore >= 0.999,
      summary,
      runtimeMs
    };
  } catch (error) {
    return {
      fixtureId: fixture.fixtureId,
      label: fixture.label,
      intent: fixture.intent,
      status: 'error',
      error: error instanceof Error ? error.message : String(error),
      runtimeMs: performance.now() - startedAt
    };
  }
}

function benchmarkPolicyRuntime(policy, fixtures, repetitions) {
  const samples = [];

  for (let repetition = 0; repetition < Math.max(1, Number(repetitions) || 1); repetition += 1) {
    for (const fixture of fixtures) {
      const startedAt = performance.now();
      try {
        let estimate = null;
        for (const message of fixture.messages) {
          const result = policy.importMessage(estimate, message, {
            maxPoses: 240,
            defaultLabel: 'Live Localization Estimate'
          });
          estimate = result.estimate;
        }
        samples.push(performance.now() - startedAt);
      } catch {
        continue;
      }
    }
  }

  if (samples.length === 0) {
    return { repetitions, sampleCount: 0, meanMs: null, medianMs: null };
  }

  const sorted = [...samples].sort((a, b) => a - b);
  return {
    repetitions,
    sampleCount: samples.length,
    meanMs: meanOrNull(samples),
    medianMs: sorted[Math.floor(sorted.length / 2)]
  };
}

function evaluateReadability(policy) {
  const source = String(policy.importMessage);
  const branchCount =
    (source.match(/\bif\b/g) ?? []).length +
    (source.match(/\?\s*[^:]+:/g) ?? []).length;
  const linesOfCode = source
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith('//')).length;
  const score = Math.max(1, 10 - Math.max(0, linesOfCode - 12) * 0.2 - Math.max(0, branchCount - 3) * 0.7);

  return {
    score: Number(score.toFixed(1)),
    linesOfCode,
    branchCount
  };
}

function evaluateExtensibility(policy) {
  const weights = {
    supportsCanonicalMessages: 2,
    supportsWrapperAliases: 2.5,
    supportsTopLevelPoseShortcuts: 2.5,
    supportsMessageAliases: 3
  };
  const supportedCapabilities = Object.entries(policy.capabilities)
    .filter(([, enabled]) => enabled)
    .map(([name]) => name);
  const score = supportedCapabilities.reduce((sum, capability) => sum + (weights[capability] ?? 0), 0);

  return {
    score: Number(score.toFixed(1)),
    supportedCapabilities
  };
}

function summarizePolicy(policy, fixtureReports, runtime) {
  const successful = fixtureReports.filter((report) => report.status === 'ok');

  return {
    name: policy.name,
    label: policy.label,
    style: policy.style,
    tier: policy.tier,
    capabilities: { ...policy.capabilities },
    fixtures: fixtureReports,
    aggregate: {
      successRate: successful.length / Math.max(1, fixtureReports.length),
      exactMatchRate: meanOrNull(successful.map((report) => (report.exactMatch ? 1 : 0))),
      meanMatchScore: meanOrNull(successful.map((report) => report.matchScore)),
      failedFixtures: fixtureReports
        .filter((report) => report.status !== 'ok')
        .map((report) => report.fixtureId)
    },
    runtime,
    readability: evaluateReadability(policy),
    extensibility: evaluateExtensibility(policy)
  };
}

export function buildLiveLocalizationStreamImportExperimentReport({
  repetitions = 200
} = {}) {
  const fixtures = buildFixtures();
  const policyReports = EXPERIMENT_LIVE_LOCALIZATION_STREAM_IMPORT_POLICIES.map((policy) =>
    summarizePolicy(
      policy,
      fixtures.map((fixture) => evaluateFixture(policy, fixture)),
      benchmarkPolicyRuntime(policy, fixtures, repetitions)
    )
  );

  const bestFit = [...policyReports].sort((left, right) => {
    const leftTuple = [
      left.aggregate.successRate ?? 0,
      left.aggregate.meanMatchScore ?? 0,
      left.aggregate.exactMatchRate ?? 0
    ];
    const rightTuple = [
      right.aggregate.successRate ?? 0,
      right.aggregate.meanMatchScore ?? 0,
      right.aggregate.exactMatchRate ?? 0
    ];
    return JSON.stringify(rightTuple).localeCompare(JSON.stringify(leftTuple), undefined, {
      numeric: true
    });
  })[0];
  const fastest = [...policyReports]
    .filter((report) => Number.isFinite(report.runtime.medianMs))
    .sort((left, right) => left.runtime.medianMs - right.runtime.medianMs)[0];
  const mostReadable = [...policyReports].sort(
    (left, right) => right.readability.score - left.readability.score
  )[0];
  const mostExtensible = [...policyReports].sort(
    (left, right) => right.extensibility.score - left.extensibility.score
  )[0];

  return {
    protocol: 'dreamwalker-web-experiment-report/v1',
    type: 'live-localization-stream-import-experiment-report',
    createdAt: new Date().toISOString(),
    problem: {
      name: 'live-localization-stream-import',
      statement:
        'Import live localization websocket messages without freezing one message envelope for browser, SDK, and quick local tooling clients.',
      stableInterface:
        'importLiveLocalizationStreamMessage(previousEstimate, rawMessage, options, policy?)'
    },
    fixtures: fixtures.map((fixture) => ({
      fixtureId: fixture.fixtureId,
      label: fixture.label,
      intent: fixture.intent,
      expectedSummary: fixture.expectedSummary
    })),
    metrics: {
      quality: ['successRate', 'exactMatchRate', 'meanMatchScore'],
      runtime: ['meanMs', 'medianMs'],
      readability: ['score', 'linesOfCode', 'branchCount'],
      extensibility: ['score', 'supportedCapabilities'],
      heuristicNotice: 'Readability/extensibility are generated heuristics, not objective truth.'
    },
    policies: policyReports,
    highlights: {
      bestFit: {
        policy: bestFit.name,
        label: bestFit.label,
        meanMatchScore: bestFit.aggregate.meanMatchScore
      },
      fastestMedianRuntime: {
        policy: fastest.name,
        label: fastest.label,
        medianMs: fastest.runtime.medianMs
      },
      mostReadable: {
        policy: mostReadable.name,
        label: mostReadable.label,
        score: mostReadable.readability.score
      },
      mostExtensible: {
        policy: mostExtensible.name,
        label: mostExtensible.label,
        score: mostExtensible.extensibility.score
      }
    }
  };
}

function parseArgs(argv) {
  const args = {
    repetitions: 200,
    output: '',
    json: false
  };

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--repetitions') {
      args.repetitions = Number(argv[index + 1] ?? '200');
      index += 1;
      continue;
    }
    if (token === '--output') {
      args.output = String(argv[index + 1] ?? '');
      index += 1;
      continue;
    }
    if (token === '--json') {
      args.json = true;
    }
  }

  return args;
}

export async function runCli(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const report = buildLiveLocalizationStreamImportExperimentReport({
    repetitions: args.repetitions
  });

  if (args.output) {
    const outputPath = path.resolve(args.output);
    await fs.mkdir(path.dirname(outputPath), { recursive: true });
    await fs.writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  }

  const payload = args.json
    ? report
    : {
        type: report.type,
        policyCount: report.policies.length,
        fixtureCount: report.fixtures.length,
        bestFit: report.highlights.bestFit,
        fastestMedianRuntime: report.highlights.fastestMedianRuntime
      };
  process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
}

const executedPath = process.argv[1] ? path.resolve(process.argv[1]) : '';
const modulePath = fileURLToPath(import.meta.url);

if (executedPath === modulePath) {
  runCli().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
