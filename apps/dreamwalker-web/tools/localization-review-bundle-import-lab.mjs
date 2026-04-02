import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

import {
  EXPERIMENT_LOCALIZATION_REVIEW_BUNDLE_IMPORT_POLICIES
} from '../src/sim2real-review-bundle-import.js';

function meanOrNull(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return null;
  }

  return values.reduce((sum, value) => sum + Number(value), 0) / values.length;
}

function buildRenderResult(position) {
  return {
    type: 'render-result',
    frameId: 'dreamwalker_map',
    width: 8,
    height: 8,
    colorJpegBase64: 'ZmFrZS1qcGVn',
    depthBase64: 'AAAAAA==',
    pose: {
      position,
      orientation: [0, 0, 0, 1]
    }
  };
}

function buildCaptureBundle(fragmentLabel, position) {
  return {
    protocol: 'dreamwalker-sim2real-capture/v1',
    type: 'route-capture-bundle',
    capturedAt: '2026-04-02T00:00:00.000Z',
    fragmentId: 'residency',
    fragmentLabel,
    captures: [
      {
        index: 0,
        label: 'gt:1',
        capturedAt: '2026-04-02T00:00:00.000Z',
        relativeTimeSeconds: 0,
        pose: {
          position,
          yawDegrees: 0
        },
        response: buildRenderResult(position)
      }
    ]
  };
}

function buildRunSnapshot({ id, label, sourceId, bundle }) {
  return {
    protocol: 'dreamwalker-localization-run/v1',
    type: 'localization-run-snapshot',
    id,
    label,
    savedAt: '2026-04-02T00:00:00.000Z',
    groundTruth: {
      sourceId,
      label: 'Ground Truth Capture',
      ...(bundle ? { bundle } : {})
    },
    estimate: {
      type: 'localization-estimate',
      label,
      sourceType: 'live-stream',
      poses: [
        {
          position: [0, 0, 0],
          orientation: [0, 0, 0, 1],
          timestampSeconds: 0
        }
      ]
    },
    benchmark: {
      alignment: 'timestamp',
      requestedAlignment: 'timestamp'
    },
    summary: {
      createdAt: '2026-04-02T00:00:00.000Z',
      alignment: 'timestamp',
      requestedAlignment: 'timestamp',
      groundTruthLabel: 'Ground Truth Capture',
      estimateLabel: label,
      sourceType: 'live-stream',
      matchedCount: 1,
      estimatePoseCount: 1,
      groundTruthPoseCount: 1,
      ateRmseMeters: 0,
      yawRmseDegrees: 0
    },
    imageBenchmark: {
      type: 'localization-image-benchmark-report',
      createdAt: '2026-04-02T00:00:00.000Z',
      alignment: 'timestamp',
      estimate: { label },
      groundTruth: { label: 'Ground Truth Capture' },
      matching: { matchedCount: 1 },
      metrics: {
        summary: {
          psnr: { mean: 28.5 },
          ssim: { mean: 0.92 },
          lpips: { mean: 0.12 }
        },
        highlights: {
          lpips: {
            ordering: 'max',
            frameIndex: 0,
            value: 0.24,
            groundTruthLabel: 'gt:1',
            estimateLabel: label,
            groundTruthColorJpegBase64: 'ZmFrZS1qcGVn',
            renderedColorJpegBase64: 'ZmFrZS1qcGVn'
          }
        }
      }
    }
  };
}

function buildFixtures() {
  const canonicalBundle = buildCaptureBundle('Canonical Capture', [0, 0, 0]);
  const linkedFallbackBundle = buildCaptureBundle('Linked Capture', [1, 0, 0]);
  const aliasBundle = buildCaptureBundle('Alias Capture', [2, 0, 0]);

  return [
    {
      fixtureId: 'canonical-embedded',
      label: 'Canonical Embedded Snapshot',
      intent: 'Keep the current exported review bundle shape stable when snapshots already embed ground truth bundles.',
      rawDocument: {
        protocol: 'dreamwalker-localization-review-bundle/v1',
        type: 'localization-review-bundle',
        createdAt: '2026-04-02T00:00:00.000Z',
        selection: {
          runIds: ['run-a'],
          baselineRunId: 'run-a',
          baselineLabel: 'Run A'
        },
        compareReport: {
          type: 'localization-run-compare-report',
          baselineRunId: 'run-a',
          baselineLabel: 'Run A',
          rows: [{ id: 'run-a' }]
        },
        linkedCaptures: [
          {
            sourceId: 'current-capture',
            label: 'Current Capture',
            bundle: canonicalBundle
          }
        ],
        runs: [
          {
            id: 'run-a',
            label: 'Run A',
            snapshot: buildRunSnapshot({
              id: 'run-a',
              label: 'Run A',
              sourceId: 'current-capture',
              bundle: canonicalBundle
            })
          }
        ]
      },
      expectedSummary: {
        runCount: 1,
        captureCount: 1,
        baselineRunId: 'run-a',
        firstRunLabel: 'Run A',
        firstGroundTruthSourceId: 'current-capture'
      }
    },
    {
      fixtureId: 'linked-capture-fallback',
      label: 'Linked Capture Fallback',
      intent: 'Recover snapshot ground truth bundles from linked captures when portable snapshots omit the embedded bundle.',
      rawDocument: {
        protocol: 'dreamwalker-localization-review-bundle/v1',
        type: 'localization-review-bundle',
        createdAt: '2026-04-02T00:00:01.000Z',
        selection: {
          runIds: ['run-b'],
          baselineRunId: 'run-b',
          baselineLabel: 'Run B'
        },
        compareReport: {
          type: 'localization-run-compare-report',
          baselineRunId: 'run-b',
          baselineLabel: 'Run B',
          rows: [{ id: 'run-b' }]
        },
        linkedCaptures: [
          {
            sourceId: 'capture-shelf:linked-capture',
            label: 'Linked Capture',
            bundle: linkedFallbackBundle
          }
        ],
        runs: [
          {
            id: 'run-b',
            label: 'Run B',
            snapshot: buildRunSnapshot({
              id: 'run-b',
              label: 'Run B',
              sourceId: 'capture-shelf:linked-capture',
              bundle: null
            })
          }
        ]
      },
      expectedSummary: {
        runCount: 1,
        captureCount: 1,
        baselineRunId: 'run-b',
        firstRunLabel: 'Run B',
        firstGroundTruthSourceId: 'capture-shelf:linked-capture'
      }
    },
    {
      fixtureId: 'alias-wrapper',
      label: 'Alias Wrapper',
      intent: 'Accept review-bundle wrappers that rename runs, compare report, capture bundle fields, and snapshot keys.',
      rawDocument: {
        reviewBundle: {
          protocol: 'dreamwalker-localization-review-bundle/v1',
          type: 'localization-review-bundle',
          createdAt: '2026-04-02T00:00:02.000Z',
          compareSelection: {
            runIds: ['run-c'],
            baselineId: 'run-c',
            baselineLabel: 'Run C'
          },
          compare: {
            type: 'localization-run-compare-report',
            baselineRunId: 'run-c',
            baselineLabel: 'Run C',
            rows: [{ id: 'run-c' }]
          },
          captures: [
            {
              captureSourceId: 'capture-shelf:alias-capture',
              label: 'Alias Capture',
              captureBundle: aliasBundle
            }
          ],
          portableRuns: [
            {
              runId: 'run-c',
              runLabel: 'Run C',
              portableSnapshot: {
                ...buildRunSnapshot({
                  id: 'run-c',
                  label: 'Run C',
                  sourceId: 'capture-shelf:alias-capture',
                  bundle: null
                }),
                runId: 'run-c',
                runLabel: 'Run C',
                groundTruth: {
                  captureSourceId: 'capture-shelf:alias-capture',
                  label: 'Ground Truth Capture'
                }
              }
            }
          ]
        }
      },
      expectedSummary: {
        runCount: 1,
        captureCount: 1,
        baselineRunId: 'run-c',
        firstRunLabel: 'Run C',
        firstGroundTruthSourceId: 'capture-shelf:alias-capture'
      }
    }
  ];
}

function summarizeBundle(result) {
  const firstRun = result.runShelfEntries[0] ?? null;
  return {
    runCount: result.runShelfEntries.length,
    captureCount: result.captureShelfEntries.length,
    baselineRunId: result.baselineRunId || '',
    firstRunLabel: firstRun?.label || '',
    firstGroundTruthSourceId: firstRun?.groundTruth?.sourceId || ''
  };
}

function evaluateFixture(policy, fixture) {
  const startedAt = performance.now();

  try {
    const result = policy.importDocument(fixture.rawDocument);
    const runtimeMs = performance.now() - startedAt;
    const summary = summarizeBundle(result);
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
      summary,
      matchScore,
      exactMatch: matchScore >= 0.999,
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
        policy.importDocument(fixture.rawDocument);
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
  const source = String(policy.importDocument);
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
    supportsCanonicalEnvelope: 2,
    supportsLinkedCaptureFallback: 3,
    supportsWrapperAliases: 2.5,
    supportsSnapshotAliases: 2.5
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

export function buildLocalizationReviewBundleImportExperimentReport({
  repetitions = 200
} = {}) {
  const fixtures = buildFixtures();
  const policyReports = EXPERIMENT_LOCALIZATION_REVIEW_BUNDLE_IMPORT_POLICIES.map((policy) =>
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
    type: 'localization-review-bundle-import-experiment-report',
    createdAt: new Date().toISOString(),
    problem: {
      name: 'localization-review-bundle-import',
      statement:
        'Import localization review bundles without freezing one document shape for canonical exports, linked-capture fallback, and wrapper-friendly sharing flows.',
      stableInterface:
        'importLocalizationReviewBundleDocument(rawDocument, policy?)'
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
  const report = buildLocalizationReviewBundleImportExperimentReport({
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
