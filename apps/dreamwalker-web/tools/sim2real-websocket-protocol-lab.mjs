import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

import {
  EXPERIMENT_SIM2REAL_WEBSOCKET_MESSAGE_POLICIES,
  importSim2realWebsocketMessage,
  localizationImageBenchmarkProtocolId,
  sim2realQueryProtocolId
} from '../src/sim2real-websocket-protocol.js';

function meanOrNull(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return null;
  }

  return values.reduce((sum, value) => sum + Number(value), 0) / values.length;
}

function summarizeMessage(message) {
  if (!message || typeof message !== 'object') {
    return {
      type: 'none'
    }
  }

  if (message.type === 'query-ready') {
    return {
      type: 'query-ready',
      transport: message.transport,
      endpoint: message.endpoint,
      width: message.defaults?.width ?? null,
      requestTypes: Array.isArray(message.requestTypes) ? [...message.requestTypes] : []
    };
  }

  if (message.type === 'render-result') {
    return {
      type: 'render-result',
      width: message.width ?? null,
      height: message.height ?? null,
      position: Array.isArray(message.pose?.position) ? [...message.pose.position] : null,
      hasColor: Boolean(message.colorJpegBase64),
      hasDepth: Boolean(message.depthBase64)
    };
  }

  if (message.type === 'localization-image-benchmark-report') {
    return {
      type: 'localization-image-benchmark-report',
      matchedCount: Number(message.matching?.matchedCount ?? 0),
      frameCount: Array.isArray(message.frames) ? message.frames.length : 0
    };
  }

  return {
    type: 'error',
    error: message.error ?? ''
  };
}

function buildFixtures() {
  return [
    {
      fixtureId: 'canonical-query-ready',
      label: 'Canonical Query Ready',
      intent: 'Keep the current websocket handshake stable for browser clients.',
      rawMessage: {
        protocol: sim2realQueryProtocolId,
        type: 'query-ready',
        transport: 'ws',
        endpoint: 'ws://127.0.0.1:8781/sim2real',
        frameId: 'dreamwalker_map',
        renderer: 'gsplat',
        rendererReason: 'auto-selected',
        requestTypes: ['render', 'localization-image-benchmark'],
        defaults: {
          width: 640,
          height: 480,
          fovDegrees: 60,
          nearClip: 0.05,
          farClip: 50,
          pointRadius: 1
        }
      },
      expectedSummary: {
        type: 'query-ready',
        transport: 'ws',
        endpoint: 'ws://127.0.0.1:8781/sim2real',
        width: 640,
        requestTypes: ['render', 'localization-image-benchmark']
      }
    },
    {
      fixtureId: 'wrapped-render-result',
      label: 'Wrapped Render Result',
      intent: 'Accept envelope wrappers that keep canonical render-result fields inside result payloads.',
      rawMessage: {
        type: 'render-result',
        result: {
          protocol: sim2realQueryProtocolId,
          frameId: 'dreamwalker_map',
          width: 64,
          height: 48,
          fovDegrees: 60,
          nearClip: 0.05,
          farClip: 50,
          pointRadius: 1,
          pose: {
            position: [1, 2, 3],
            orientation: [0, 0, 0, 1]
          },
          cameraInfo: {
            frameId: 'dreamwalker_map',
            width: 64,
            height: 48
          },
          colorJpegBase64: 'ZmFrZS1qcGVn',
          depthBase64: 'AAAAAA=='
        }
      },
      expectedSummary: {
        type: 'render-result',
        width: 64,
        height: 48,
        position: [1, 2, 3],
        hasColor: true,
        hasDepth: true
      }
    },
    {
      fixtureId: 'alias-ready-wrapper',
      label: 'Alias Ready Wrapper',
      intent: 'Support SDK-style ready messages that rename type and defaults fields.',
      rawMessage: {
        protocol: sim2realQueryProtocolId,
        messageType: 'ready',
        server: {
          queryTransport: 'ws',
          url: 'ws://127.0.0.1:8781/sim2real',
          mapFrameId: 'dreamwalker_map',
          backend: 'simple',
          backendReason: 'fallback',
          supportedRequests: ['render'],
          queryDefaults: {
            imageWidth: 320,
            imageHeight: 240,
            fov: 55,
            near: 0.1,
            far: 25,
            radius: 2
          }
        }
      },
      expectedSummary: {
        type: 'query-ready',
        transport: 'ws',
        endpoint: 'ws://127.0.0.1:8781/sim2real',
        width: 320,
        requestTypes: ['render']
      }
    },
    {
      fixtureId: 'wrapped-benchmark-report',
      label: 'Wrapped Benchmark Report',
      intent: 'Allow benchmark responses to live under report wrappers while keeping the canonical report body.',
      rawMessage: {
        type: 'localization-image-benchmark-report',
        report: {
          protocol: localizationImageBenchmarkProtocolId,
          matching: {
            matchedCount: 4
          },
          frames: [{ frameIndex: 0 }, { frameIndex: 1 }],
          metrics: {
            summary: {
              lpips: {
                mean: 0.12
              }
            }
          }
        }
      },
      expectedSummary: {
        type: 'localization-image-benchmark-report',
        matchedCount: 4,
        frameCount: 2
      }
    },
    {
      fixtureId: 'error-alias',
      label: 'Error Alias',
      intent: 'Accept thin-tooling error messages that use kind/detail instead of the canonical error envelope.',
      rawMessage: {
        kind: 'err',
        detail: 'query timed out'
      },
      expectedSummary: {
        type: 'error',
        error: 'query timed out'
      }
    }
  ];
}

function evaluateFixture(policy, fixture) {
  const startedAt = performance.now();

  try {
    const message = policy.importMessage(fixture.rawMessage);
    const runtimeMs = performance.now() - startedAt;
    const summary = summarizeMessage(message);
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
        policy.importMessage(fixture.rawMessage);
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
    supportsCanonicalEnvelope: 2,
    supportsNestedWrappers: 2.5,
    supportsTypeAliases: 2.5,
    supportsFieldAliases: 3
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

export function buildSim2realWebsocketProtocolExperimentReport({
  repetitions = 200
} = {}) {
  const fixtures = buildFixtures();
  const policyReports = EXPERIMENT_SIM2REAL_WEBSOCKET_MESSAGE_POLICIES.map((policy) =>
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
    type: 'sim2real-websocket-protocol-experiment-report',
    createdAt: new Date().toISOString(),
    problem: {
      name: 'sim2real-websocket-protocol',
      statement:
        'Import sim2real websocket envelopes without freezing one message shape for canonical server responses, wrapped browser adapters, and thin tooling aliases.',
      stableInterface:
        'importSim2realWebsocketMessage(rawMessage, policy?)'
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
  const report = buildSim2realWebsocketProtocolExperimentReport({
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
