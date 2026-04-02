import test from 'node:test';
import assert from 'node:assert/strict';

import { importLocalizationReviewBundleDocument } from '../src/sim2real-review-bundle-import.js';
import { buildLocalizationReviewBundleImportExperimentReport } from '../tools/localization-review-bundle-import-lab.mjs';

test('alias friendly localization review bundle importer recovers linked captures through wrappers', () => {
  const imported = importLocalizationReviewBundleDocument({
    reviewBundle: {
      protocol: 'dreamwalker-localization-review-bundle/v1',
      type: 'localization-review-bundle',
      createdAt: '2026-04-02T00:00:00.000Z',
      compareSelection: {
        runIds: ['run-a'],
        baselineId: 'run-a'
      },
      captures: [
        {
          captureSourceId: 'capture-shelf:linked-capture',
          label: 'Linked Capture',
          captureBundle: {
            type: 'route-capture-bundle',
            captures: [
              {
                pose: {
                  position: [1, 0, 0],
                  yawDegrees: 0
                },
                response: {
                  type: 'render-result',
                  width: 8,
                  height: 8,
                  colorJpegBase64: 'ZmFrZS1qcGVn',
                  depthBase64: 'AAAAAA=='
                }
              }
            ]
          }
        }
      ],
      portableRuns: [
        {
          runId: 'run-a',
          portableSnapshot: {
            runId: 'run-a',
            label: 'Run A',
            groundTruth: {
              captureSourceId: 'capture-shelf:linked-capture',
              label: 'Ground Truth Capture'
            },
            estimate: {
              type: 'localization-estimate',
              label: 'Run A',
              poses: [{ position: [0, 0, 0], orientation: [0, 0, 0, 1] }]
            }
          }
        }
      ]
    }
  });

  assert.equal(imported.type, 'localization-review-bundle');
  assert.equal(imported.baselineRunId, 'run-a');
  assert.equal(imported.captureShelfEntries.length, 1);
  assert.equal(imported.runShelfEntries.length, 1);
  assert.equal(imported.runShelfEntries[0].groundTruth.sourceId, 'capture-shelf:linked-capture');
  assert.equal(imported.runShelfEntries[0].groundTruth.bundle.type, 'route-capture-bundle');
});

test('localization review bundle import lab compares three policies and keeps alias friendly as best fit', () => {
  const report = buildLocalizationReviewBundleImportExperimentReport({ repetitions: 4 });

  assert.equal(report.type, 'localization-review-bundle-import-experiment-report');
  assert.ok(report.fixtures.length >= 3);
  assert.ok(report.policies.length >= 3);
  const policyNames = new Set(report.policies.map((policy) => policy.name));
  assert.ok(policyNames.has('strict_canonical'));
  assert.ok(policyNames.has('linked_capture_fallback'));
  assert.ok(policyNames.has('alias_friendly'));
  assert.equal(report.highlights.bestFit.policy, 'alias_friendly');
});
