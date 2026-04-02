import test from 'node:test';
import assert from 'node:assert/strict';

import { importLiveLocalizationStreamMessage } from '../src/sim2real-live-localization-import.js';
import { buildLiveLocalizationStreamImportExperimentReport } from '../tools/live-localization-stream-import-lab.mjs';

test('alias friendly live importer accepts top-level pose shortcut append messages', () => {
  const result = importLiveLocalizationStreamMessage(
    null,
    {
      type: 'append',
      label: 'CLI Stream',
      position: [4, 5, 6],
      orientation: [0, 0, 0, 1],
      time: 3
    },
    {
      maxPoses: 240,
      defaultLabel: 'Live Localization Estimate'
    }
  );

  assert.equal(result.kind, 'append');
  assert.equal(result.estimate.label, 'CLI Stream');
  assert.equal(result.estimate.sourceType, 'live-stream');
  assert.equal(result.estimate.poses.length, 1);
  assert.deepEqual(result.estimate.poses[0].position, [4, 5, 6]);
});

test('live localization stream import lab compares three policies and keeps alias friendly as best fit', () => {
  const report = buildLiveLocalizationStreamImportExperimentReport({ repetitions: 4 });

  assert.equal(report.type, 'live-localization-stream-import-experiment-report');
  assert.ok(report.fixtures.length >= 5);
  assert.ok(report.policies.length >= 3);
  const policyNames = new Set(report.policies.map((policy) => policy.name));
  assert.ok(policyNames.has('strict_canonical'));
  assert.ok(policyNames.has('wrapped_pose'));
  assert.ok(policyNames.has('alias_friendly'));
  assert.equal(report.highlights.bestFit.policy, 'alias_friendly');
});
