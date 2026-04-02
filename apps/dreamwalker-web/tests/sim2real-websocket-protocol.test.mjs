import test from 'node:test';
import assert from 'node:assert/strict';

import {
  importSim2realWebsocketMessage,
  sim2realQueryProtocolId
} from '../src/sim2real-websocket-protocol.js';
import { buildSim2realWebsocketProtocolExperimentReport } from '../tools/sim2real-websocket-protocol-lab.mjs';

test('alias friendly sim2real websocket importer accepts ready wrappers with field aliases', () => {
  const message = importSim2realWebsocketMessage({
    protocol: sim2realQueryProtocolId,
    messageType: 'ready',
    server: {
      queryTransport: 'ws',
      url: 'ws://127.0.0.1:8781/sim2real',
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
  });

  assert.equal(message.type, 'query-ready');
  assert.equal(message.transport, 'ws');
  assert.equal(message.endpoint, 'ws://127.0.0.1:8781/sim2real');
  assert.deepEqual(message.requestTypes, ['render']);
  assert.equal(message.defaults.width, 320);
});

test('sim2real websocket protocol lab compares three policies and keeps alias friendly as best fit', () => {
  const report = buildSim2realWebsocketProtocolExperimentReport({ repetitions: 4 });

  assert.equal(report.type, 'sim2real-websocket-protocol-experiment-report');
  assert.ok(report.fixtures.length >= 5);
  assert.ok(report.policies.length >= 3);
  const policyNames = new Set(report.policies.map((policy) => policy.name));
  assert.ok(policyNames.has('strict_canonical'));
  assert.ok(policyNames.has('envelope_first'));
  assert.ok(policyNames.has('alias_friendly'));
  assert.equal(report.highlights.bestFit.policy, 'alias_friendly');
});
