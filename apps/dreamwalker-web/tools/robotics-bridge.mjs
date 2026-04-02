import http from 'node:http';
import { WebSocketServer } from 'ws';
import {
  normalizeRobotBridgeMessage,
  robotBridgeProtocolId,
  robotBridgeServerSource,
  stringifyRobotBridgeMessage
} from '../src/robotics-bridge.js';

const port = Number(process.env.DREAMWALKER_ROBOTICS_BRIDGE_PORT || 8790);
const host = process.env.DREAMWALKER_ROBOTICS_BRIDGE_HOST || '127.0.0.1';
const clients = new Map();
let nextClientId = 1;

function sendJson(response, statusCode, body) {
  response.writeHead(statusCode, {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'GET,OPTIONS',
    'Content-Type': 'application/json; charset=utf-8'
  });
  response.end(JSON.stringify(body));
}

const server = http.createServer((request, response) => {
  if (request.method === 'OPTIONS') {
    response.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Access-Control-Allow-Methods': 'GET,OPTIONS'
    });
    response.end();
    return;
  }

  if (request.url === '/health') {
    sendJson(response, 200, {
      ok: true,
      clients: clients.size,
      protocol: robotBridgeProtocolId
    });
    return;
  }

  sendJson(response, 404, {
    ok: false,
    error: 'Not found'
  });
});

const wss = new WebSocketServer({ noServer: true });

function broadcast(sender, payload) {
  for (const client of wss.clients) {
    if (client === sender || client.readyState !== 1) {
      continue;
    }

    client.send(payload);
  }
}

wss.on('connection', (socket) => {
  const clientId = `robot-bridge-${nextClientId.toString(36)}`;
  nextClientId += 1;
  clients.set(socket, clientId);

  socket.send(
    stringifyRobotBridgeMessage(
      'bridge-ready',
      {
        clientId
      },
      {
        source: robotBridgeServerSource
      }
    )
  );

  socket.on('message', (payload, isBinary) => {
    if (isBinary) {
      broadcast(socket, payload);
      return;
    }

    try {
      const normalizedPayload = JSON.stringify(
        normalizeRobotBridgeMessage(payload.toString(), {
          fallbackSource: clientId
        })
      );
      broadcast(socket, normalizedPayload);
    } catch (error) {
      socket.send(
        stringifyRobotBridgeMessage(
          'bridge-error',
          {
            error: error instanceof Error ? error.message : String(error)
          },
          {
            source: robotBridgeServerSource
          }
        )
      );
    }
  });

  socket.on('close', () => {
    clients.delete(socket);
  });
});

server.on('upgrade', (request, socket, head) => {
  const requestUrl = new URL(request.url ?? '/', `http://${request.headers.host ?? `${host}:${port}`}`);

  if (requestUrl.pathname !== '/robotics') {
    socket.destroy();
    return;
  }

  wss.handleUpgrade(request, socket, head, (websocket) => {
    wss.emit('connection', websocket, request);
  });
});

server.listen(port, host, () => {
  console.log(`[dreamwalker-robotics-bridge] listening on ws://${host}:${port}/robotics`);
});
