import { createServer } from 'node:http';

const port = Number(process.env.DREAMWALKER_OVERLAY_RELAY_PORT || 8787);
const host = process.env.DREAMWALKER_OVERLAY_RELAY_HOST || '127.0.0.1';
const maxBodyBytes = 1024 * 1024;
const heartbeatIntervalMs = 15_000;

let latestState = null;
const clients = new Set();

function setCorsHeaders(response) {
  response.setHeader('Access-Control-Allow-Origin', '*');
  response.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  response.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
}

function writeJson(response, statusCode, payload) {
  setCorsHeaders(response);
  response.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-cache'
  });
  response.end(JSON.stringify(payload));
}

function writeSse(response, eventName, payload) {
  response.write(`event: ${eventName}\n`);
  for (const line of payload.split('\n')) {
    response.write(`data: ${line}\n`);
  }
  response.write('\n');
}

function broadcastOverlayState(serializedState) {
  for (const client of clients) {
    writeSse(client, 'overlay', serializedState);
  }
}

const server = createServer((request, response) => {
  const requestUrl = new URL(request.url || '/', `http://${request.headers.host || host}`);

  if (request.method === 'OPTIONS') {
    setCorsHeaders(response);
    response.writeHead(204);
    response.end();
    return;
  }

  if (request.method === 'GET' && requestUrl.pathname === '/health') {
    writeJson(response, 200, {
      ok: true,
      clients: clients.size,
      hasState: Boolean(latestState)
    });
    return;
  }

  if (request.method === 'GET' && requestUrl.pathname === '/state') {
    writeJson(response, 200, {
      state: latestState ? JSON.parse(latestState) : null
    });
    return;
  }

  if (request.method === 'GET' && requestUrl.pathname === '/events') {
    setCorsHeaders(response);
    response.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no'
    });
    response.write(': connected\n\n');

    if (latestState) {
      writeSse(response, 'overlay', latestState);
    }

    const heartbeatId = setInterval(() => {
      response.write(': heartbeat\n\n');
    }, heartbeatIntervalMs);

    clients.add(response);

    request.on('close', () => {
      clearInterval(heartbeatId);
      clients.delete(response);
      response.end();
    });

    return;
  }

  if (request.method === 'POST' && requestUrl.pathname === '/publish') {
    let requestBody = '';
    let didOverflow = false;

    request.setEncoding('utf8');
    request.on('data', (chunk) => {
      if (didOverflow) {
        return;
      }

      requestBody += chunk;
      if (requestBody.length > maxBodyBytes) {
        didOverflow = true;
        writeJson(response, 413, {
          error: 'Payload too large'
        });
        request.destroy();
      }
    });
    request.on('end', () => {
      if (didOverflow) {
        return;
      }

      try {
        const parsedState = JSON.parse(requestBody || '{}');
        latestState = JSON.stringify(parsedState);
        broadcastOverlayState(latestState);
        setCorsHeaders(response);
        response.writeHead(204);
        response.end();
      } catch {
        writeJson(response, 400, {
          error: 'Invalid JSON payload'
        });
      }
    });

    return;
  }

  writeJson(response, 404, {
    error: 'Not found'
  });
});

server.listen(port, host, () => {
  console.log(`[dreamwalker-overlay-relay] listening on http://${host}:${port}`);
});

function shutdown() {
  for (const client of clients) {
    client.end();
  }
  server.close(() => process.exit(0));
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
