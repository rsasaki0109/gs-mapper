import { useEffect, useMemo, useState } from 'react';
import { ObsOverlayView } from './overlay-ui.jsx';
import {
  buildRelayEndpoint,
  loadOverlayState,
  overlayStateKey,
  parseOverlayRelayConfigFromSearch
} from './overlay-shared.js';

export default function OverlayApp() {
  const relayConfig = useMemo(() => parseOverlayRelayConfigFromSearch(), []);
  const [overlayState, setOverlayState] = useState(loadOverlayState);

  useEffect(() => {
    setOverlayState(loadOverlayState());

    function handleStorage(event) {
      if (event.key !== overlayStateKey) {
        return;
      }

      try {
        setOverlayState(event.newValue ? JSON.parse(event.newValue) : null);
      } catch {
        setOverlayState(null);
      }
    }

    window.addEventListener('storage', handleStorage);
    let eventSource = null;
    let handleOverlayEvent = null;

    if (relayConfig.enabled) {
      try {
        eventSource = new EventSource(buildRelayEndpoint(relayConfig.url, '/events'));
        handleOverlayEvent = (event) => {
          try {
            setOverlayState(event.data ? JSON.parse(event.data) : null);
          } catch {
            setOverlayState(null);
          }
        };
        eventSource.addEventListener('overlay', handleOverlayEvent);
      } catch {
        eventSource = null;
      }
    }

    return () => {
      window.removeEventListener('storage', handleStorage);
      if (eventSource) {
        if (handleOverlayEvent) {
          eventSource.removeEventListener('overlay', handleOverlayEvent);
        }
        eventSource.close();
      }
    };
  }, [relayConfig.enabled, relayConfig.url]);

  return <ObsOverlayView overlayState={overlayState} />;
}
