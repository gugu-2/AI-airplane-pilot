import { useState, useEffect } from 'react';
import { Plane, Battery, ShieldAlert, Cpu, Activity, Signal, Eye, Navigation, Crosshair } from 'lucide-react';
import './index.css';

function App() {
  const [connected, setConnected] = useState(false);
  const [telemetry, setTelemetry] = useState({
    lat: 47.397742,
    lon: 8.545594,
    alt: 488.0,
    battery: 15.2,
    mode: 'STANDBY',
    cpu: 45,
  });
  const [logs, setLogs] = useState([
    { time: '02:59:01', msg: 'System initialized. Waiting for drone connection...', type: 'info' }
  ]);
  const [objects, setObjects] = useState([]);
  const [mapObstacles, setMapObstacles] = useState([]);

  // Connect to WebSockets
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8765');

    ws.onopen = () => {
      setConnected(true);
      addLog('Uplink established. Receiving live telemetry.', 'info');
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'telemetry') {
        setTelemetry(msg.data);
      } else if (msg.type === 'vision') {
        setObjects(msg.data);
      } else if (msg.type === 'log') {
        addLog(msg.data.msg, msg.data.level);
      } else if (msg.type === 'map_state') {
        setMapObstacles(msg.data);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      addLog('Uplink lost. Attempting to reconnect...', 'critical');
    };

    return () => ws.close();
  }, []);

  const addLog = (msg, type) => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    setLogs(prev => [{ time, msg, type }, ...prev].slice(0, 50));
  };

  return (
    <div className="dashboard-container">
      
      {/* Header */}
      <header className="header">
        <div className="brand">
          <Plane size={24} />
          <span>Aegis Flight Command</span>
        </div>
        <div className="status-indicator">
          <span className={`dot ${connected ? 'live' : ''}`}></span>
          {connected ? 'Uplink Secure' : 'Connecting...'}
        </div>
      </header>

      {/* Left Sidebar - Telemetry */}
      <aside className="telemetry-sidebar">
        <div className="panel">
          <div className="panel-title">
            <Activity size={16} /> Flight Telemetry
          </div>
          <div className="data-row">
            <span className="data-label">LATITUDE</span>
            <span className="data-value mono">{telemetry.lat.toFixed(6)}</span>
          </div>
          <div className="data-row">
            <span className="data-label">LONGITUDE</span>
            <span className="data-value mono">{telemetry.lon.toFixed(6)}</span>
          </div>
          <div className="data-row">
            <span className="data-label">ALTITUDE</span>
            <span className="data-value mono">{telemetry.alt.toFixed(1)} m</span>
          </div>
          <div className="data-row">
            <span className="data-label">FLIGHT MODE</span>
            <span className="data-value mono">{telemetry.mode}</span>
          </div>
        </div>

        <div className="panel">
          <div className="panel-title">
            <Cpu size={16} /> System Health
          </div>
          <div className="data-row">
            <span className="data-label"><Battery size={14} className="inline mr-2"/> BATTERY</span>
            <span className="data-value mono">{telemetry.battery.toFixed(1)} V</span>
          </div>
          <div className="data-row">
            <span className="data-label"><Signal size={14} className="inline mr-2"/> SIGNAL</span>
            <span className="data-value mono">{connected ? '98%' : '0%'}</span>
          </div>
          <div className="data-row">
            <span className="data-label">EDGE COMPUTE</span>
            <span className="data-value mono">{telemetry.cpu}% LOAD</span>
          </div>
        </div>
      </aside>

      {/* Center - Visuals */}
      <main className="center-feed">
        <div className="panel" style={{ flex: 1, padding: 0 }}>
          <div className="panel-title" style={{ padding: '15px 15px 0 15px', position: 'absolute', zIndex: 10 }}>
            <Eye size={16} /> Primary Optics (YOLOv8)
          </div>
          <div className="video-feed-container">
            {!connected ? (
              <div className="no-signal">NO VIDEO FEED</div>
            ) : (
              <>
                <div className="crosshair"></div>
                {/* Mocking a video feed bounding box */}
                {objects.map((obj, i) => (
                  <div key={i} style={{
                    position: 'absolute',
                    border: '2px solid var(--accent-warning)',
                    width: '150px', height: '200px',
                    top: '30%', left: '40%',
                    display: 'flex', flexDirection: 'column'
                  }}>
                    <span style={{
                      background: 'var(--accent-warning)', color: '#000', 
                      fontSize: '0.7rem', fontWeight: 'bold', padding: '2px 5px', width: 'fit-content'
                    }}>
                      {obj.class} (ID:{obj.id}) {(obj.conf * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </>
            )}
          </div>
        </div>
      </main>

      {/* Right Sidebar - Mapping & Logs */}
      <aside className="intelligence-sidebar">
        <div className="panel">
          <div className="panel-title">
            <Navigation size={16} /> Semantic Memory Map
          </div>
          <div className="semantic-map">
            {!connected ? (
              <span className="mono" style={{ color: 'var(--text-muted)' }}>Awaiting Grid Data...</span>
            ) : (
              <div style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden' }}>
                {/* Center dot (home) */}
                <div style={{
                  position: 'absolute', top: '50%', left: '50%', 
                  width: '8px', height: '8px', background: 'var(--accent-success)', 
                  borderRadius: '50%', transform: 'translate(-50%, -50%)', zIndex: 10
                }}></div>
                
                {/* Current Drone Position */}
                <div style={{
                  position: 'absolute',
                  top: `${50 + (47.397742 - telemetry.lat) * 500000}%`,
                  left: `${50 + (telemetry.lon - 8.545594) * 500000}%`,
                  width: '8px', height: '8px', background: 'var(--accent-primary)',
                  borderRadius: '50%', transform: 'translate(-50%, -50%)', zIndex: 11,
                  boxShadow: '0 0 10px var(--accent-primary)'
                }}></div>

                {/* Map Obstacles */}
                {mapObstacles.map((obs, i) => (
                  <div key={i} style={{
                    position: 'absolute',
                    top: `${50 + (47.397742 - obs.lat) * 500000}%`,
                    left: `${50 + (obs.lon - 8.545594) * 500000}%`,
                    width: '8px', height: '8px', background: 'var(--accent-danger)',
                    transform: 'translate(-50%, -50%)'
                  }}></div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="panel" style={{ flex: 1 }}>
          <div className="panel-title">
            <ShieldAlert size={16} /> System Event Log
          </div>
          <div className="event-log mono">
            {logs.map((log, i) => (
              <div key={i} className={`log-entry ${log.type === 'warning' ? 'log-warning' : ''} ${log.type === 'critical' ? 'log-critical' : ''}`}>
                <span className="log-time">[{log.time}]</span>
                {log.msg}
              </div>
            ))}
          </div>
        </div>
      </aside>

    </div>
  );
}

export default App;
