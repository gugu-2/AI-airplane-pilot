import { useState, useEffect, useRef } from 'react';
import { Plane, Battery, ShieldAlert, Cpu, Activity, Signal, Eye, Navigation, Crosshair, Radio, Database, History, Video, CloudRain, Wind } from 'lucide-react';
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
  const [swarmDrones, setSwarmDrones] = useState({});
  const [atcInput, setAtcInput] = useState('');
  const [atcLogs, setAtcLogs] = useState([]);
  
  // View State
  const [currentView, setCurrentView] = useState('LIVE'); // 'LIVE' or 'HISTORY'
  const [flightLogs, setFlightLogs] = useState([]);
  
  // Aviation Awareness
  const [adsbTraffic, setAdsbTraffic] = useState([]);
  const [weather, setWeather] = useState({ condition: 'AWAITING', wind_speed_knots: 0, wind_dir_deg: 0 });
  const [tcasAlert, setTcasAlert] = useState(false);
  
  const wsRef = useRef(null);

  // Connect to WebSockets
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8765');
    wsRef.current = ws;

    ws.onopen = () => {
      // A1 FIX: Server now requires auth token as first message (Fix R6).
      // Key must match AEGIS_WS_KEY env variable on the server (default: 'aegis-local-dev-key').
      ws.send(JSON.stringify({ type: 'AUTH', key: 'aegis-local-dev-key' }));
      setConnected(true);
      addLog('Uplink established. Receiving live telemetry.', 'info');
      addAtcLog('ATC Console connected to VHF frequency.', 'sys');
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
      } else if (msg.type === 'VHF_READBACK') {
        addAtcLog(`Aegis 1: ${msg.readback}`, 'rx');
        addLog(`ATC Intent Parsed: ${msg.intent}`, 'info');
      } else if (msg.type === 'swarm_telemetry') {
        setSwarmDrones(prev => ({
          ...prev,
          [msg.data.drone_id]: {
            lat: msg.data.lat,
            lon: msg.data.lon,
            alt: msg.data.alt
          }
        }));
      } else if (msg.type === 'FLIGHT_LOGS_DATA') {
        setFlightLogs(msg.data);
      } else if (msg.type === 'aviation_awareness') {
        setAdsbTraffic(msg.data.traffic);
        setWeather(msg.data.weather);
        if (msg.data.tcas_alert) {
          setTcasAlert(true);
          setTimeout(() => setTcasAlert(false), 3000);
        }
      } else if (msg.type === 'ERROR') {
        // A1 FIX: Handle auth rejection from server
        addLog(`Server rejected connection: ${msg.msg}`, 'critical');
        setConnected(false);
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

  const addAtcLog = (msg, type) => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    setAtcLogs(prev => [{ time, msg, type }, ...prev].slice(0, 50));
  };

  const handleSendATC = (e) => {
    e.preventDefault();
    if (!atcInput.trim() || !wsRef.current) return;
    
    // Add to local UI
    addAtcLog(`Tower: ${atcInput}`, 'tx');
    
    // Send to drone AI Brain
    wsRef.current.send(JSON.stringify({ type: "VHF_RADIO", text: atcInput }));
    setAtcInput('');
  };

  const fetchFlightLogs = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'GET_FLIGHT_LOGS' }));
    }
  };

  useEffect(() => {
    if (currentView === 'HISTORY') {
      fetchFlightLogs();
    }
  }, [currentView]);

  return (
    <div className="dashboard-container">
      
      {/* Header */}
      <header className="header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div className="brand" style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          <Plane size={24} />
          <span>Aegis Flight Command</span>
          
          <div style={{ display: 'flex', gap: '10px', marginLeft: '30px' }}>
            <button 
              onClick={() => setCurrentView('LIVE')}
              style={{ background: currentView === 'LIVE' ? 'var(--accent-primary)' : 'transparent', color: currentView === 'LIVE' ? '#000' : 'var(--text-main)', border: '1px solid var(--accent-primary)', padding: '5px 15px', borderRadius: '4px', cursor: 'pointer', display: 'flex', gap: '5px', alignItems: 'center' }}
            >
              <Video size={14} /> LIVE COMMAND
            </button>
            <button 
              onClick={() => setCurrentView('HISTORY')}
              style={{ background: currentView === 'HISTORY' ? 'var(--accent-primary)' : 'transparent', color: currentView === 'HISTORY' ? '#000' : 'var(--text-main)', border: '1px solid var(--accent-primary)', padding: '5px 15px', borderRadius: '4px', cursor: 'pointer', display: 'flex', gap: '5px', alignItems: 'center' }}
            >
              <History size={14} /> FLIGHT LOGS
            </button>
          </div>
        </div>
        <div className="status-indicator">
          <span className={`dot ${connected ? 'live' : ''}`}></span>
          {connected ? 'Uplink Secure' : 'Connecting...'}
        </div>
      </header>

      {currentView === 'LIVE' ? (
        <>
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
            <div className="panel">
              <div className="panel-title">
                <CloudRain size={16} /> Aviation Weather
              </div>
              <div className="data-row">
                <span className="data-label">CONDITIONS</span>
                <span className="data-value mono" style={{ color: weather.condition === 'STORM' ? '#ff3366' : '#00ffcc' }}>{weather.condition}</span>
              </div>
              <div className="data-row">
                <span className="data-label"><Wind size={14} className="inline mr-2"/> WIND SPEED</span>
                <span className="data-value mono">{weather.wind_speed_knots} KT</span>
              </div>
              <div className="data-row">
                <span className="data-label">HEADING</span>
                <span className="data-value mono">{weather.wind_dir_deg}°</span>
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

            {/* ATC Radio Console at bottom of center */}
            <div className="panel" style={{ marginTop: '15px', height: '250px', display: 'flex', flexDirection: 'column' }}>
              <div className="panel-title">
                <Radio size={16} /> VHF Radio Communications
              </div>
              <div className="event-log mono" style={{ flex: 1, backgroundColor: '#050a0f', border: '1px solid #1a2a3a', padding: '10px', marginBottom: '10px' }}>
                {atcLogs.map((log, i) => (
                  <div key={i} style={{ marginBottom: '5px', color: log.type === 'tx' ? '#00ffcc' : log.type === 'rx' ? '#ffcc00' : '#888' }}>
                    <span style={{ opacity: 0.6, marginRight: '8px' }}>[{log.time}]</span>
                    {log.msg}
                  </div>
                ))}
              </div>
              <form onSubmit={handleSendATC} style={{ display: 'flex', gap: '10px' }}>
                <input 
                  type="text" 
                  value={atcInput}
                  onChange={(e) => setAtcInput(e.target.value)}
                  placeholder="Type ATC clearance (e.g. 'Aegis 1, abort landing')..." 
                  style={{ flex: 1, background: '#0a1018', border: '1px solid var(--accent-primary)', color: '#fff', padding: '10px', fontFamily: 'monospace' }}
                  disabled={!connected}
                />
                <button type="submit" disabled={!connected} style={{ background: 'var(--accent-primary)', color: '#000', border: 'none', padding: '0 20px', fontWeight: 'bold', cursor: 'pointer' }}>
                  TRANSMIT
                </button>
              </form>
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
                    
                    {/* Swarm Drones (Bravo, Charlie, etc) */}
                    {Object.entries(swarmDrones).map(([id, drone]) => (
                      <div key={id} style={{
                        position: 'absolute',
                        top: `${50 + (47.397742 - drone.lat) * 500000}%`,
                        left: `${50 + (drone.lon - 8.545594) * 500000}%`,
                        width: '8px', height: '8px', background: 'var(--accent-success)',
                        borderRadius: '50%', transform: 'translate(-50%, -50%)', zIndex: 11,
                        boxShadow: '0 0 10px var(--accent-success)'
                      }}>
                        <span style={{ position: 'absolute', top: '10px', left: '-10px', color: '#fff', fontSize: '0.6rem', fontFamily: 'monospace' }}>
                        <span style={{ position: 'absolute', top: '10px', left: '-10px', color: '#fff', fontSize: '0.6rem', fontFamily: 'monospace' }}>
                          {id}
                        </span>
                      </div>
                    ))}
                    
                    {/* ADS-B Commercial Traffic */}
                    {adsbTraffic.map((ac, i) => (
                      <div key={i} style={{
                        position: 'absolute',
                        top: `${50 + (47.397742 - ac.lat) * 500000}%`,
                        left: `${50 + (ac.lon - 8.545594) * 500000}%`,
                        width: '0', height: '0', 
                        borderLeft: '5px solid transparent',
                        borderRight: '5px solid transparent',
                        borderBottom: '10px solid #ff3366',
                        transform: `translate(-50%, -50%) rotate(${ac.heading}deg)`,
                        zIndex: 12,
                        filter: tcasAlert ? 'drop-shadow(0 0 8px #ff3366)' : 'none'
                      }}>
                        <span style={{ position: 'absolute', top: '15px', left: '-15px', color: '#ff3366', fontSize: '0.6rem', fontFamily: 'monospace', whiteSpace: 'nowrap' }}>
                          {ac.callsign} ({ac.alt}m)
                        </span>
                      </div>
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
        </>
      ) : (
        /* MISSION HISTORY VIEW */
        <main className="history-view" style={{ flex: 1, padding: '20px', overflowY: 'auto' }}>
          <div className="panel" style={{ minHeight: '80vh' }}>
            <div className="panel-title" style={{ marginBottom: '20px', fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Database size={20} /> Aegis Flight Blackbox Recorder
              <button onClick={fetchFlightLogs} style={{ marginLeft: 'auto', background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', color: '#fff', padding: '5px 15px', cursor: 'pointer' }}>
                Refresh Logs
              </button>
            </div>
            
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--accent-primary)' }}>
                  <th style={{ padding: '12px' }}>MISSION ID</th>
                  <th style={{ padding: '12px' }}>START TIME</th>
                  <th style={{ padding: '12px' }}>STATUS</th>
                  <th style={{ padding: '12px' }}>DISTANCE</th>
                  <th style={{ padding: '12px' }}>OBSTACLES</th>
                  <th style={{ padding: '12px' }}>EVASIONS</th>
                  <th style={{ padding: '12px' }}>ERRORS</th>
                </tr>
              </thead>
              <tbody className="mono" style={{ fontSize: '0.9rem' }}>
                {flightLogs.map((log, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #1a2a3a' }}>
                    <td style={{ padding: '12px' }}>{log.mission_id}</td>
                    <td style={{ padding: '12px' }}>{log.start_time}</td>
                    <td style={{ padding: '12px' }}>
                      <span style={{ 
                        padding: '3px 8px', borderRadius: '12px', fontSize: '0.75rem', fontWeight: 'bold',
                        background: log.status === 'SUCCESS' ? 'rgba(0, 255, 204, 0.2)' : 
                                    log.status === 'IN_PROGRESS' ? 'rgba(255, 204, 0, 0.2)' : 'rgba(255, 51, 102, 0.2)',
                        color: log.status === 'SUCCESS' ? '#00ffcc' : 
                               log.status === 'IN_PROGRESS' ? '#ffcc00' : '#ff3366'
                      }}>
                        {log.status}
                      </span>
                    </td>
                    <td style={{ padding: '12px' }}>{log.distance ? log.distance.toFixed(1) : 0} m</td>
                    <td style={{ padding: '12px', color: log.obstacles > 0 ? '#ffcc00' : 'inherit' }}>{log.obstacles}</td>
                    <td style={{ padding: '12px', color: log.evasions > 0 ? '#ffcc00' : 'inherit' }}>{log.evasions}</td>
                    <td style={{ padding: '12px', color: log.errors > 0 ? '#ff3366' : 'inherit' }}>{log.errors}</td>
                  </tr>
                ))}
                {flightLogs.length === 0 && (
                  <tr>
                    <td colSpan="7" style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)' }}>
                      No flight logs found in database.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </main>
      )}

    </div>
  );
}

export default App;
