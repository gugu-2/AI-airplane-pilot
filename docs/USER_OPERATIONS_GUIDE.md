# Aegis Flight OS — User Operations & GCS Manual
This guide provides operators, pilots, and air traffic controllers (ATC) with instructions on running missions, operating the Ground Control Station (GCS) dashboard, communicating with the AI pilot over VHF radio, and accessing flight telemetry databases.

---

## 1. Operating the Ground Control Station (GCS)

The Ground Control Station is a high-performance web dashboard showing the live state of the drone and the surrounding airspace.

```
+--------------------------------------------------------------------------------+
|  [AEGIS GROUND CONTROL]                      Uplink Status: [CONNECTED]        |
+--------------------------------------------------------------------------------+
|  [ AIRSPACE MAP ]                     |  [ LIVE TELEMETRY PANELS ]             |
|                                       |  Altitude:  488.0 m    Airspeed: 15.2m/s |
|     (Drone Position)                  |  Battery:   14.8 V     Mode:     MISSION |
|     * Swarm Drone #2                  |  CPU Load:  32 %       EKF:      ACTIVE  |
|     ! Obstacle Detected               |----------------------------------------|
|                                       |  [ TCAS TRAFFIC RADAR ]                |
|                                       |  * Intruder AGL122 (1.2km) - DESCEND   |
+--------------------------------------------------------------------------------+
|  [ VHF RADIO CONSOLE ]                                                        |
|  Tower: "Aegis 1, climb and maintain 100"                                      |
|  Aegis 1: "Aegis 1, climbing to 100."                                          |
|  [Send ATC command: [                                                ] [Send] ]|
+--------------------------------------------------------------------------------+
```

### Establishing the Uplink
1. Launch the backend server: `python src/main_pilot.py`
2. Start the dashboard front-end: `cd dashboard && npm run dev`
3. Open a browser to `http://localhost:5173`.
4. The GCS will automatically establish a secure WebSocket uplink. 
   - *Note: In production environments, the dashboard authenticates with a secure key using `AEGIS_WS_KEY`. Ensure this matches the server's key.*

---

## 2. Air Traffic Control (ATC) Voice commands

The GCS contains a virtual **VHF Radio Console** that connects directly to the drone's **Cognitive NLP Agent**. The AI pilot parses commands following standard **ICAO / FAA aviation phraseology** and automatically reads back clearances before executing maneuvers.

### Standard Aviation Command List

| Command Class | Operator Phraseology (Tower Command) | AI Pilot Readback (Drone Action) |
|---|---|---|
| **Takeoff** | `"Aegis 1, cleared for takeoff"` | `"Aegis 1, cleared for takeoff."` (Drone arms and ascends to hover altitude) |
| **Landing** | `"Aegis 1, cleared to land runway 24"` | `"Aegis 1, cleared to land."` (Drone initiates autoland at current position) |
| **Hold** | `"Aegis 1, hold position immediately"` | `"Aegis 1, holding position."` (Drone cancels active waypoint and hovers in place) |
| **Altitude Change** | `"Aegis 1, climb and maintain 100"` | `"Aegis 1, climbing to 100."` (Drone moves vertically to 100 feet / 30 meters) |
| **Abort / Go-Around** | `"Aegis 1, traffic on runway, go around!"` | `"Aegis 1, going around."` (Drone instantly aborts descent, climbs to safety altitude) |

---

## 3. Emergency Safe-States & Fail-Safes

Aegis Flight OS continuously monitors flight parameters. If any safety boundary is violated, it overrides pilot or mission control commands to protect the aircraft.

### 1. Dynamic Geofence Violation
- **Trigger**: The drone exceeds `MAX_CEILING_M` (120m), flies below `MIN_ALTITUDE_M` (2m), or travels outside the dynamic lateral geofence radius (relative to takeoff GPS lock).
- **Behavior**: The **Envelope Protection System** overrides the command, clamps the coordinates to the boundary, and forces the drone back inside the safe envelope.

### 2. TCAS Immediate Evasion
- **Trigger**: The ADS-B sensor registers an aircraft within **200 meters** horizontally and **50 meters** vertically.
- **Behavior**: The mission loop is bypassed instantly. The drone executes an emergency vertical descent/climb to resolve the conflict.

### 3. Severe Weather RTL
- **Trigger**: Wind speed exceeds **25 Knots** or sensor reads sudden storm indicator.
- **Behavior**: Telemetry logs a critical alarm. The drone ignores current waypoints and initiates an autonomous **Return-to-Launch (RTL)** to land safely at takeoff coordinates.

---

## 4. Extracting and Analyzing Flight Logs

Aegis records every sensor reading, actuator command, and safety alarm into a high-integrity dual logging pipeline.

### Querying the SQLite Black Box
The main logger writes to `flight_data_recorder.db`. You can query this database using standard SQL:

```bash
sqlite3 flight_data_recorder.db
```

#### Querying Telemetry History
```sql
SELECT timestamp, latitude, longitude, altitude, battery_v, message 
FROM telemetry 
ORDER BY timestamp DESC 
LIMIT 10;
```

#### Checking Evasions and Obstacles Encountered
```sql
SELECT status, total_distance_m, obstacles_encountered, evasions_triggered, errors_warnings
FROM flight_summary;
```

### Inspecting Rotating Text Logs
If database locks occur or write latency rises, a text-based logger writes continuously to:
`logs/aegis_flight.log`

This log rotates automatically when it exceeds **10MB** and maintains 5 archived rotations. You can view live logs from a terminal using:
```bash
tail -f logs/aegis_flight.log
```
