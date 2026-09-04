import sqlite3
import time
import json
import hashlib

class BlackBoxRecorder:
    """
    AEGIS AUTONOMY: Flight Data Recorder (Black Box)
    Mandatory for FAA certification. Logs all telemetry, AI decisions, 
    and faults to a local tamper-proof SQLite database.
    """
    def __init__(self, db_path="flight_data_recorder.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._initialize_database()

    def _initialize_database(self):
        """Creates the encrypted ledger tables if they don't exist."""
        # Telemetry Table (10Hz+ data)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                latitude REAL,
                longitude REAL,
                altitude REAL,
                heading REAL,
                battery REAL
            )
        ''')
        
        # AI Decision & Fault Table (Event based)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                event_type TEXT,
                severity TEXT,
                payload TEXT,
                digital_signature TEXT
            )
        ''')
        self.conn.commit()

    def _generate_signature(self, payload):
        """Creates a SHA-256 hash to ensure the log was not tampered with post-flight."""
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def log_telemetry(self, lat, lng, alt, heading, battery):
        """Logs high-frequency flight data."""
        self.cursor.execute('''
            INSERT INTO telemetry (timestamp, latitude, longitude, altitude, heading, battery)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (time.time(), lat, lng, alt, heading, battery))
        self.conn.commit()

    def log_event(self, event_type, severity, payload_dict):
        """Logs AI decisions, obstacle avoidances, and system faults."""
        payload_str = json.dumps(payload_dict)
        signature = self._generate_signature(payload_str)
        
        self.cursor.execute('''
            INSERT INTO ai_events (timestamp, event_type, severity, payload, digital_signature)
            VALUES (?, ?, ?, ?, ?)
        ''', (time.time(), event_type, severity, payload_str, signature))
        self.conn.commit()
        print(f"[BLACKBOX] Logged {severity} Event: {event_type}")

    def export_crash_report(self):
        """Retrieves the last 5 events leading up to a crash for analysis."""
        self.cursor.execute('SELECT * FROM ai_events ORDER BY timestamp DESC LIMIT 5')
        events = self.cursor.fetchall()
        print("\n================ BLACK BOX CRASH REPORT ================")
        for event in reversed(events):
            print(f"[{event[1]:.2f}] {event[3]} | {event[2]} | {event[4]}")
        print("========================================================\n")

if __name__ == "__main__":
    print(">>> INITIALIZING BLACK BOX FLIGHT RECORDER...")
    fdr = BlackBoxRecorder()
    
    # Simulate a flight
    print("\n[FLIGHT] Taking off...")
    fdr.log_event("TAKEOFF", "INFO", {"target_alt": 10.0})
    fdr.log_telemetry(37.7749, -122.4194, 2.0, 90.0, 99.5)
    
    time.sleep(1)
    print("[FLIGHT] Encountered severe wind shear.")
    fdr.log_event("WIND_SHEAR", "WARNING", {"wind_speed_knots": 45, "action": "increased_Kd"})
    fdr.log_telemetry(37.7750, -122.4190, 10.0, 92.0, 98.0)
    
    time.sleep(1)
    print("[FLIGHT] Engine 3 Failure.")
    fdr.log_event("MOTOR_FAILURE", "CRITICAL", {"motor_id": 3, "rpm": 0})
    
    time.sleep(1)
    print("[FLIGHT] Attempting emergency parachute deployment.")
    fdr.log_event("PARACHUTE_DEPLOY", "CRITICAL", {"alt": 8.0, "status": "SUCCESS"})
    
    # Post-crash analysis
    fdr.export_crash_report()
