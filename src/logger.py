"""
Aegis OS \u2014 Flight Logger (Black Box)
Fix #7: Batch SQLite commits (no more per-row disk writes at 100Hz)
Fix #16: Thread-safe SQLite connection with Lock
Fix #27: Absolute DB path (no more relative path ambiguity)
"""
import sqlite3
import os
import threading
import math
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

class FlightLogger:
    # Fix #27: Compute absolute path relative to this file, not the launch CWD
    DEFAULT_DB_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'flight_data_recorder.db'
    )

    def __init__(self, db_path=None):
        """
        Initializes a SQLite-based 'Black Box' flight logger.
        B5 FIX: Also initializes a rotating text log file in the logs/ directory
                as a fallback in case the SQLite DB gets corrupted.
        """
        self.db_path = db_path or self.DEFAULT_DB_PATH
        
        # Fix #16: check_same_thread=False + Lock for async/threaded access safety
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self.cursor = self.conn.cursor()
        
        # Fix #7: Write queue for batching
        self._telemetry_queue = []
        self._hw_queue = []
        self.BATCH_SIZE = 50
        
        # B5 FIX: Set up a rotating text log file alongside SQLite
        _logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
        os.makedirs(_logs_dir, exist_ok=True)
        _log_file = os.path.join(_logs_dir, 'aegis_flight.log')
        
        self._text_logger = logging.getLogger('aegis_flight')
        self._text_logger.setLevel(logging.DEBUG)
        if not self._text_logger.handlers:  # Avoid duplicate handlers on re-init
            handler = RotatingFileHandler(
                _log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB per file
                backupCount=5               # Keep last 5 rotations = 50MB total
            )
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
            self._text_logger.addHandler(handler)
        
        self._text_logger.info(f'FlightLogger started. DB: {self.db_path}')
        
        self._create_tables()
        
        self.mission_id = "MISSION_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.obstacles_encountered = 0
        self.evasions_triggered = 0
        self.errors_warnings = 0
        self.total_distance = 0.0
        self.last_lat = None
        self.last_lon = None
        
        print(f"[Logger] SQLite Black Box: {os.path.abspath(self.db_path)} (Mission: {self.mission_id})")
        
        with self._lock:
            self.cursor.execute(
                'INSERT INTO flight_summary (mission_id, start_time, status) VALUES (?, ?, ?)',
                (self.mission_id, self.start_time, "IN_PROGRESS")
            )
            self.conn.commit()

    def _create_tables(self):
        with self._lock:
            self.cursor.executescript('''
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mission_id TEXT,
                    timestamp TEXT,
                    latitude REAL,
                    longitude REAL,
                    altitude REAL,
                    battery_v REAL,
                    message TEXT
                );
                CREATE TABLE IF NOT EXISTS hardware_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mission_id TEXT,
                    timestamp TEXT,
                    pitot_airspeed REAL,
                    lidar_distance REAL,
                    vision_targets_count INTEGER
                );
                CREATE TABLE IF NOT EXISTS flight_summary (
                    mission_id TEXT PRIMARY KEY,
                    start_time TEXT,
                    end_time TEXT,
                    status TEXT,
                    total_distance_m REAL,
                    obstacles_encountered INTEGER,
                    evasions_triggered INTEGER,
                    errors_warnings INTEGER
                );
            ''')
            self.conn.commit()

    def log_telemetry(self, lat: float, lon: float, alt: float, battery_v: float, message: str = "", level: str = "info"):
        """Fix #7: Queues telemetry data. Commits to disk in batches of BATCH_SIZE."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        self._telemetry_queue.append((self.mission_id, timestamp, lat, lon, alt, battery_v, message))
        
        if level in ["warning", "error", "critical"]:
            self.errors_warnings += 1
            # Critical events always flush to SQLite immediately
            self._flush_telemetry_queue()
            # B5 FIX: Also write to text log as fallback in case SQLite is corrupted
            log_line = f"[{level.upper()}] ({lat:.6f},{lon:.6f}) alt={alt:.1f}m | {message}"
            if level == "critical":
                self._text_logger.critical(log_line)
            elif level == "error":
                self._text_logger.error(log_line)
            else:
                self._text_logger.warning(log_line)
        
        # Track distance
        if self.last_lat is not None:
            R = 6371000
            phi1, phi2 = math.radians(self.last_lat), math.radians(lat)
            dphi = math.radians(lat - self.last_lat)
            dlambda = math.radians(lon - self.last_lon)
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            self.total_distance += R * c
        self.last_lat = lat
        self.last_lon = lon
        
        # Batch commit (Fix #7)
        if len(self._telemetry_queue) >= self.BATCH_SIZE:
            self._flush_telemetry_queue()

    def _flush_telemetry_queue(self):
        """Writes the buffered telemetry rows to disk in a single transaction."""
        if not self._telemetry_queue:
            return
        with self._lock:
            self.cursor.executemany(
                'INSERT INTO telemetry (mission_id, timestamp, latitude, longitude, altitude, battery_v, message) VALUES (?,?,?,?,?,?,?)',
                self._telemetry_queue
            )
            self.conn.commit()
        self._telemetry_queue.clear()

    def log_hardware_telemetry(self, pitot_airspeed: float, lidar_distance: float, vision_targets_count: int):
        """Fix #7: Queues hardware sensor data. Commits to disk in batches."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self._hw_queue.append((self.mission_id, timestamp, pitot_airspeed, lidar_distance, vision_targets_count))
        
        if len(self._hw_queue) >= self.BATCH_SIZE:
            self._flush_hw_queue()

    def _flush_hw_queue(self):
        if not self._hw_queue:
            return
        with self._lock:
            self.cursor.executemany(
                'INSERT INTO hardware_telemetry (mission_id, timestamp, pitot_airspeed, lidar_distance, vision_targets_count) VALUES (?,?,?,?,?)',
                self._hw_queue
            )
            self.conn.commit()
        self._hw_queue.clear()

    def record_obstacle(self):
        self.obstacles_encountered += 1
        
    def record_evasion(self):
        self.evasions_triggered += 1

    def finalize_flight(self, status="SUCCESS"):
        """Flushes all queued data and writes final summary to DB."""
        # Flush any remaining buffered rows first
        self._flush_telemetry_queue()
        self._flush_hw_queue()
        
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            self.cursor.execute(
                '''UPDATE flight_summary SET end_time=?, status=?, total_distance_m=?,
                   obstacles_encountered=?, evasions_triggered=?, errors_warnings=? WHERE mission_id=?''',
                (end_time, status, self.total_distance, self.obstacles_encountered,
                 self.evasions_triggered, self.errors_warnings, self.mission_id)
            )
            self.conn.commit()
        print(f"\n[Logger] Flight {self.mission_id} finalized: {status}")
        print(f"[Logger] Stats: {self.total_distance:.1f}m traveled, {self.obstacles_encountered} obstacles, {self.errors_warnings} errors.")

    def close(self):
        """Ensure all remaining data is flushed before closing the DB connection."""
        self._flush_telemetry_queue()
        self._flush_hw_queue()
        self.conn.close()
