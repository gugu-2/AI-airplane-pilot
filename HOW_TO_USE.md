# 🚀 How to Use Aegis Autonomy

## 1. Launch the Ground Control Station (GCS)
The GCS is a Palantir-style web dashboard that tracks telemetry, live map data, and system health.
```bash
python src/dashboard/app.py
```
*Open a web browser and navigate to `http://127.0.0.1:5000`*

## 2. Run the Full Simulated Mission
This script compresses the core research milestones into a single execution. It launches a drone, navigates waypoints, detects obstacles with YOLOv8, and autonomously returns to land.
```bash
python scripts/ultimate_month3_mission.py
```

## 3. Test Individual Safety Modules
You can test the specialized AI and safety firewalls by running their individual nodes:
```bash
# Test the Airspace Geofence Engine
python src/control/geofence_engine.py

# Test the Black Box Flight Recorder
python src/control/blackbox_node.py

# Test the Weather API pre-flight checklist
python src/fusion/weather_api_node.py

# Test ADS-B Traffic TCAS evasion
python src/fusion/adsb_traffic_node.py
```

## 4. Deploy to Physical Hardware
If you are deploying to an NVIDIA Jetson Orin Nano companion computer, run the hardware mapping shell script to install CUDA, TensorRT, and setup the UDEV rules.
```bash
bash scripts/deploy_to_jetson.sh
```
