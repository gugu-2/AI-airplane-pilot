from flask import Flask, render_template
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = 'aegis-secret!'

# Suppress standard Flask logging to keep console clean
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print(">>> STARTING AEGIS GROUND CONTROL STATION (GCS) ON PORT 5000")
    print(">>> Frontend will connect to ws://127.0.0.1:8765 for live telemetry.")
    app.run(host='0.0.0.0', port=5000, debug=False)
