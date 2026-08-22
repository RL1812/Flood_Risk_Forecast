import json
import os
from flask import Flask, render_template, jsonify

app = Flask(__name__)

app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/geojson')
def get_geojson():
    geojson_path = os.path.join(os.path.dirname(__file__), 'enriched_planning_areas.geojson')
    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify(data)

