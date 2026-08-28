/**
 * Singapore Flood Risk Intelligence & Real-Time Forecasting System
 * Leaflet Map Integration & Dual-Mode Controller:
 * 1. Flood Vulnerability Map in Singapore (COP30 DEM Topographic Sensitivity)
 * 2. Flood Risk Forecast in 15 Mins (Live 1h45m Data.gov.sg Ingestion + XGBoost ML Inference)
 */

document.addEventListener('DOMContentLoaded', function () {
  // Current active mode: 'vulnerability' | 'forecast'
  let currentMode = 'vulnerability';

  // 1. Initialize Leaflet Map centered on Singapore
  const initialCenter = [1.3521, 103.8198];
  const initialZoom = 11;
  const map = L.map('map').setView(initialCenter, initialZoom);

  // 2. Base Tile Layer (CartoDB Positron)
  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    maxZoom: 19,
  }).addTo(map);

  let geojsonLayer = null;
  let selectedLayer = null;
  let selectedAreaName = null;
  const layersByName = {};
  let currentElevationLegend = null;

  // 3. Color Scales
  function getElevationColor(elevation) {
    return elevation > 30
      ? '#005824'
      : elevation > 20
      ? '#238b45'
      : elevation > 15
      ? '#41ae76'
      : elevation > 10
      ? '#66c2a4'
      : elevation > 5
      ? '#99d8c9'
      : elevation > 2
      ? '#ccece6'
      : '#edf8fb';
  }

  function getFeatureStyle(feature) {
    if (currentMode === 'vulnerability') {
      const meanElev = feature.properties.elev_mean || 0;
      return {
        fillColor: getElevationColor(meanElev),
        weight: 1.5,
        opacity: 1,
        color: '#495057',
        fillOpacity: 0.7,
      };
    } else {
      // Forecast Mode: Clean oceanic slate styling ready for alert highlights
      return {
        fillColor: '#6c757d',
        weight: 1.5,
        opacity: 1,
        color: '#343a40',
        fillOpacity: 0.35,
      };
    }
  }

  // 4. Update Vulnerability Details Panel
  function updateVulnerabilityPanel(props) {
    document.getElementById('noSelectionPrompt').style.display = 'none';
    document.getElementById('selectionContent').style.display = 'block';

    const name = props.PLN_AREA_N || 'Unknown Area';
    const region = props.REGION_N || 'Singapore';
    const mean = props.elev_mean !== undefined ? props.elev_mean : 0;
    const min = props.elev_min !== undefined ? props.elev_min : 0;
    const std = props.elev_std !== undefined ? props.elev_std : 0;

    document.getElementById('panelAreaName').innerText = name;
    document.getElementById('panelRegionName').innerText = region;
    document.getElementById('selectedRegionBadge').innerText = name;
    document.getElementById('panelMeanElev').innerText = `${mean.toFixed(2)} m`;
    document.getElementById('panelMinElev').innerText = `${min.toFixed(2)} m`;
    document.getElementById('panelStdElev').innerText = `±${std.toFixed(2)} m`;
    document.getElementById('panelRange').innerText = `${min.toFixed(1)}m to ${(mean + std * 2).toFixed(1)}m`;

    const riskBadge = document.getElementById('panelRiskBadge');
    const progressBar = document.getElementById('panelProgressBar');
    const pct = Math.min(100, Math.max(5, (mean / 35) * 100));
    progressBar.style.width = `${pct}%`;

    if (mean < 5 || min < 0) {
      riskBadge.className = 'risk-badge risk-high';
      riskBadge.innerText = '⚠️ Low-Lying / Coastal Flood Sensitivity';
      progressBar.className = 'progress-bar bg-danger';
    } else if (mean < 15) {
      riskBadge.className = 'risk-badge risk-med';
      riskBadge.innerText = '⚡ Moderate Elevation Buffer';
      progressBar.className = 'progress-bar bg-warning';
    } else {
      riskBadge.className = 'risk-badge risk-low';
      riskBadge.innerText = '🛡️ High Elevation / Low Inundation Risk';
      progressBar.className = 'progress-bar bg-success';
    }
  }

  // 5. Update Forecast Panel by requesting live telemetry & running XGBoost ML
  function updateForecastPanel(areaName) {
    if (!areaName) return;

    document.getElementById('forecastEmptyPrompt').style.display = 'none';
    document.getElementById('forecastResultsContent').style.display = 'none';
    document.getElementById('forecastLoadingSpinner').style.display = 'block';
    document.getElementById('forecastRegionBadge').innerText = areaName;

    fetch(`/api/forecast?pln_area=${encodeURIComponent(areaName)}`)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Failed to fetch forecast (${res.status})`);
        }
        return res.json();
      })
      .then((data) => {
        document.getElementById('forecastLoadingSpinner').style.display = 'none';
        document.getElementById('forecastResultsContent').style.display = 'block';

        renderForecastData(data);
      })
      .catch((err) => {
        document.getElementById('forecastLoadingSpinner').style.display = 'none';
        document.getElementById('forecastEmptyPrompt').style.display = 'block';
        alert(`Error fetching real-time forecast for ${areaName}: ${err.message}`);
      });
  }

  // 6. Render Forecast Response into UI
  function renderForecastData(data) {
    const plnArea = data.pln_area;
    const pred = data.prediction || {};
    const metrics = data.rainfall_metrics || {};
    const sensor = data.sensor || {};
    const elevation = data.elevation || {};
    const series = data.readings_series || [];

    // Header & Badges
    document.getElementById('fcAreaName').innerText = plnArea;
    document.getElementById('forecastRegionBadge').innerText = plnArea;
    const nowTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    document.getElementById('fcTimestamp').innerText = `Live Evaluated: ${nowTime} SGT`;

    const tier = pred.alert_tier || 'NORMAL';
    const tierBadge = document.getElementById('fcTierBadge');
    const actionBox = document.getElementById('fcActionBox');
    const actionIcon = document.getElementById('fcActionIcon');
    const actionTitle = document.getElementById('fcActionTitle');
    const actionText = document.getElementById('fcActionText');
    const probBar = document.getElementById('fcProbBar');
    const probText = document.getElementById('fcProbText');

    const probPct = pred.prob_percentage || 0;
    probText.innerText = `${probPct.toFixed(1)}%`;
    probBar.style.width = `${Math.min(100, Math.max(5, probPct * 4))}%`;

    // Tier styling
    if (tier === 'WARNING') {
      tierBadge.className = 'badge bg-danger fs-6 px-3 py-2 animate-pulse';
      tierBadge.innerHTML = '🚨 FLOOD WARNING (TIER 2)';
      actionBox.className = 'alert alert-danger d-flex align-items-start mb-3 py-2 px-3 shadow-sm border-danger';
      actionIcon.className = 'bi bi-exclamation-triangle-fill fs-4 text-danger me-2';
      actionTitle.innerText = 'Immediate Tactical Response Required:';
      probBar.className = 'progress-bar bg-danger progress-bar-striped progress-bar-animated';
    } else if (tier === 'WATCH') {
      tierBadge.className = 'badge bg-warning text-dark fs-6 px-3 py-2';
      tierBadge.innerHTML = '⚡ FLOOD WATCH (TIER 1)';
      actionBox.className = 'alert alert-warning d-flex align-items-start mb-3 py-2 px-3 shadow-sm border-warning';
      actionIcon.className = 'bi bi-bell-fill fs-4 text-warning me-2';
      actionTitle.innerText = 'Heightened Situational Awareness (High Recall):';
      probBar.className = 'progress-bar bg-warning progress-bar-striped';
    } else {
      tierBadge.className = 'badge bg-success fs-6 px-3 py-2';
      tierBadge.innerHTML = '🛡️ NORMAL (NO ALERT)';
      actionBox.className = 'alert alert-success d-flex align-items-start mb-3 py-2 px-3 shadow-sm border-success';
      actionIcon.className = 'bi bi-shield-check fs-4 text-success me-2';
      actionTitle.innerText = 'Routine Operations:';
      probBar.className = 'progress-bar bg-success';
    }

    actionText.innerText = pred.action_recommendation || 'Continue routine sensor telemetry.';

    // Sensor Information
    const sensorBadge = document.getElementById('fcSensorIdBadge');
    sensorBadge.innerText = `Station: ${sensor.id || 'N/A'}`;
    const directNote = sensor.direct_match ? 'Direct Station' : (sensor.nearest_note || 'Adjacent Station');
    document.getElementById('fcSensorDetails').innerText = 
      `${sensor.name || sensor.id} (${directNote}) | Lat: ${sensor.latitude}, Lon: ${sensor.longitude}`;

    // Rainfall feature metrics
    document.getElementById('fcRain15m').innerText = `${metrics.rain_sum_15m || 0.0} mm`;
    document.getElementById('fcRain30m').innerText = `${metrics.rain_sum_30m || 0.0} mm`;
    document.getElementById('fcRain90m').innerText = `${metrics.rain_sum_90m || 0.0} mm`;
    document.getElementById('fcRainMax5m').innerText = `${metrics.rain_max_5m || 0.0} mm`;

    // Render Timeline Bar Chart
    const timelineContainer = document.getElementById('fcTimelineContainer');
    timelineContainer.innerHTML = '';

    const maxValInSeries = Math.max(1.0, ...series.map((s) => s.value));

    series.forEach((pt, idx) => {
      const bar = document.createElement('div');
      const val = pt.value || 0;
      const heightPct = Math.max(8, (val / maxValInSeries) * 100);
      bar.className = 'timeline-bar';
      bar.style.height = `${heightPct}%`;
      bar.style.flex = '1';
      bar.style.backgroundColor = val > 5 ? '#dc3545' : val > 0.5 ? '#0d6efd' : val > 0 ? '#6ea8fe' : '#dee2e6';
      bar.title = `${pt.timestamp || `Step ${idx+1}`}: ${val} mm`;
      timelineContainer.appendChild(bar);
    });

    // Elevation summary
    document.getElementById('fcMeanElev').innerText = `${(elevation.elev_mean || 0).toFixed(1)} m`;
    document.getElementById('fcMinElev').innerText = `${(elevation.elev_min || 0).toFixed(1)} m`;
    document.getElementById('fcStdElev').innerText = `±${(elevation.elev_std || 0).toFixed(1)} m`;

    // Technical breakdown
    document.getElementById('fcRawProb').innerText = `${((pred.raw_prob_flood || 0) * 100).toFixed(2)}%`;
    document.getElementById('fcCalProb').innerText = `${((pred.calibrated_prob_flood || 0) * 100).toFixed(2)}%`;
    document.getElementById('fcTempResistance').innerHTML = pred.temporal_resistance_triggered
      ? '<span class="badge bg-info-subtle text-dark">Active (&lt;0.2mm override)</span>'
      : '<span class="badge bg-light text-muted">Inactive (Rainfall &ge; 0.2mm)</span>';
  }

  // 7. Layer Selection Handler
  function highlightFeature(e) {
    const layer = e.target;
    if (layer !== selectedLayer) {
      layer.setStyle({
        weight: 3,
        color: '#0d6efd',
        fillOpacity: currentMode === 'vulnerability' ? 0.85 : 0.6,
      });
    }
  }

  function resetHighlight(e) {
    const layer = e.target;
    if (layer !== selectedLayer) {
      geojsonLayer.resetStyle(layer);
    }
  }

  function selectFeature(layer) {
    if (selectedLayer && selectedLayer !== layer) {
      geojsonLayer.resetStyle(selectedLayer);
    }

    selectedLayer = layer;
    selectedAreaName = selectedLayer.feature.properties.PLN_AREA_N;

    selectedLayer.setStyle({
      weight: 3.5,
      color: '#dc3545',
      fillOpacity: 0.9,
    });

    if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {
      selectedLayer.bringToFront();
    }

    map.fitBounds(selectedLayer.getBounds(), {
      padding: [40, 40],
      maxZoom: 14,
    });

    // Update Dropdown
    const select = document.getElementById('regionSelect');
    if (select) {
      select.value = selectedAreaName;
    }

    // Trigger active mode panel update
    if (currentMode === 'vulnerability') {
      updateVulnerabilityPanel(selectedLayer.feature.properties);
    } else {
      updateForecastPanel(selectedAreaName);
    }
  }

  // 8. Bind interactions per region feature
  function onEachFeature(feature, layer) {
    const props = feature.properties;
    const areaName = props.PLN_AREA_N || 'Unknown';
    layersByName[areaName] = layer;

    // Hover tooltip
    layer.bindTooltip(
      `
      <div class="fw-bold">${areaName}</div>
      <div class="small text-muted">${props.REGION_N || ''}</div>
      <div class="small">Mean Elevation: <strong>${(props.elev_mean || 0).toFixed(1)} m</strong></div>
    `,
      { sticky: true }
    );

    // Event listeners
    layer.on({
      mouseover: highlightFeature,
      mouseout: resetHighlight,
      click: function () {
        selectFeature(layer);
      },
    });
  }

  // 9. Fetch and Render GeoJSON
  fetch('/api/geojson')
    .then((res) => {
      if (!res.ok) throw new Error('Failed to load GeoJSON');
      return res.json();
    })
    .then((data) => {
      geojsonLayer = L.geoJson(data, {
        style: getFeatureStyle,
        onEachFeature: onEachFeature,
      }).addTo(map);

      // Populate Dropdown
      const select = document.getElementById('regionSelect');
      const names = Object.keys(layersByName).sort();
      names.forEach((name) => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        select.appendChild(option);
      });

      select.addEventListener('change', function () {
        const name = this.value;
        if (name && layersByName[name]) {
          selectFeature(layersByName[name]);
        }
      });
    })
    .catch((err) => {
      console.error(err);
      alert('Could not load enriched GeoJSON: ' + err.message);
    });

  // 10. Elevation Legend Control
  function createElevationLegend() {
    if (currentElevationLegend) {
      map.removeControl(currentElevationLegend);
    }
    const legend = L.control({ position: 'bottomright' });
    legend.onAdd = function () {
      const div = L.DomUtil.create('div', 'legend');
      const grades = [0, 2, 5, 10, 15, 20, 30];
      div.innerHTML = '<strong>Mean Elevation (m)</strong><br>';
      for (let i = 0; i < grades.length; i++) {
        div.innerHTML +=
          '<i style="background:' +
          getElevationColor(grades[i] + 1) +
          '"></i> ' +
          grades[i] +
          (grades[i + 1] ? '&ndash;' + grades[i + 1] + '<br>' : '+');
      }
      return div;
    };
    legend.addTo(map);
    currentElevationLegend = legend;
  }

  createElevationLegend();

  // 11. Mode Switching Logic (Nav Bar Tabs)
  const navVulnerability = document.getElementById('navVulnerability');
  const navForecast = document.getElementById('navForecast');
  const vulnerabilityPanel = document.getElementById('vulnerabilityPanel');
  const forecastPanel = document.getElementById('forecastPanel');
  const viewHeaderTitle = document.getElementById('viewHeaderTitle');
  const viewHeaderSubtitle = document.getElementById('viewHeaderSubtitle');
  const modeBadge = document.getElementById('modeBadge');
  const dataGovBadge = document.getElementById('dataGovBadge');
  const mapInstruction = document.getElementById('mapInstruction');

  function setMode(mode) {
    currentMode = mode;

    if (mode === 'vulnerability') {
      navVulnerability.classList.add('active');
      navForecast.classList.remove('active');
      vulnerabilityPanel.style.display = 'block';
      forecastPanel.style.display = 'none';

      viewHeaderTitle.innerText = 'Singapore Flood Vulnerability Map';
      viewHeaderSubtitle.innerText = 'Interactive Digital Elevation Model (COP30 DEM) & Topographic Inundation Analysis';
      modeBadge.className = 'badge bg-primary-subtle text-primary border border-primary-subtle px-2 py-1 small';
      modeBadge.innerHTML = '<i class="bi bi-layers-fill"></i> Vulnerability Mode';
      dataGovBadge.style.display = 'none';
      mapInstruction.innerText = '💡 Click any planning area polygon on the map to inspect elevation and topographic risk.';

      createElevationLegend();

      if (geojsonLayer) {
        geojsonLayer.eachLayer(function (layer) {
          if (layer !== selectedLayer) {
            geojsonLayer.resetStyle(layer);
          }
        });
      }

      if (selectedLayer) {
        updateVulnerabilityPanel(selectedLayer.feature.properties);
      }
    } else {
      navForecast.classList.add('active');
      navVulnerability.classList.remove('active');
      forecastPanel.style.display = 'block';
      vulnerabilityPanel.style.display = 'none';

      viewHeaderTitle.innerText = 'Real-Time 15-Minute Flood Risk Forecast';
      viewHeaderSubtitle.innerText = 'Live 1h 45m Data.gov.sg rainfall ingestion & XGBoost Machine Learning inference by planning area';
      modeBadge.className = 'badge bg-info-subtle text-info-emphasis border border-info-subtle px-2 py-1 small';
      modeBadge.innerHTML = '<i class="bi bi-stopwatch-fill"></i> 15-Min Forecast Mode';
      dataGovBadge.style.display = 'inline-block';
      mapInstruction.innerText = '⏱️ Click any planning area to ingest live rainfall from Data.gov.sg and generate 15-min flood prediction.';

      if (currentElevationLegend) {
        map.removeControl(currentElevationLegend);
        currentElevationLegend = null;
      }

      if (geojsonLayer) {
        geojsonLayer.eachLayer(function (layer) {
          if (layer !== selectedLayer) {
            geojsonLayer.resetStyle(layer);
          }
        });
      }

      if (selectedLayer) {
        updateForecastPanel(selectedAreaName);
      }
    }
  }

  navVulnerability.addEventListener('click', function (e) {
    e.preventDefault();
    setMode('vulnerability');
  });

  navForecast.addEventListener('click', function (e) {
    e.preventDefault();
    setMode('forecast');
  });

  // 12. Reset Map View Handler
  document.getElementById('btnResetView').addEventListener('click', function () {
    if (selectedLayer && geojsonLayer) {
      geojsonLayer.resetStyle(selectedLayer);
      selectedLayer = null;
      selectedAreaName = null;
    }
    map.setView(initialCenter, initialZoom);
    document.getElementById('regionSelect').value = '';

    // Reset Vulnerability panel
    document.getElementById('noSelectionPrompt').style.display = 'block';
    document.getElementById('selectionContent').style.display = 'none';
    document.getElementById('selectedRegionBadge').innerText = 'No Region Selected';

    // Reset Forecast panel
    document.getElementById('forecastEmptyPrompt').style.display = 'block';
    document.getElementById('forecastResultsContent').style.display = 'none';
    document.getElementById('forecastLoadingSpinner').style.display = 'none';
    document.getElementById('forecastRegionBadge').innerText = 'No Region Selected';
  });
});
