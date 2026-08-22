/**
 * Singapore Flood Risk & Elevation Explorer - Leaflet Map Integration
 */

document.addEventListener('DOMContentLoaded', function () {
  // 1. Initialize Leaflet Map centered on Singapore
  const initialCenter = [1.3521, 103.8198];
  const initialZoom = 11;
  const map = L.map('map').setView(initialCenter, initialZoom);

  // 2. Add Base Tile Layer (CartoDB Positron for high contrast)
  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    maxZoom: 19,
  }).addTo(map);

  let geojsonLayer = null;
  let selectedLayer = null;
  const layersByName = {};

  // 3. Color scale based on mean elevation (meters)
  function getColor(elevation) {
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

  // 4. Default style function for polygons
  function style(feature) {
    const meanElev = feature.properties.elev_mean || 0;
    return {
      fillColor: getColor(meanElev),
      weight: 1.5,
      opacity: 1,
      color: '#495057',
      fillOpacity: 0.7,
    };
  }

  // 5. Update the sidebar panel with region statistics
  function updateDetailsPanel(props) {
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
    document.getElementById('panelRange').innerText = `${min.toFixed(1)}m to ${(
      mean +
      std * 2
    ).toFixed(1)}m`;

    // Vulnerability classification
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

  // 6. Highlight and selection handlers
  function highlightFeature(e) {
    const layer = e.target;
    if (layer !== selectedLayer) {
      layer.setStyle({
        weight: 3,
        color: '#0d6efd',
        fillOpacity: 0.85,
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
    updateDetailsPanel(selectedLayer.feature.properties);

    // Sync dropdown
    const select = document.getElementById('regionSelect');
    if (select) {
      select.value = selectedLayer.feature.properties.PLN_AREA_N;
    }
  }

  // 7. Bind interactions to each region feature
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

    // Click popup
    const popupContent = `
      <div style="min-width: 180px;">
        <h6 class="mb-1 fw-bold text-primary">${areaName}</h6>
        <div class="small text-muted mb-2">${props.REGION_N || ''}</div>
        <table class="table table-sm table-borderless mb-0 small">
          <tr><td><strong>Mean Elev:</strong></td><td class="text-end">${(props.elev_mean || 0).toFixed(2)} m</td></tr>
          <tr><td><strong>Min Elev:</strong></td><td class="text-end">${(props.elev_min || 0).toFixed(2)} m</td></tr>
          <tr><td><strong>Std Dev:</strong></td><td class="text-end">±${(props.elev_std || 0).toFixed(2)} m</td></tr>
        </table>
      </div>
    `;
    layer.bindPopup(popupContent);

    // Event listeners
    layer.on({
      mouseover: highlightFeature,
      mouseout: resetHighlight,
      click: function () {
        selectFeature(layer);
      },
    });
  }

  // 8. Fetch GeoJSON from backend endpoint
  fetch('/api/geojson')
    .then((response) => {
      if (!response.ok) {
        throw new Error('Failed to load GeoJSON data');
      }
      return response.json();
    })
    .then((data) => {
      geojsonLayer = L.geoJson(data, {
        style: style,
        onEachFeature: onEachFeature,
      }).addTo(map);

      // Populate dropdown selector
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
          layersByName[name].openPopup();
        }
      });
    })
    .catch((err) => {
      console.error(err);
      alert('Could not load enriched GeoJSON: ' + err.message);
    });

  // 9. Add Elevation Legend
  const legend = L.control({ position: 'bottomright' });
  legend.onAdd = function () {
    const div = L.DomUtil.create('div', 'legend');
    const grades = [0, 2, 5, 10, 15, 20, 30];
    div.innerHTML = '<strong>Mean Elevation (m)</strong><br>';

    for (let i = 0; i < grades.length; i++) {
      div.innerHTML +=
        '<i style="background:' +
        getColor(grades[i] + 1) +
        '"></i> ' +
        grades[i] +
        (grades[i + 1] ? '&ndash;' + grades[i + 1] + '<br>' : '+');
    }
    return div;
  };
  legend.addTo(map);

  // 10. Reset View button handler
  document.getElementById('btnResetView').addEventListener('click', function () {
    if (selectedLayer && geojsonLayer) {
      geojsonLayer.resetStyle(selectedLayer);
      selectedLayer = null;
    }
    map.setView(initialCenter, initialZoom);
    document.getElementById('regionSelect').value = '';
    document.getElementById('noSelectionPrompt').style.display = 'block';
    document.getElementById('selectionContent').style.display = 'none';
    document.getElementById('selectedRegionBadge').innerText = 'No Region Selected';
  });
});
