(function () {
  "use strict";

  const TRACY_DEPOT = { lat: 37.7397, lng: -121.4252 };

  const PRESETS = {
    clear: {
      temp_min_mean: 45,
      temp_max_mean: 72,
      snow_depth_mean: 0,
      prcp_total_mean: 0,
      visibility_mean: 10,
      wind_speed_mean: 5,
      wind_gust_max_mean: 12,
    },
    moderate: {
      temp_min_mean: 38,
      temp_max_mean: 60,
      snow_depth_mean: 0,
      prcp_total_mean: 0.1,
      visibility_mean: 2,
      wind_speed_mean: 15,
      wind_gust_max_mean: 29,
    },
    snow: {
      temp_min_mean: 28,
      temp_max_mean: 35,
      snow_depth_mean: 5,
      prcp_total_mean: 2,
      visibility_mean: 3,
      wind_speed_mean: 12,
      wind_gust_max_mean: 35,
    },
  };

  const sourceDisplay = document.getElementById("source-display");
  const sourceModeSelect = document.getElementById("source-mode");
  const sourceCityWrap = document.getElementById("source-city-wrap");
  const sourceGpsWrap = document.getElementById("source-gps-wrap");
  const sourceCityInput = document.getElementById("source-city");
  const sourceGpsBtn = document.getElementById("source-gps-btn");
  const destinationSelect = document.getElementById("destination");
  const routesBtn = document.getElementById("routes-btn");
  const predictDelayBtn = document.getElementById("predict-delay-btn");
  const mapSection = document.getElementById("map-section");
  const mapDiv = document.getElementById("map");
  const routesResults = document.getElementById("routes-results");
  const routeCards = document.getElementById("route-cards");
  const modelSelect = document.getElementById("model-select");

  let lastRoutes = null;
  let lastOrigin = null;
  let lastDestination = null;
  let selectedOrigin = { lat: TRACY_DEPOT.lat, lng: TRACY_DEPOT.lng, name: "Tracy Depot, CA" };
  let routesAbortController = null;
  const form = document.getElementById("weather-form");
  const predictBtn = document.getElementById("predict-btn");
  const resultsSection = document.getElementById("results-section");
  const resultsCards = document.getElementById("results-cards");
  const chartCanvas = document.getElementById("chart");
  const errorToast = document.getElementById("error-toast");
  const loaderOverlay = document.getElementById("loader-overlay");
  const loaderMessage = document.getElementById("loader-message");
  const loaderSteps = document.getElementById("loader-steps");

  // Step-wise messages for Predict delay (route-delays) loader
  const PREDICT_DELAY_STEPS = [
    "Breaking the route into 20 mile pitstops",
    "Finding nearest weather stations for each pitstop",
    "Calculating delay % from the model for each segment",
    "Combining results",
  ];

  const FEATURE_IDS = [
    "temp_min_mean", "temp_max_mean", "snow_depth_mean", "prcp_total_mean",
    "visibility_mean", "wind_speed_mean", "wind_gust_max_mean",
  ];

  let chartInstance = null;
  let mapInstance = null;
  let mapPolylines = [];
  let mapMarkers = [];
  let mapsApiKey = null;

  function clearMapOverlays() {
    mapPolylines.forEach(function (p) { if (p && p.setMap) p.setMap(null); });
    mapMarkers.forEach(function (m) { if (m && m.setMap) m.setMap(null); });
    mapPolylines = [];
    mapMarkers = [];
  }

  function showError(msg) {
    if (errorToast) {
      errorToast.textContent = msg;
      errorToast.setAttribute("aria-hidden", "false");
      setTimeout(function () {
        errorToast.setAttribute("aria-hidden", "true");
      }, 6000);
    }
  }

  function showLoader(messageOrSteps) {
    if (!loaderOverlay) return;
    const isSteps = Array.isArray(messageOrSteps);
    if (loaderMessage) {
      loaderMessage.textContent = isSteps ? "" : (messageOrSteps || "Loading…");
      loaderMessage.setAttribute("aria-hidden", isSteps ? "true" : "false");
    }
    if (loaderSteps) {
      if (isSteps && messageOrSteps.length > 0) {
        loaderSteps.innerHTML = messageOrSteps
          .map(function (text) { return "<li class=\"loader-step\">" + text + "</li>"; })
          .join("");
        loaderSteps.setAttribute("aria-hidden", "false");
      } else {
        loaderSteps.innerHTML = "";
        loaderSteps.setAttribute("aria-hidden", "true");
      }
    }
    loaderOverlay.setAttribute("aria-hidden", "false");
  }

  function hideLoader() {
    if (loaderOverlay) loaderOverlay.setAttribute("aria-hidden", "true");
  }

  // --- Source & warehouses ---
  function setSelectedOrigin(origin) {
    selectedOrigin = {
      lat: Number(origin.lat),
      lng: Number(origin.lng),
      name: origin.name || "Custom source",
    };
    if (sourceDisplay) {
      sourceDisplay.textContent =
        selectedOrigin.name + " (" + selectedOrigin.lat.toFixed(4) + ", " + selectedOrigin.lng.toFixed(4) + ")";
    }
  }

  function refreshSourceModeUI() {
    if (!sourceModeSelect) return;
    const mode = sourceModeSelect.value || "tracy";
    if (sourceCityWrap) sourceCityWrap.style.display = mode === "city" ? "block" : "none";
    if (sourceGpsWrap) sourceGpsWrap.style.display = mode === "gps" ? "block" : "none";
    if (mode === "tracy") {
      setSelectedOrigin({ lat: TRACY_DEPOT.lat, lng: TRACY_DEPOT.lng, name: "Tracy Depot, CA" });
    }
  }

  function geocodeSourceCity(cityText) {
    return fetch("/api/geocode-source", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ city: cityText }),
    }).then(function (res) {
      return res.json().then(function (body) {
        if (!res.ok) throw new Error(body.error || "Failed to geocode city");
        return body;
      });
    });
  }

  function useGpsSource() {
    if (!navigator.geolocation) {
      showError("Geolocation is not supported in this browser");
      return;
    }
    if (sourceGpsBtn) {
      sourceGpsBtn.disabled = true;
      sourceGpsBtn.textContent = "Detecting…";
    }
    navigator.geolocation.getCurrentPosition(
      function (pos) {
        const lat = pos.coords.latitude;
        const lng = pos.coords.longitude;
        setSelectedOrigin({ lat: lat, lng: lng, name: "My current location" });
        if (sourceGpsBtn) {
          sourceGpsBtn.disabled = false;
          sourceGpsBtn.textContent = "Use my current location";
        }
      },
      function (err) {
        showError("Could not get GPS location: " + (err && err.message ? err.message : "permission denied"));
        if (sourceGpsBtn) {
          sourceGpsBtn.disabled = false;
          sourceGpsBtn.textContent = "Use my current location";
        }
      },
      { enableHighAccuracy: true, timeout: 12000 }
    );
  }

  function loadSource() {
    fetch("/api/source")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        setSelectedOrigin({ lat: data.lat, lng: data.lng, name: data.name || "Tracy Depot, CA" });
      })
      .catch(function () {
        setSelectedOrigin({ lat: TRACY_DEPOT.lat, lng: TRACY_DEPOT.lng, name: "Tracy Depot, CA" });
      });
  }

  function loadWarehouses() {
    fetch("/api/warehouses")
      .then(function (r) { return r.json(); })
      .then(function (list) {
        destinationSelect.innerHTML = '<option value="">— Select destination —</option>';
        if (Array.isArray(list)) {
          list.forEach(function (w) {
            const opt = document.createElement("option");
            const lat = w.lat != null ? w.lat : w.latitude;
            const lng = w.lng != null ? w.lng : w.longitude;
            if (lat == null || lng == null) return;
            opt.value = Number(lat) + "," + Number(lng);
            opt.textContent = (w.name || "Warehouse") + " (ID " + (w.id || "") + ")";
            opt.dataset.name = w.name || "";
            destinationSelect.appendChild(opt);
          });
        }
      })
      .catch(function (err) {
        destinationSelect.innerHTML = '<option value="">— Failed to load —</option>';
        showError("Could not load destinations");
      });
  }

  // --- Routes & delay ---
  function getSelectedDestination() {
    const v = destinationSelect.value;
    if (!v || v.indexOf(",") === -1) return null;
    const parts = v.split(",");
    const lat = parseFloat(parts[0]);
    const lng = parseFloat(parts[1]);
    if (Number.isNaN(lat) || Number.isNaN(lng)) return null;
    return { lat: lat, lng: lng };
  }

  function formatDurationFromSeconds(totalSeconds) {
    if (totalSeconds == null || !Number.isFinite(totalSeconds) || totalSeconds < 0) return "—";
    var sec = Math.round(totalSeconds);
    var h = Math.floor(sec / 3600);
    var m = Math.floor((sec % 3600) / 60);
    if (h > 0 && m > 0) return h + " h " + m + " min";
    if (h > 0) return h + " h";
    if (m > 0) return m + " min";
    return sec + " s";
  }

  function renderRouteCards(routes) {
    if (!routeCards) return;
    if (!routes || routes.length === 0) {
      routeCards.innerHTML = "<p>No routes.</p>";
      return;
    }
    const modelName = modelSelect ? modelSelect.value : "Ridge";
    // Best route = shortest total time after delay % is applied (adjusted duration)
    var bestRouteIndex = null;
    var bestAdjustedSeconds = Infinity;
    routes.forEach(function (r) {
      var baseS = r.duration_s != null ? Number(r.duration_s) : null;
      var delayVal = (r.delays || {})[modelName];
      if (baseS != null && Number.isFinite(baseS) && delayVal != null && typeof delayVal === "number") {
        var adjustedS = baseS * (1 + delayVal / 100);
        if (adjustedS < bestAdjustedSeconds) {
          bestAdjustedSeconds = adjustedS;
          bestRouteIndex = r.route_index;
        }
      }
    });
    routeCards.innerHTML = routes
      .map(function (r) {
        const delays = r.delays || {};
        const d = delays;
        const pitstopNote = r.pitstop_count != null ? " <small>(from " + r.pitstop_count + " pitstops)</small>" : "";
        const delayVal = d[modelName];
        const hasDelay = delayVal != null;
        const delayHtml = hasDelay
          ? '<span class="route-delay route-delay-single">' + modelName + ": " + delayVal + "%</span>"
          : '<span class="route-delay-placeholder">— Run Predict delay</span>';
        const baseDurationS = r.duration_s != null ? Number(r.duration_s) : null;
        const adjustedDurationS = hasDelay && baseDurationS != null && Number.isFinite(baseDurationS)
          ? baseDurationS * (1 + delayVal / 100)
          : null;
        const timeWithDelayHtml = adjustedDurationS != null
          ? '<p class="route-time-with-delay">Est. time with delay: <strong>' + formatDurationFromSeconds(adjustedDurationS) + '</strong></p>'
          : '';
        var isBest = bestRouteIndex != null && r.route_index === bestRouteIndex;
        var bestClass = isBest ? " route-card-best" : "";
        return (
          '<div class="route-card' + bestClass + '" data-route-index="' + r.route_index + '">' +
          '<h4 class="route-card-title">Route ' + r.route_index + '</h4>' +
          '<p class="route-meta">' + (r.distance_text || "") + " · " + (r.duration_text || "") + pitstopNote + '</p>' +
          '<div class="route-delays">' + delayHtml + "</div>" +
          timeWithDelayHtml +
          "</div>"
        );
      })
      .join("");
  }

  function drawMap(origin, destination, routes) {
    if (!mapDiv || !mapsApiKey || !window.google || !window.google.maps) return;
    clearMapOverlays();
    if (mapInstance) {
      mapInstance.setCenter({ lat: (origin.lat + destination.lat) / 2, lng: (origin.lng + destination.lng) / 2 });
    } else {
      mapInstance = new google.maps.Map(mapDiv, {
        center: { lat: (origin.lat + destination.lat) / 2, lng: (origin.lng + destination.lng) / 2 },
        zoom: 6,
        mapTypeId: "roadmap",
      });
    }
    const bounds = new google.maps.LatLngBounds();
    [origin, destination].forEach(function (p) {
      bounds.extend(new google.maps.LatLng(p.lat, p.lng));
    });

    var originMarker = new google.maps.Marker({
      position: origin,
      map: mapInstance,
      title: "Tracy Depot",
      label: "S",
    });
    var destMarker = new google.maps.Marker({
      position: destination,
      map: mapInstance,
      title: "Destination",
      label: "D",
    });
    mapMarkers.push(originMarker, destMarker);

    const colors = ["#005DAA", "#0d9488", "#E31837"];
    (routes || []).forEach(function (r, i) {
      const enc = r.polyline;
      if (!enc) return;
      let path = [];
      try {
        if (window.google.maps.geometry && window.google.maps.geometry.encoding) {
          path = google.maps.geometry.encoding.decodePath(enc);
        }
      } catch (e) {}
      if (path.length === 0) return;
      path.forEach(function (p) { bounds.extend(p); });
      var poly = new google.maps.Polyline({
        path: path,
        map: mapInstance,
        strokeColor: colors[i % colors.length],
        strokeWeight: 4,
        strokeOpacity: 0.8,
      });
      mapPolylines.push(poly);
    });
    mapInstance.fitBounds(bounds);
  }

  function onGetRoutes() {
    const dest = getSelectedDestination();
    if (!dest) {
      showError("Please select a destination");
      return;
    }
    if (!routesBtn) return;

    if (routesAbortController) routesAbortController.abort();
    routesAbortController = new AbortController();
    const signal = routesAbortController.signal;

    routesBtn.disabled = true;
    routesBtn.textContent = "Loading…";
    showLoader("Getting routes…");

    const mode = sourceModeSelect ? sourceModeSelect.value : "tracy";
    const resolveOriginPromise = mode === "city"
      ? (function () {
          const city = sourceCityInput ? String(sourceCityInput.value || "").trim() : "";
          if (!city) return Promise.reject(new Error("Please enter a source city name"));
          return geocodeSourceCity(city).then(function (g) {
            setSelectedOrigin({ lat: g.lat, lng: g.lng, name: g.name || city });
            return { lat: g.lat, lng: g.lng };
          });
        })()
      : Promise.resolve({ lat: selectedOrigin.lat, lng: selectedOrigin.lng });

    resolveOriginPromise.then(function (originToUse) {
      return fetch("/api/routes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        origin: originToUse,
        destination: dest,
      }),
      signal: signal,
      });
    })
      .then(function (res) {
        return res.json().then(function (body) {
          if (!res.ok) throw new Error(body.error || "Request failed");
          return body;
        });
      })
      .then(function (body) {
        const routes = body.routes || [];
        lastRoutes = routes;
        lastOrigin = body.origin || { lat: selectedOrigin.lat, lng: selectedOrigin.lng };
        lastDestination = body.destination || dest;
        if (routesResults) routesResults.setAttribute("aria-hidden", "false");
        renderRouteCards(routes);
        if (mapSection) mapSection.setAttribute("aria-hidden", "false");
        if (mapDiv && mapsApiKey && window.google && window.google.maps) {
          drawMap(lastOrigin, lastDestination, routes);
        } else if (mapDiv) {
          mapDiv.innerHTML = "<p class='map-placeholder'>Set GOOGLE_MAPS_API_KEY in backend .env to show the map.</p>";
        }
        if (predictDelayBtn) predictDelayBtn.disabled = false;
      })
      .catch(function (err) {
        if (err.name === "AbortError") return;
        showError(err.message || "Failed to get routes");
      })
      .finally(function () {
        hideLoader();
        routesAbortController = null;
        if (routesBtn) {
          routesBtn.disabled = false;
          routesBtn.textContent = "Get routes";
        }
      });
  }

  function onPredictDelay() {
    if (!lastRoutes || lastRoutes.length === 0) {
      showError("Get routes first");
      return;
    }
    if (!predictDelayBtn) return;
    predictDelayBtn.disabled = true;
    predictDelayBtn.classList.add("btn-loading");
    predictDelayBtn.innerHTML = '<span class="btn-spinner" aria-hidden="true"></span> Calculating…';
    showLoader(PREDICT_DELAY_STEPS);

    fetch("/api/route-delays", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        routes: lastRoutes,
        journey_start_date: document.getElementById("journey-start-date") ? document.getElementById("journey-start-date").value : null,
        journey_start_time: document.getElementById("journey-start-time") ? document.getElementById("journey-start-time").value : null,
      }),
    })
      .then(function (res) {
        return res.json().then(function (body) {
          if (!res.ok) throw new Error(body.error || "Request failed");
          return body;
        });
      })
      .then(function (body) {
        const routesWithDelays = body.routes || [];
        lastRoutes = routesWithDelays;
        renderRouteCards(lastRoutes);
      })
      .catch(function (err) {
        showError(err.message || "Delay prediction failed");
      })
      .finally(function () {
        hideLoader();
        if (predictDelayBtn) {
          predictDelayBtn.classList.remove("btn-loading");
          predictDelayBtn.disabled = lastRoutes == null || lastRoutes.length === 0;
          predictDelayBtn.textContent = "Predict delay";
        }
      });
  }

  // --- Load Maps API if key available ---
  function loadMapsScript(key) {
    if (!key || window.google) return;
    const script = document.createElement("script");
    script.src = "https://maps.googleapis.com/maps/api/js?key=" + encodeURIComponent(key) + "&libraries=geometry";
    script.async = true;
    script.defer = true;
    document.head.appendChild(script);
  }

  fetch("/api/config")
    .then(function (r) { return r.json(); })
    .then(function (data) {
      mapsApiKey = data.mapsApiKey || "";
      if (mapsApiKey) loadMapsScript(mapsApiKey);
    })
    .catch(function () {});

  // --- Manual weather prediction ---
  function getFormData() {
    const data = {};
    FEATURE_IDS.forEach(function (id) {
      const el = document.getElementById(id);
      data[id] = el ? el.value : 0;
    });
    return data;
  }

  function setFormData(data) {
    FEATURE_IDS.forEach(function (id) {
      const el = document.getElementById(id);
      if (el && data[id] != null) el.value = data[id];
    });
  }

  function renderCards(predictions) {
    if (!resultsCards) return;
    const order = ["Ridge", "Random Forest", "XGBoost"];
    const classMap = { Ridge: "ridge", "Random Forest": "random_forest", "XGBoost": "xgboost" };
    const sorted = order.filter(function (name) { return predictions[name] != null; });
    if (sorted.length === 0) {
      resultsCards.innerHTML = "<p>No predictions.</p>";
      return;
    }
    resultsCards.innerHTML = sorted
      .map(function (name) {
        const pct = predictions[name];
        const cls = classMap[name] || "";
        return (
          '<div class="card card-' + cls + '">' +
          '<p class="card-name">' + name + "</p>" +
          '<p class="card-value">' + (typeof pct === "number" ? pct.toFixed(1) : pct) + '</p><p class="card-unit">delay %</p></div>'
        );
      })
      .join("");
  }

  function updateChart(labels, values) {
    if (!chartCanvas) return;
    const colors = ["#005DAA", "#0d9488", "#E31837"];
    if (chartInstance) {
      chartInstance.data.labels = labels;
      chartInstance.data.datasets[0].data = values;
      chartInstance.data.datasets[0].backgroundColor = colors.slice(0, labels.length);
      chartInstance.update();
      return;
    }
    chartInstance = new Chart(chartCanvas, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{ label: "Delay %", data: values, backgroundColor: colors.slice(0, labels.length), borderWidth: 0 }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: function (ctx) { return ctx.raw.toFixed(1) + "% delay"; } } } },
        scales: {
          y: { beginAtZero: true, max: 60, ticks: { callback: function (v) { return v + "%"; } }, grid: { color: "rgba(0,93,170,0.08)" } },
          x: { grid: { display: false }, ticks: { font: { size: 12 } } },
        },
      },
    });
  }

  function onPredict() {
    const data = getFormData();
    if (document.getElementById("demo-journey-start-date")) {
      data.journey_start_date = document.getElementById("demo-journey-start-date").value || null;
    }
    if (document.getElementById("demo-journey-start-time")) {
      data.journey_start_time = document.getElementById("demo-journey-start-time").value || null;
    }
    if (predictBtn) {
      predictBtn.disabled = true;
      predictBtn.textContent = "Predicting…";
    }
    showLoader("Predicting delay…");
    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    })
      .then(function (res) {
        return res.json().then(function (body) {
          if (!res.ok) throw new Error(body.error || "Request failed");
          return body;
        });
      })
      .then(function (body) {
        const predictions = body.predictions || {};
        if (resultsSection) resultsSection.setAttribute("aria-hidden", "false");
        renderCards(predictions);
        const labels = Object.keys(predictions);
        const values = labels.map(function (k) { return predictions[k]; });
        updateChart(labels, values);
      })
      .catch(function (err) { showError(err.message || "Prediction failed"); })
      .finally(function () {
        hideLoader();
        if (predictBtn) {
          predictBtn.disabled = false;
          predictBtn.textContent = "Predict delay";
        }
      });
  }

  // --- Init (main page: source, warehouses, routes; demo page: weather form only) ---
  if (sourceDisplay) loadSource();
  if (sourceModeSelect) {
    sourceModeSelect.addEventListener("change", refreshSourceModeUI);
    refreshSourceModeUI();
  }
  if (sourceGpsBtn) sourceGpsBtn.addEventListener("click", useGpsSource);
  if (destinationSelect) loadWarehouses();

  if (routesBtn) routesBtn.addEventListener("click", onGetRoutes);
  if (predictDelayBtn) predictDelayBtn.addEventListener("click", onPredictDelay);
  if (modelSelect) modelSelect.addEventListener("change", function () {
    if (lastRoutes) renderRouteCards(lastRoutes);
  });
  if (predictBtn) predictBtn.addEventListener("click", onPredict);
  if (form) form.addEventListener("submit", function (e) { e.preventDefault(); onPredict(); });

  document.querySelectorAll(".btn-preset").forEach(function (btn) {
    btn.addEventListener("click", function () {
      const preset = btn.getAttribute("data-preset");
      if (PRESETS[preset]) setFormData(PRESETS[preset]);
    });
  });
})();
