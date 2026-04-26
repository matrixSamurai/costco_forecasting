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

  const mapSection = document.getElementById("map-section");
  const mapDiv = document.getElementById("map");
  const routesResults = document.getElementById("routes-results");
  const routeCards = document.getElementById("route-cards");
  const modelSelect = document.getElementById("model-select");

  let lastRoutes = null;
  let lastOrigin = null;
  let lastDestination = null;
  let lastDecisionData = null;
  let lastRouteLosses = null;
  let selectedOrigin = { lat: TRACY_DEPOT.lat, lng: TRACY_DEPOT.lng, name: "Tracy, CA" };

  function infoTip(text, extraClass) {
    return '<span class="info-tip' + (extraClass ? ' ' + extraClass : '') + '" tabindex="0"><span class="info-tip-icon">i</span><span class="info-tip-content">' + text + '</span></span>';
  }
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
      setSelectedOrigin({ lat: TRACY_DEPOT.lat, lng: TRACY_DEPOT.lng, name: "Tracy, CA" });
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
        setSelectedOrigin({ lat: data.lat, lng: data.lng, name: data.name || "Tracy, CA" });
      })
      .catch(function () {
        setSelectedOrigin({ lat: TRACY_DEPOT.lat, lng: TRACY_DEPOT.lng, name: "Tracy, CA" });
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

  function getBestRoute(routes, modelName) {
    var best = null;
    var bestAdjusted = Infinity;
    routes.forEach(function (r) {
      var baseS = r.duration_s != null ? Number(r.duration_s) : null;
      var delayVal = (r.delays || {})[modelName];
      if (baseS != null && Number.isFinite(baseS) && delayVal != null) {
        var adj = baseS * (1 + delayVal / 100);
        if (adj < bestAdjusted) {
          bestAdjusted = adj;
          best = r;
        }
      }
    });
    return best;
  }

  function getCargoField(name) {
    var el = document.getElementById("cargo-" + name);
    if (!el) return null;
    if (name === "organic") return parseInt(el.value, 10);
    if (name === "quantity_lb") return parseFloat(el.value) || 1000;
    return el.value;
  }

  function renderShipmentDecision(data, bestRoute, modelName, lossMap) {
    var decisionDiv = document.getElementById("shipment-decision");
    if (!decisionDiv) return;
    lossMap = lossMap || {};

    var delayPct = (bestRoute.delays || {})["Ensemble"] != null
      ? (bestRoute.delays || {})["Ensemble"]
      : (bestRoute.delays || {})[modelName];
    var baseDuration = bestRoute.duration_s || 0;
    var delayAdded = baseDuration * (delayPct / 100);
    var bestRouteFuelCost = bestRoute.fuel_cost_usd != null ? Number(bestRoute.fuel_cost_usd) : null;

    var recs = data.recommendations || [];
    var topRec = recs[0] || { level: "info", message: "Proceed with standard shipment schedule." };

    var levelClass = "decision-action-info";
    var levelIcon = "✅";
    if (topRec.level === "critical") { levelClass = "decision-action-critical"; levelIcon = "🚨"; }
    else if (topRec.level === "warning") { levelClass = "decision-action-warning"; levelIcon = "⚠️"; }

    var supplyRisk = data.supply_risk || {};
    var riskLevel = (supplyRisk.risk_level || "unknown").toUpperCase();
    var sourceRegion = supplyRisk.source_region || "";
    var lettuceName = getCargoField("lettuce_type") || "romaine";
    var qtyLb = getCargoField("quantity_lb") || 1000;

    // Find route with lowest total cost (revenue_loss + fuel_cost)
    var lowestLossRouteIndex = null;
    var lowestTotalCost = Infinity;
    Object.keys(lossMap).forEach(function (idx) {
      var ld = lossMap[idx];
      var routeObj = (lastRoutes || []).filter(function (x) { return x.route_index === Number(idx); })[0];
      var fc = routeObj && routeObj.fuel_cost_usd != null ? Number(routeObj.fuel_cost_usd) : 0;
      if (ld && ld.revenue_loss != null) {
        var totalCost = Number(ld.revenue_loss) + fc;
        if (totalCost < lowestTotalCost) {
          lowestTotalCost = totalCost;
          lowestLossRouteIndex = Number(idx);
        }
      }
    });

    // Build tradeoff note if lowest-cost route differs from fastest
    var tradeoffHtml = "";
    if (lowestLossRouteIndex !== null && lowestLossRouteIndex !== bestRoute.route_index) {
      var bestTotalCost = Number(data.revenue_loss) + (bestRouteFuelCost || 0);
      var saving = bestTotalCost - lowestTotalCost;
      var lowestLossRouteData = (lastRoutes || []).filter(function (r) { return r.route_index === lowestLossRouteIndex; })[0];
      var timeDiffStr = "";
      if (lowestLossRouteData) {
        var lowestLossDelayPct = (lowestLossRouteData.delays || {})["Ensemble"] != null
          ? (lowestLossRouteData.delays || {})["Ensemble"]
          : (lowestLossRouteData.delays || {})[modelName] || 0;
        var lowestAdjusted = Number(lowestLossRouteData.duration_s) * (1 + lowestLossDelayPct / 100);
        var fastestAdjusted = baseDuration * (1 + delayPct / 100);
        var timeDiff = lowestAdjusted - fastestAdjusted;
        if (timeDiff > 0) timeDiffStr = " but takes " + formatDurationFromSeconds(timeDiff) + " longer";
      }
      tradeoffHtml = '<p class="decision-tradeoff">📊 Route ' + lowestLossRouteIndex + ' has $' + saving.toFixed(0) + ' lower total cost ($' + lowestTotalCost.toFixed(0) + ' vs $' + bestTotalCost.toFixed(0) + ')' + timeDiffStr + '.</p>';
    }

    var lossParams = new URLSearchParams({
      lettuce_type: lettuceName,
      organic: String(getCargoField("organic") || 0),
      quantity_lb: String(qtyLb),
      month: String(new Date().getMonth() + 1),
      delay_hours: (delayAdded / 3600).toFixed(2),
      delay_temp_f: "41",
      transit_distance_km: String(Math.round((bestRoute.distance_m || 0) / 1000)),
      pre_delay_consumption_pct: "0",
    });
    var lossUrl = "/loss?" + lossParams.toString();

    var adjustedDuration = formatDurationFromSeconds(baseDuration * (1 + delayPct / 100));
    var delayAddedStr = formatDurationFromSeconds(delayAdded);
    decisionDiv.innerHTML =
      '<div class="decision-card">' +

      // Header bar
      '<div class="decision-header">' +
      '<span class="decision-header-title">Shipment Decision</span>' +
      '<a href="' + lossUrl + '" class="decision-link">Full loss report →</a>' +
      '</div>' +

      // Stats row: route is hero, others are supporting
      '<div class="decision-body">' +

      '<div class="decision-stat decision-stat-hero">' +
        '<span class="decision-stat-label">Recommended Route</span>' +
        '<span class="decision-stat-value">Route ' + bestRoute.route_index + '</span>' +
        '<span class="decision-stat-sub">' + (bestRoute.distance_text || "") + ' · est. ' + adjustedDuration + '</span>' +
      '</div>' +

      '<div class="decision-stat">' +
        '<span class="decision-stat-label">Weather Delay</span>' +
        '<span class="decision-stat-value decision-stat-delay">' + delayPct + '%</span>' +
        '<span class="decision-stat-sub">+' + delayAddedStr + ' added</span>' +
      '</div>' +

      '<div class="decision-stat">' +
        '<span class="decision-stat-label">Spoilage Loss' + infoTip("Estimated revenue lost to produce spoilage caused by the weather-related transit delay. Calculated from delay duration, in-transit temperature, cargo weight, lettuce type, and seasonal price index.") + '</span>' +
        '<span class="decision-stat-value decision-stat-money">$' + (data.revenue_loss != null ? Number(data.revenue_loss).toFixed(0) : "—") + '</span>' +
        '<span class="decision-stat-sub">' + qtyLb + ' lb ' + lettuceName +
          (bestRouteFuelCost != null ? ' · $' + bestRouteFuelCost.toFixed(0) + ' est. fuel' : '') +
        '</span>' +
      '</div>' +

      '<div class="decision-stat">' +
        '<span class="decision-stat-label">Supply Risk</span>' +
        '<span class="decision-stat-value decision-stat-supply-' + (supplyRisk.risk_level || "unknown").toLowerCase() + '">' + riskLevel + '</span>' +
        '<span class="decision-stat-sub">' + (sourceRegion || "—") + '</span>' +
      '</div>' +

      '</div>' +

      // Recommendation footer strip
      '<div class="decision-footer ' + levelClass + '">' + levelIcon + ' ' + topRec.message +
      (tradeoffHtml ? ' &nbsp;·&nbsp; ' + tradeoffHtml : '') +
      '</div>' +

      '</div>';
    decisionDiv.setAttribute("aria-hidden", "false");
  }

  function fetchAllRouteLosses(routes, modelName) {
    var journeyDateEl = document.getElementById("journey-start-date");
    var journeyDateVal = journeyDateEl ? journeyDateEl.value : "";
    var month = journeyDateVal
      ? new Date(journeyDateVal + "T00:00:00").getMonth() + 1
      : new Date().getMonth() + 1;
    var lettuceType = getCargoField("lettuce_type") || "romaine";
    var organic = getCargoField("organic") || 0;
    var qtyLb = getCargoField("quantity_lb") || 1000;

    var promises = routes.map(function (r) {
      var ensembleDelay = (r.delays || {})["Ensemble"];
      var delayPct = ensembleDelay != null ? ensembleDelay : (r.delays || {})[modelName];
      if (delayPct == null || r.duration_s == null) {
        return Promise.resolve({ route_index: r.route_index, error: true });
      }
      var delayHours = (Number(r.duration_s) / 3600) * (delayPct / 100);
      var distanceKm = (r.distance_m || 0) / 1000;
      return fetch("/api/predict-loss", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          delay_hours: delayHours,
          delay_temp_f: 41,
          lettuce_type: lettuceType,
          organic: organic,
          quantity_lb: qtyLb,
          month: month,
          transit_distance_km: distanceKm,
          pre_delay_consumption_pct: 0,
        }),
      })
        .then(function (res) { return res.json(); })
        .then(function (data) { return { route_index: r.route_index, data: data }; })
        .catch(function () { return { route_index: r.route_index, error: true }; });
    });

    return Promise.all(promises).then(function (results) {
      var map = {};
      results.forEach(function (r) { if (!r.error) map[r.route_index] = r.data; });
      return map;
    });
  }

  function fetchAndRenderAllLosses(routes, modelName) {
    var decisionDiv = document.getElementById("shipment-decision");
    if (decisionDiv) {
      decisionDiv.innerHTML = '<p class="decision-loading">Calculating revenue risk for all routes…</p>';
      decisionDiv.setAttribute("aria-hidden", "false");
    }
    fetchAllRouteLosses(routes, "Ensemble")
      .then(function (lossMap) {
        lastRouteLosses = lossMap;
        renderRouteCards(routes, lossMap);
        var best = getBestRoute(routes, "Ensemble");
        if (best && lossMap[best.route_index]) {
          lastDecisionData = { data: lossMap[best.route_index], bestRoute: best, modelName: "Ensemble" };
          renderShipmentDecision(lossMap[best.route_index], best, "Ensemble", lossMap);
        } else if (decisionDiv) {
          decisionDiv.setAttribute("aria-hidden", "true");
        }
      })
      .catch(function () {
        if (decisionDiv) decisionDiv.setAttribute("aria-hidden", "true");
      });
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

  function renderRouteCards(routes, routeLosses) {
    if (!routeCards) return;
    if (!routes || routes.length === 0) {
      routeCards.innerHTML = "<p>No routes.</p>";
      return;
    }
    var modelName = "Ensemble";
    routeLosses = routeLosses || {};

    // Fastest route = shortest adjusted duration (using Ensemble delay)
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

    // Lowest total cost route (revenue_loss + fuel_cost)
    var lowestLossRouteIndex = null;
    var lowestTotalCost = Infinity;
    routes.forEach(function (r) {
      var ld = routeLosses[r.route_index];
      var fuelCost = r.fuel_cost_usd != null ? Number(r.fuel_cost_usd) : 0;
      if (ld && ld.revenue_loss != null) {
        var totalCost = Number(ld.revenue_loss) + fuelCost;
        if (totalCost < lowestTotalCost) {
          lowestTotalCost = totalCost;
          lowestLossRouteIndex = r.route_index;
        }
      }
    });
    var showLowestLossBadge = lowestLossRouteIndex !== null && lowestLossRouteIndex !== bestRouteIndex;

    // Compute max total duration for bar scaling
    var maxTotalS = 0;
    routes.forEach(function (r) {
      var baseS = r.duration_s != null ? Number(r.duration_s) : 0;
      var delayVal = (r.delays || {})[modelName];
      var totalS = delayVal != null ? baseS * (1 + delayVal / 100) : baseS;
      if (totalS > maxTotalS) maxTotalS = totalS;
    });
    if (maxTotalS === 0) maxTotalS = 1;

    // Journey start time for arrival calculation
    var startTimeEl = document.getElementById("journey-start-time");
    var startTimeVal = startTimeEl ? startTimeEl.value : "";

    function arrivalTime(adjustedS) {
      if (!startTimeVal || !adjustedS) return "";
      var p = startTimeVal.split(":");
      var startMin = (parseInt(p[0]) || 0) * 60 + (parseInt(p[1]) || 0);
      var arrMin = startMin + Math.round(adjustedS / 60);
      var days = Math.floor(arrMin / 1440);
      var h = Math.floor((arrMin % 1440) / 60);
      var m = arrMin % 60;
      return (days > 0 ? "+" + days + "d " : "") +
        (h < 10 ? "0" : "") + h + ":" + (m < 10 ? "0" : "") + m;
    }

    var anyDelay = routes.some(function (r) { return (r.delays || {})[modelName] != null; });

    var html = '<div class="route-timeline">' +
      '<div class="tl-header">' +
      '<div class="tl-col-label"></div>' +
      '<div class="tl-col-bar"><span class="tl-legend"><span class="tl-legend-dot tl-legend-base"></span>Base journey</span><span class="tl-legend"><span class="tl-legend-dot tl-legend-delay"></span>Delay added</span></div>' +
      '<div class="tl-col-stats">' +
        '<span>Delay</span>' +
        '<span>Est. total</span>' +
        '<span>Fuel cost' + infoTip("Estimated fuel cost based on route distance. Adjusted for weather conditions (bad weather reduces MPG) and seasonal diesel price variation.") + '</span>' +
        '<span>Spoilage Loss' + infoTip("Estimated revenue lost to produce spoilage caused by the weather-related transit delay. Calculated from delay duration, in-transit temperature, cargo weight, lettuce type, and seasonal price index.") + '</span>' +
      '</div>' +
      '</div>';

    routes.forEach(function (r) {
      var baseS = r.duration_s != null ? Number(r.duration_s) : 0;
      var delayVal = (r.delays || {})[modelName];
      var hasDelay = delayVal != null;
      var adjustedS = hasDelay ? baseS * (1 + delayVal / 100) : baseS;
      var baseW = ((baseS / maxTotalS) * 100).toFixed(1);
      var delayW = hasDelay ? (((adjustedS - baseS) / maxTotalS) * 100).toFixed(1) : "0";

      var isBest = bestRouteIndex != null && r.route_index === bestRouteIndex;
      var isLowestLoss = showLowestLossBadge && r.route_index === lowestLossRouteIndex;
      var rowClass = "tl-row" + (isBest ? " tl-row-best" : "") + (isLowestLoss ? " tl-row-lowest-loss" : "");

      var lossData = routeLosses[r.route_index];
      var lossText = lossData && lossData.revenue_loss != null
        ? "$" + Number(lossData.revenue_loss).toFixed(0)
        : (hasDelay ? "<span class='tl-loading'>…</span>" : "<span class='tl-empty'>—</span>");

      var fuelCost = r.fuel_cost_usd != null ? Number(r.fuel_cost_usd) : null;
      var fuelText = fuelCost != null ? "$" + fuelCost.toFixed(0) : "<span class='tl-empty'>—</span>";

      var delayText = hasDelay ? delayVal + "%" : "<span class='tl-empty'>—</span>";
      var durationText = hasDelay ? formatDurationFromSeconds(adjustedS) : (r.duration_text || "—");
      var arrival = arrivalTime(adjustedS);
      var labelPrefix = isBest ? "★ " : isLowestLoss ? "💰 " : "";
      var stops = r.pitstop_count != null ? r.pitstop_count + " stops" : "";

      html +=
        '<div class="' + rowClass + '" data-route-index="' + r.route_index + '">' +
        '<div class="tl-col-label">' +
          '<span class="tl-route-name">' + labelPrefix + 'Route ' + r.route_index + '</span>' +
          '<span class="tl-route-meta">' + (r.distance_text || "") + (stops ? " · " + stops : "") + '</span>' +
        '</div>' +
        '<div class="tl-col-bar">' +
          '<div class="tl-bar">' +
            '<div class="tl-bar-base" style="width:' + baseW + '%"></div>' +
            (parseFloat(delayW) > 0 ? '<div class="tl-bar-delay" style="width:' + delayW + '%"></div>' : '') +
          '</div>' +
          '<div class="tl-bar-labels"><span>' + (r.duration_text || "") + '</span>' + (hasDelay ? '<span class="tl-bar-label-delay">+' + formatDurationFromSeconds(adjustedS - baseS) + ' delay</span>' : '') + '</div>' +
        '</div>' +
        '<div class="tl-col-stats">' +
          '<span class="tl-stat-delay' + (hasDelay && delayVal > 25 ? " tl-stat-high" : "") + '">' + delayText + '</span>' +
          '<span class="tl-stat-duration">' + durationText + (arrival ? '<span class="tl-arrival"> → ' + arrival + '</span>' : '') + '</span>' +
          '<span class="tl-stat-fuel">' + fuelText + '</span>' +
          '<span class="tl-stat-loss">' + lossText + '</span>' +
        '</div>' +
        '</div>';
    });

    html += '</div>';
    routeCards.innerHTML = html;
  }

  function drawMap(origin, destination, routes, bestRouteIndex) {
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
      position: origin, map: mapInstance, title: "Origin", label: "S",
    });
    var destMarker = new google.maps.Marker({
      position: destination, map: mapInstance, title: "Destination", label: "D",
    });
    mapMarkers.push(originMarker, destMarker);

    const colors = ["#005DAA", "#0d9488", "#E31837"];
    // Draw non-best routes first so best renders on top
    var sorted = (routes || []).slice().sort(function (a, b) {
      var aIsBest = a.route_index === bestRouteIndex ? 1 : 0;
      var bIsBest = b.route_index === bestRouteIndex ? 1 : 0;
      return aIsBest - bIsBest;
    });
    sorted.forEach(function (r, _i) {
      var i = (routes || []).indexOf(r);
      var isBest = bestRouteIndex != null && r.route_index === bestRouteIndex;
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
        strokeWeight: isBest ? 6 : 3,
        strokeOpacity: isBest ? 1.0 : 0.45,
        zIndex: isBest ? 10 : 1,
      });
      mapPolylines.push(poly);
    });
    mapInstance.fitBounds(bounds);
  }

  function onGetRoutes() {
    const dest = getSelectedDestination();
    if (!dest) { showError("Please select a destination"); return; }
    if (!routesBtn) return;

    if (routesAbortController) routesAbortController.abort();
    routesAbortController = new AbortController();
    const signal = routesAbortController.signal;

    routesBtn.disabled = true;
    routesBtn.innerHTML = '<span class="btn-spinner" aria-hidden="true"></span> Calculating…';
    routesBtn.classList.add("btn-loading");

    var journeyStartDate = document.getElementById("journey-start-date") ? document.getElementById("journey-start-date").value : null;
    var journeyStartTime = document.getElementById("journey-start-time") ? document.getElementById("journey-start-time").value : null;

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

    // Step 1 — fetch routes
    showLoader("Getting routes…");
    resolveOriginPromise
      .then(function (originToUse) {
        return fetch("/api/routes", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ origin: originToUse, destination: dest }),
          signal: signal,
        });
      })
      .then(function (res) {
        return res.json().then(function (body) {
          if (!res.ok) throw new Error(body.error || "Request failed");
          return body;
        });
      })
      .then(function (routeBody) {
        lastRoutes = routeBody.routes || [];
        lastOrigin = routeBody.origin || { lat: selectedOrigin.lat, lng: selectedOrigin.lng };
        lastDestination = routeBody.destination || dest;

        // Show map immediately with base routes
        if (mapSection) mapSection.setAttribute("aria-hidden", "false");
        if (mapDiv && mapsApiKey && window.google && window.google.maps) {
          drawMap(lastOrigin, lastDestination, lastRoutes, null);
        } else if (mapDiv) {
          mapDiv.innerHTML = "<p class='map-placeholder'>Set GOOGLE_MAPS_API_KEY in backend .env to show the map.</p>";
        }

        // Step 2 — predict delays
        showLoader(PREDICT_DELAY_STEPS);
        return fetch("/api/route-delays", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            routes: lastRoutes,
            journey_start_date: journeyStartDate,
            journey_start_time: journeyStartTime,
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
      .then(function (delayBody) {
        lastRoutes = delayBody.routes || [];
        lastRouteLosses = null;
        var currentModel = "Ensemble";
        var bestRoute = getBestRoute(lastRoutes, currentModel);
        var bestIdx = bestRoute ? bestRoute.route_index : null;

        // Render all sections
        if (delayBody.classification) renderRouteClassification(delayBody.classification);
        if (routesResults) routesResults.setAttribute("aria-hidden", "false");
        renderRouteCards(lastRoutes, null);

        // Re-draw map with best route highlighted + legend
        if (mapDiv && mapsApiKey && window.google && window.google.maps) {
          drawMap(lastOrigin, lastDestination, lastRoutes, bestIdx);
        }
        renderMapLegend(lastRoutes, bestIdx, "Ensemble");

        fetchAndRenderAllLosses(lastRoutes, "Ensemble");
      })
      .catch(function (err) {
        if (err.name === "AbortError") return;
        showError(err.message || "Failed to get routes and delays");
      })
      .finally(function () {
        hideLoader();
        routesAbortController = null;
        if (routesBtn) {
          routesBtn.disabled = false;
          routesBtn.classList.remove("btn-loading");
          routesBtn.textContent = "Show Routes & Predict Delay";
        }
      });
  }

  function renderMapLegend(routes, bestRouteIndex, modelName) {
    var legendDiv = document.getElementById("map-legend");
    if (!legendDiv) return;
    var colors = ["#005DAA", "#0d9488", "#E31837"];
    legendDiv.innerHTML = routes.map(function (r, i) {
      var isBest = r.route_index === bestRouteIndex;
      var delayVal = (r.delays || {})[modelName];
      var delayText = delayVal != null ? delayVal + "% delay" : "";
      return '<div class="map-legend-row' + (isBest ? " map-legend-best" : "") + '">' +
        '<span class="map-legend-swatch" style="background:' + colors[i % colors.length] + '"></span>' +
        '<span class="map-legend-label">' + (isBest ? "★ " : "") + "Route " + r.route_index + "</span>" +
        '<span class="map-legend-meta">' + (r.distance_text || "") + (delayText ? " · " + delayText : "") + "</span>" +
        "</div>";
    }).join("");
    legendDiv.setAttribute("aria-hidden", "false");
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

  function renderClassification(data) {
    var section = document.getElementById("classification-section");
    var banner = document.getElementById("classification-banner");
    var details = document.getElementById("classification-details");
    if (!section || !banner || !details) return;
    section.setAttribute("aria-hidden", "false");
    var hasDelay = data.has_delay;
    var prob = data.probability;
    var modelName = data.model_name || "XGBoost";
    if (hasDelay) {
      banner.className = "classification-banner has-delay";
      banner.innerHTML = '<span class="banner-icon">&#9888;</span><span>Severe Weather Likely &mdash; Running delay model (probability: ' + (prob * 100).toFixed(1) + '%)</span>';
    } else {
      banner.className = "classification-banner no-delay";
      banner.innerHTML = '<span class="banner-icon">&#10003;</span><span>No Severe Weather &mdash; Delay = 0%, Model 1 skipped (probability: ' + (prob * 100).toFixed(1) + '%)</span>';
    }
    details.innerHTML =
      '<div class="detail-card"><p class="detail-label">Prediction</p><p class="detail-value ' + (hasDelay ? "positive" : "negative") + '">' + (hasDelay ? "Severe Weather" : "Clear") + '</p></div>' +
      '<div class="detail-card"><p class="detail-label">Probability</p><p class="detail-value ' + (hasDelay ? "positive" : "negative") + '">' + (prob * 100).toFixed(1) + '%</p></div>' +
      '<div class="detail-card"><p class="detail-label">Classifier</p><p class="detail-value">' + modelName + '</p></div>';
  }

  function renderRouteClassification(data) {
    var section = document.getElementById("classification-section");
    var banner = document.getElementById("classification-banner");
    var details = document.getElementById("classification-details");
    if (!section || !banner || !details) return;
    section.setAttribute("aria-hidden", "false");
    var hasDelay = data.has_delay;
    var avgProb = data.probability || 0;
    var severePitstops = data.severe_pitstops || 0;
    var totalPitstops = data.total_pitstops || 0;
    var severePct = totalPitstops > 0 ? ((severePitstops / totalPitstops) * 100).toFixed(0) : "0";
    var riskLevel = data.risk_level || (hasDelay ? "High" : "Low");
    var riskClass = { "Low": "negative", "Moderate": "moderate", "High": "positive", "Critical": "positive" }[riskLevel] || "positive";

    if (hasDelay) {
      banner.className = "classification-banner has-delay";
      banner.innerHTML = '<span class="banner-icon">&#9888;</span><span>' + riskLevel + ' Weather Risk &mdash; ' + severePct + '% of your route (' + severePitstops + ' of ' + totalPitstops + ' stops) in severe weather zones</span>';
    } else {
      banner.className = "classification-banner no-delay";
      banner.innerHTML = '<span class="banner-icon">&#10003;</span><span>Route Looks Clear &mdash; All ' + totalPitstops + ' stops checked, weather risk is low</span>';
    }
    details.innerHTML =
      '<div class="detail-card"><p class="detail-label">Risk Level' + infoTip('<b>Route-level severity</b> (by highest-probability stop):<br><br><b>Low</b> &mdash; No severe weather detected<br><b>Moderate</b> &mdash; Risk present; max stop prob &lt; 70%<br><b>High</b> &mdash; Max stop prob &ge; 70%<br><b>Critical</b> &mdash; Max stop prob &ge; 85%; consider delaying', 'info-tip-wide') + '</p><p class="detail-value ' + riskClass + '">' + riskLevel + '</p></div>' +
      '<div class="detail-card"><p class="detail-label">Avg Weather Risk' + infoTip("Average probability of a weather-related delay across all checkpoints sampled every ~20 miles along the route. Values above 50% indicate meaningful risk at a typical stop.") + '</p><p class="detail-value ' + (hasDelay ? "positive" : "negative") + '">' + (avgProb * 100).toFixed(1) + '%</p></div>' +
      '<div class="detail-card"><p class="detail-label">Route Affected' + infoTip("Percentage of route checkpoints where severe weather conditions are predicted. A higher value means more of the route is exposed to adverse weather.") + '</p><p class="detail-value ' + (hasDelay ? "positive" : "negative") + '">' + severePct + '% of stops</p></div>' +
      '<div class="detail-card"><p class="detail-label">Stops at Risk' + infoTip("Number of checkpoints (out of all sampled stops) where the weather delay probability exceeds 50%.") + '</p><p class="detail-value">' + severePitstops + ' / ' + totalPitstops + '</p></div>';
  }

  function onPredict() {
    var data = getFormData();
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
    showLoader("Classifying weather & predicting delay…");
    var classifyBody = {
      weather: {
        temp_mean: ((parseFloat(data.temp_min_mean) || 0) + (parseFloat(data.temp_max_mean) || 0)) / 2,
        temp_min: parseFloat(data.temp_min_mean) || 0,
        temp_max: parseFloat(data.temp_max_mean) || 0,
        prcp_total: parseFloat(data.prcp_total_mean) || 0,
        snow_depth: parseFloat(data.snow_depth_mean) || 0,
        visibility: parseFloat(data.visibility_mean) || 10,
        wind_speed_mean: parseFloat(data.wind_speed_mean) || 0,
        wind_gust_max: parseFloat(data.wind_gust_max_mean) || 0,
      },
      date: data.journey_start_date || null,
      model: "all",
    };
    var classifyPromise = fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(classifyBody),
    }).then(function (res) {
      return res.json().then(function (body) {
        if (!res.ok) throw new Error(body.error || "Classification failed");
        return body;
      });
    });
    var predictPromise = fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }).then(function (res) {
      return res.json().then(function (body) {
        if (!res.ok) throw new Error(body.error || "Request failed");
        return body;
      });
    });
    Promise.all([classifyPromise, predictPromise])
      .then(function (results) {
        var classifyResult = results[0];
        var predictResult = results[1];
        var classData = classifyResult.classifications
          ? (classifyResult.classifications["XGBoost"] || Object.values(classifyResult.classifications)[0])
          : classifyResult;
        renderClassification(classData);
        var predictions = predictResult.predictions || {};
        if (resultsSection) resultsSection.setAttribute("aria-hidden", "false");
        renderCards(predictions);
        var labels = Object.keys(predictions);
        var values = labels.map(function (k) { return predictions[k]; });
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
  if (predictBtn) predictBtn.addEventListener("click", onPredict);
  if (form) form.addEventListener("submit", function (e) { e.preventDefault(); onPredict(); });

  document.querySelectorAll(".btn-preset").forEach(function (btn) {
    btn.addEventListener("click", function () {
      const preset = btn.getAttribute("data-preset");
      if (PRESETS[preset]) setFormData(PRESETS[preset]);
    });
  });
})();
