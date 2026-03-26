(function () {
  "use strict";

  var PRESETS = {
    mild:     { delay_hours: 0.5,  delay_temp_f: 32, lettuce_type: "iceberg",  organic: "0" },
    moderate: { delay_hours: 24,   delay_temp_f: 41, lettuce_type: "romaine",  organic: "0" },
    severe:   { delay_hours: 72,   delay_temp_f: 55, lettuce_type: "romaine",  organic: "0" },
  };

  // Harvest region lookup: month → source region, avg transit hours to Tracy CA, pre-delay %
  // CA Salinas Apr-Oct (~130 km, 2h transit), AZ Yuma Nov-Mar (~800 km, 10h transit)
  var HARVEST_REGIONS = {
    1:  { region: "Yuma, AZ",        transit_km: 800, pre_delay_pct: 8 },
    2:  { region: "Yuma, AZ",        transit_km: 800, pre_delay_pct: 8 },
    3:  { region: "Yuma, AZ",        transit_km: 800, pre_delay_pct: 8 },
    4:  { region: "Salinas, CA",     transit_km: 200, pre_delay_pct: 3 },
    5:  { region: "Salinas, CA",     transit_km: 200, pre_delay_pct: 3 },
    6:  { region: "Salinas, CA",     transit_km: 200, pre_delay_pct: 3 },
    7:  { region: "Salinas, CA",     transit_km: 200, pre_delay_pct: 3 },
    8:  { region: "Salinas, CA",     transit_km: 200, pre_delay_pct: 3 },
    9:  { region: "Salinas, CA",     transit_km: 200, pre_delay_pct: 3 },
    10: { region: "Salinas, CA",     transit_km: 200, pre_delay_pct: 3 },
    11: { region: "Yuma, AZ",        transit_km: 800, pre_delay_pct: 8 },
    12: { region: "Yuma, AZ",        transit_km: 800, pre_delay_pct: 8 },
  };

  var FIELD_IDS = [
    "lettuce_type", "organic", "quantity_lb", "month",
    "delay_hours", "delay_temp_f", "transit_distance_km", "pre_delay_consumption_pct",
  ];

  var predictBtn = document.getElementById("predict-loss-btn");
  var resultsSection = document.getElementById("results-section");
  var lossCards = document.getElementById("loss-cards");
  var chartCanvas = document.getElementById("chart");
  var detailWrap = document.getElementById("detail-table-wrap");
  var recsWrap = document.getElementById("recommendations-wrap");
  var errorToast = document.getElementById("error-toast");
  var loaderOverlay = document.getElementById("loader-overlay");
  var loaderMessage = document.getElementById("loader-message");

  var chartInstance = null;

  function showError(msg) {
    if (!errorToast) return;
    errorToast.textContent = msg;
    errorToast.setAttribute("aria-hidden", "false");
    setTimeout(function () { errorToast.setAttribute("aria-hidden", "true"); }, 6000);
  }

  function showLoader(msg) {
    if (!loaderOverlay) return;
    if (loaderMessage) loaderMessage.textContent = msg || "Loading…";
    loaderOverlay.setAttribute("aria-hidden", "false");
  }

  function hideLoader() {
    if (loaderOverlay) loaderOverlay.setAttribute("aria-hidden", "true");
  }

  function getFormData() {
    var data = {};
    FIELD_IDS.forEach(function (id) {
      var el = document.getElementById(id);
      if (el) data[id] = el.value;
    });
    ["quantity_lb", "month", "delay_hours", "delay_temp_f",
     "transit_distance_km", "pre_delay_consumption_pct", "organic"].forEach(function (k) {
      if (data[k] != null) data[k] = Number(data[k]);
    });
    return data;
  }

  function setFormFields(preset) {
    Object.keys(preset).forEach(function (key) {
      var el = document.getElementById(key);
      if (el) el.value = preset[key];
    });
  }

  function numberFmt(x) {
    if (x == null) return "—";
    return Number(x).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  var RISK_COLORS = { high: "#E31837", medium: "#d97706", low: "#0d9488" };

  function renderLossCards(result) {
    if (!lossCards) return;
    var lossPercent = (result.loss_rate * 100).toFixed(1);
    var sr = result.supply_risk || {};
    var riskLevel = (sr.risk_level || "low").toLowerCase();
    var riskColor = RISK_COLORS[riskLevel] || "#0d9488";
    var riskLabel = riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1);

    var cards = [
      { label: "Loss Rate (XGB)", value: lossPercent + "%", color: "#E31837" },
      { label: "Revenue Loss", value: "$" + numberFmt(result.revenue_loss), color: "#005DAA" },
      { label: "Shelf Life", value: result.shelf_life_days.toFixed(1) + " days", color: "#0d9488" },
      { label: "Supply Risk", value: riskLabel, color: riskColor,
        sub: sr.source_region || "" },
    ];
    lossCards.innerHTML = cards.map(function (c) {
      var subHtml = c.sub ? '<p class="card-sub">' + c.sub + '</p>' : '';
      return '<div class="loss-card" style="border-left-color:' + c.color + '">' +
        '<p class="card-name">' + c.label + '</p>' +
        '<p class="card-value">' + c.value + '</p>' +
        subHtml +
        '</div>';
    }).join("");
  }

  function renderRecommendations(result) {
    if (!recsWrap) return;
    var recs = result.recommendations || [];
    if (recs.length === 0) {
      recsWrap.innerHTML = "";
      return;
    }
    var html = '<h3 class="recs-title">Recommendations</h3>';
    html += '<div class="recs-list">';
    recs.forEach(function (r) {
      var level = (r.level || "info").toLowerCase();
      html += '<div class="rec-item rec-' + level + '">' +
        '<span class="rec-badge">' + level.toUpperCase() + '</span>' +
        '<span class="rec-message">' + r.message + '</span>' +
        '</div>';
    });
    html += '</div>';
    recsWrap.innerHTML = html;
  }

  function renderLossChart(result) {
    if (!chartCanvas) return;
    var labels = ["XGBoost", "Sigmoid"];
    var values = [
      +(result.loss_rate * 100).toFixed(2),
      +(result.sigmoid_loss_rate * 100).toFixed(2),
    ];
    var bgColors = ["#E31837", "#005DAA"];

    if (chartInstance) {
      chartInstance.data.labels = labels;
      chartInstance.data.datasets[0].data = values;
      chartInstance.data.datasets[0].backgroundColor = bgColors;
      chartInstance.update();
      return;
    }
    chartInstance = new Chart(chartCanvas, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "Loss Rate %",
          data: values,
          backgroundColor: bgColors,
          borderWidth: 0,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) { return ctx.raw.toFixed(1) + "% loss"; },
            },
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            ticks: { callback: function (v) { return v + "%"; } },
            grid: { color: "rgba(0,93,170,0.08)" },
          },
          x: { grid: { display: false }, ticks: { font: { size: 13 } } },
        },
      },
    });
  }

  function renderDetailTable(result) {
    if (!detailWrap) return;
    var rows = [
      ["Lettuce Type", result.lettuce_type || "—"],
      ["Organic", result.organic ? "Yes" : "No"],
      ["Price per lb", "$" + (result.price_per_lb || 0).toFixed(2)],
      ["Total Shipment Value", "$" + numberFmt(result.total_value)],
      ["Delay Hours", (result.delay_hours || 0).toFixed(1) + " h"],
      ["Temperature", (result.delay_temp_f || 0).toFixed(0) + " °F"],
      ["Shelf Life", (result.shelf_life_days || 0).toFixed(1) + " days"],
      ["Consumption %", (result.consumption_pct || 0).toFixed(1) + "%" + (result.consumption_pct > 100 ? " (exceeded shelf life — not marketable)" : "")],
      ["Loss Rate (XGBoost)", (result.loss_rate * 100).toFixed(2) + "%"],
      ["Loss Rate (Sigmoid)", (result.sigmoid_loss_rate * 100).toFixed(2) + "%"],
      ["Seasonal Price Index", (result.seasonal_price_index || 1).toFixed(4)],
      ["Revenue Loss", "$" + numberFmt(result.revenue_loss)],
      ["Seasonal Revenue Loss", "$" + numberFmt(result.revenue_loss_seasonal)],
    ];
    var sr = result.supply_risk;
    if (sr) {
      rows.push(["Supply Index", (sr.supply_index * 100).toFixed(0) + "%"]);
      rows.push(["Source Region", sr.source_region || "—"]);
      rows.push(["Supply Risk Level", (sr.risk_level || "—").toUpperCase()]);
    }
    var html = '<table class="detail-table"><tbody>';
    rows.forEach(function (row) {
      html += "<tr><td>" + row[0] + "</td><td><strong>" + row[1] + "</strong></td></tr>";
    });
    html += "</tbody></table>";
    detailWrap.innerHTML = html;
  }

  function onPredictLoss() {
    var data = getFormData();
    if (predictBtn) {
      predictBtn.disabled = true;
      predictBtn.textContent = "Predicting…";
    }
    showLoader("Predicting revenue loss…");

    fetch("/api/predict-loss", {
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
      .then(function (result) {
        if (resultsSection) resultsSection.setAttribute("aria-hidden", "false");
        renderLossCards(result);
        renderLossChart(result);
        renderRecommendations(result);
        renderDetailTable(result);
      })
      .catch(function (err) {
        showError(err.message || "Prediction failed");
      })
      .finally(function () {
        hideLoader();
        if (predictBtn) {
          predictBtn.disabled = false;
          predictBtn.textContent = "Predict loss";
        }
      });
  }

  if (predictBtn) predictBtn.addEventListener("click", onPredictLoss);

  var lossForm = document.getElementById("loss-form");
  if (lossForm) lossForm.addEventListener("submit", function (e) { e.preventDefault(); onPredictLoss(); });

  // Auto-fill pre_delay and transit_distance when month changes
  var monthSelect = document.getElementById("month");
  var harvestHint = document.getElementById("harvest-hint");
  function updateHarvestDefaults() {
    var month = parseInt(monthSelect.value, 10);
    var info = HARVEST_REGIONS[month];
    if (!info) return;
    var preDelayEl = document.getElementById("pre_delay_consumption_pct");
    var transitEl = document.getElementById("transit_distance_km");
    if (preDelayEl) preDelayEl.value = info.pre_delay_pct;
    if (transitEl) transitEl.value = info.transit_km;
    if (harvestHint) harvestHint.textContent = "Source: " + info.region;
  }
  if (monthSelect) {
    monthSelect.addEventListener("change", updateHarvestDefaults);
    updateHarvestDefaults(); // set initial values
  }

  document.querySelectorAll(".btn-preset").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var preset = btn.getAttribute("data-preset");
      if (PRESETS[preset]) setFormFields(PRESETS[preset]);
    });
  });
})();
