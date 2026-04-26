/**
 * Adds weekly weather (52 weeks) to each path coordinate in routes_output_readable.json.
 * Uses nearest weather station and a point cache so we don't recompute for the same point.
 */
import fs from "fs";

const ROUTES_INPUT = "./data/routes/routes_output_readable.json";
const ROUTES_OUTPUT = "./data/routes/routes_with_weather.json";
const POINT_CACHE_PATH = "./cache/point_weather_cache.json";
const CACHE_KEY_DECIMALS = 3; // round lat/lng to 3 decimals (~100m) for cache key

let stations = [];
let stationWeekly = {};
let pointCache = {};

function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function nearestStation(lat, lng) {
  let best = null;
  let bestD = Infinity;
  for (const s of stations) {
    const d = haversineKm(lat, lng, s.latitude, s.longitude);
    if (d < bestD) {
      bestD = d;
      best = s;
    }
  }
  return best;
}

function cacheKey(lat, lng) {
  return `${Number(lat).toFixed(CACHE_KEY_DECIMALS)}_${Number(lng).toFixed(CACHE_KEY_DECIMALS)}`;
}

function getWeatherForPoint(lat, lng) {
  const key = cacheKey(lat, lng);
  if (pointCache[key]) return pointCache[key];

  const station = nearestStation(lat, lng);
  if (!station) return null;

  const weekly = stationWeekly[station.station_id];
  if (!weekly) return null;

  const weeklyArray = [];
  for (let w = 1; w <= 52; w++) {
    const data = weekly[w] || {};
    weeklyArray.push({
      week: w,
      temp_mean_avg: data.temp_mean ?? null,
      temp_max_avg: data.temp_max ?? null,
      temp_min_avg: data.temp_min ?? null,
      snow_depth_avg: data.snow_depth ?? null,
      prcp_total_avg: data.prcp_total ?? null,
      wind_speed_mean_avg: data.wind_speed_mean ?? null,
      wind_speed_max_avg: data.wind_speed_max ?? null,
      wind_gust_max_avg: data.wind_gust_max ?? null,
      visibility_avg: data.visibility ?? null,
    });
  }

  const entry = {
    nearest_station_id: station.station_id,
    nearest_station_name: station.station_name || null,
    weekly_weather: weeklyArray,
  };
  pointCache[key] = entry;
  return entry;
}

function write(out, chunk) {
  return new Promise((resolve) => {
    const ok = out.write(chunk);
    if (ok) resolve();
    else out.once("drain", resolve);
  });
}

async function main() {
  console.log("Loading weather stations...");
  stations = JSON.parse(fs.readFileSync("./data/weather/weather_stations.json", "utf8"));
  console.log("Loading station weekly weather...");
  stationWeekly = JSON.parse(fs.readFileSync("./data/weather/station_weekly_weather.json", "utf8"));

  if (fs.existsSync(POINT_CACHE_PATH)) {
    console.log("Loading point cache...");
    pointCache = JSON.parse(fs.readFileSync(POINT_CACHE_PATH, "utf8"));
    console.log("  Cached points:", Object.keys(pointCache).length);
  }

  console.log("Loading routes...");
  const data = JSON.parse(fs.readFileSync(ROUTES_INPUT, "utf8"));
  console.log("  Warehouses:", data.length);

  let pointsProcessed = 0;
  let cacheHits = 0;

  for (const warehouse of data) {
    for (const route of warehouse.routes || []) {
      const path = route.path_coordinates || [];
      for (const pt of path) {
        const key = cacheKey(pt.latitude, pt.longitude);
        if (pointCache[key]) cacheHits++;
        const weather = getWeatherForPoint(pt.latitude, pt.longitude);
        if (weather) {
          pt.weekly_weather = weather.weekly_weather;
          pt.nearest_station_id = weather.nearest_station_id;
          pt.nearest_station_name = weather.nearest_station_name;
        }
        pointsProcessed++;
      }
    }
  }

  console.log("Points processed:", pointsProcessed, "Cache hits:", cacheHits);
  console.log("Writing", ROUTES_OUTPUT, "...");

  const out = fs.createWriteStream(ROUTES_OUTPUT, { encoding: "utf8" });
  await write(out, "[\n");

  for (let i = 0; i < data.length; i++) {
    const chunk = JSON.stringify(data[i], null, 2);
    await write(out, i === 0 ? chunk : ",\n" + chunk);
    if ((i + 1) % 100 === 0) console.log("  Written", i + 1, "warehouses");
  }

  await write(out, "\n]\n");
  out.end();
  await new Promise((resolve, reject) => {
    out.on("finish", resolve);
    out.on("error", reject);
  });

  console.log("Saving point cache...");
  fs.writeFileSync(POINT_CACHE_PATH, JSON.stringify(pointCache, null, 0), "utf8");
  console.log("  Cache size:", Object.keys(pointCache).length, "points");
  console.log("Done. Output:", ROUTES_OUTPUT);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
