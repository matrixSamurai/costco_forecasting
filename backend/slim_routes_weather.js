/**
 * 1. Creates point_weekly_weather.json - single copy of weekly weather per unique point (no duplication).
 * 2. Creates slim routes_with_weather.json - each path coordinate has only latitude, longitude, weather_key.
 *    Look up full weather in point_weekly_weather.json by weather_key.
 */
import fs from "fs";

const POINT_CACHE_PATH = "./cache/point_weather_cache.json";
const POINT_WEEKLY_WEATHER_PATH = "./data/weather/point_weekly_weather.json";
const ROUTES_INPUT = "./data/routes/routes_output_readable.json";
const ROUTES_OUTPUT = "./data/routes/routes_with_weather.json";
const CACHE_KEY_DECIMALS = 3;

function cacheKey(lat, lng) {
  return `${Number(lat).toFixed(CACHE_KEY_DECIMALS)}_${Number(lng).toFixed(CACHE_KEY_DECIMALS)}`;
}

function write(out, chunk) {
  return new Promise((resolve) => {
    const ok = out.write(chunk);
    if (ok) resolve();
    else out.once("drain", resolve);
  });
}

async function main() {
  // 1. Create point_weekly_weather.json from cache (single source of truth for weather)
  console.log("Creating", POINT_WEEKLY_WEATHER_PATH, "(weekly weather by point, no duplication)...");
  if (!fs.existsSync(POINT_CACHE_PATH)) {
    console.error("Missing", POINT_CACHE_PATH, "- run add_weather_to_routes.js first.");
    process.exit(1);
  }
  const cache = JSON.parse(fs.readFileSync(POINT_CACHE_PATH, "utf8"));
  fs.writeFileSync(POINT_WEEKLY_WEATHER_PATH, JSON.stringify(cache, null, 0), "utf8");
  console.log("  Unique points with weather:", Object.keys(cache).length);

  // 2. Create slim routes file: path coordinates have only lat, lng, weather_key
  console.log("Creating slim", ROUTES_OUTPUT, "(only weather_key per point)...");
  const data = JSON.parse(fs.readFileSync(ROUTES_INPUT, "utf8"));

  for (const warehouse of data) {
    for (const route of warehouse.routes || []) {
      const path = route.path_coordinates || [];
      for (const pt of path) {
        pt.weather_key = cacheKey(pt.latitude, pt.longitude);
        // Remove any inline weather fields if present (from a previous fat run)
        delete pt.weekly_weather;
        delete pt.nearest_station_id;
        delete pt.nearest_station_name;
      }
    }
  }

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

  const outSize = fs.statSync(ROUTES_OUTPUT).size;
  console.log("Done.");
  console.log("  point_weekly_weather.json: lookup by weather_key for 52-week averages");
  console.log("  routes_with_weather.json: slim, no duplicated weather, size ~", (outSize / 1024 / 1024).toFixed(1), "MB");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
