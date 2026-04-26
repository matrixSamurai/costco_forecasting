/**
 * For each path coordinate in routes_with_weather.json, add distance and time
 * to the next successive point using Google Routes API (DRIVE).
 * Output: routes_with_weather_and_substation_time.json
 * Uses a segment cache so same segment (A→B) is not recomputed.
 */
import fs from "fs";
import dotenv from "dotenv";
import pLimit from "p-limit";

dotenv.config();

const API_KEY = process.env.ROUTES_API_KEY;
const ROUTES_INPUT = "./data/routes/routes_with_weather.json";
const ROUTES_OUTPUT = "./data/routes/routes_with_weather_and_substation_time.json";
const SEGMENT_CACHE_PATH = "./cache/segment_time_cache.json";
const DECIMALS = 3; // round lat/lng for cache key (~100m) - more cache hits
const CONCURRENCY = 5;

const limit = pLimit(CONCURRENCY);
let segmentCache = {};

function segmentKey(lat1, lng1, lat2, lng2) {
  return [
    Number(lat1).toFixed(DECIMALS),
    Number(lng1).toFixed(DECIMALS),
    Number(lat2).toFixed(DECIMALS),
    Number(lng2).toFixed(DECIMALS),
  ].join("_");
}

async function getSegmentDistanceTime(origin, destination) {
  const key = segmentKey(origin.latitude, origin.longitude, destination.latitude, destination.longitude);
  if (segmentCache[key]) return segmentCache[key];

  const result = await limit(async () => {
    try {
      const response = await fetch(
        "https://routes.googleapis.com/directions/v2:computeRoutes",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": API_KEY,
            "X-Goog-FieldMask": "routes.distanceMeters,routes.duration",
          },
          body: JSON.stringify({
            origin: { location: { latLng: { latitude: origin.latitude, longitude: origin.longitude } } },
            destination: { location: { latLng: { latitude: destination.latitude, longitude: destination.longitude } } },
            travelMode: "DRIVE",
            routingPreference: "TRAFFIC_UNAWARE",
          }),
        }
      );
      const data = await response.json();
      if (!data.routes || data.routes.length === 0) return null;
      const r = data.routes[0];
      const durationStr = r.duration || "0s";
      const durationSec = parseInt(String(durationStr).replace("s", ""), 10) || 0;
      return {
        distance_to_next_km: (r.distanceMeters || 0) / 1000,
        duration_to_next_min: durationSec / 60,
      };
    } catch (err) {
      console.error("Segment API error:", err.message);
      return null;
    }
  });

  if (result) segmentCache[key] = result;
  return result;
}

function write(out, chunk) {
  return new Promise((resolve) => {
    const ok = out.write(chunk);
    if (ok) resolve();
    else out.once("drain", resolve);
  });
}

async function main() {
  if (!API_KEY) {
    console.error("Missing ROUTES_API_KEY in .env");
    process.exit(1);
  }

  if (fs.existsSync(SEGMENT_CACHE_PATH)) {
    console.log("Loading segment cache...");
    segmentCache = JSON.parse(fs.readFileSync(SEGMENT_CACHE_PATH, "utf8"));
    console.log("  Cached segments:", Object.keys(segmentCache).length);
  }

  console.log("Loading", ROUTES_INPUT, "...");
  const data = JSON.parse(fs.readFileSync(ROUTES_INPUT, "utf8"));

  let totalSegments = 0;
  let fetched = 0;

  for (const warehouse of data) {
    for (const route of warehouse.routes || []) {
      const path = route.path_coordinates || [];
      for (let i = 0; i < path.length - 1; i++) {
        totalSegments++;
        const from = path[i];
        const to = path[i + 1];
        const key = segmentKey(from.latitude, from.longitude, to.latitude, to.longitude);
        if (!segmentCache[key]) fetched++;
      }
    }
  }

  console.log("Total segments (point → next):", totalSegments);
  console.log("Segments to fetch from API:", fetched);
  console.log("");

  let done = 0;
  for (const warehouse of data) {
    for (const route of warehouse.routes || []) {
      const path = route.path_coordinates || [];
      for (let i = 0; i < path.length - 1; i++) {
        const from = path[i];
        const to = path[i + 1];
        const seg = await getSegmentDistanceTime(from, to);
        if (seg) {
          path[i].distance_to_next_km = seg.distance_to_next_km;
          path[i].duration_to_next_min = seg.duration_to_next_min;
        } else {
          path[i].distance_to_next_km = null;
          path[i].duration_to_next_min = null;
        }
        done++;
        if (done % 500 === 0) console.log("  Processed", done, "/", totalSegments, "segments");
        if (done % 5000 === 0 && Object.keys(segmentCache).length > 0) {
          fs.writeFileSync(SEGMENT_CACHE_PATH, JSON.stringify(segmentCache, null, 0), "utf8");
          console.log("  (segment cache saved,", Object.keys(segmentCache).length, "entries)");
        }
      }
      // Last point in path: no "next"
      if (path.length > 0) {
        path[path.length - 1].distance_to_next_km = null;
        path[path.length - 1].duration_to_next_min = null;
      }
    }
  }

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

  console.log("Saving segment cache...");
  fs.writeFileSync(SEGMENT_CACHE_PATH, JSON.stringify(segmentCache, null, 0), "utf8");
  console.log("Done. Output:", ROUTES_OUTPUT);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
