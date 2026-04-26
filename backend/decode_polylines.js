import fs from "fs";
import pkg from "@googlemaps/polyline-codec";
const { decode } = pkg;

const INPUT_FILE = "./data/routes/routes_output.json";
const OUTPUT_FILE = "./data/routes/routes_output_readable.json";

// Distance between successive kept points (km)
const POINT_SPACING_KM = 20;

// Google Routes API uses precision 5 for lat/lng (5 decimal places)
const POLYLINE_PRECISION = 5;

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

/** Keep only points ~POINT_SPACING_KM km apart along the path (always first and last). */
function thinPathToSpacing(points, spacingKm) {
  if (points.length <= 2) return points;
  const kept = [points[0]];
  let accumulated = 0;
  for (let i = 1; i < points.length; i++) {
    const [lat1, lng1] = points[i - 1];
    const [lat2, lng2] = points[i];
    accumulated += haversineKm(lat1, lng1, lat2, lng2);
    if (accumulated >= spacingKm) {
      kept.push(points[i]);
      accumulated = 0;
    }
  }
  const last = points[points.length - 1];
  if (kept[kept.length - 1][0] !== last[0] || kept[kept.length - 1][1] !== last[1]) {
    kept.push(last);
  }
  return kept;
}

console.log("Path spacing:", POINT_SPACING_KM, "km between successive points");
console.log("Reading", INPUT_FILE, "...");
const data = JSON.parse(fs.readFileSync(INPUT_FILE, "utf8"));

let totalDecoded = 0;
let totalSkipped = 0;

function write(out, chunk) {
  return new Promise((resolve) => {
    const ok = out.write(chunk);
    if (ok) resolve();
    else out.once("drain", resolve);
  });
}

const out = fs.createWriteStream(OUTPUT_FILE, { encoding: "utf8" });
await write(out, "[\n");

for (let i = 0; i < data.length; i++) {
  const warehouse = data[i];
  const entry = {
    warehouse_id: warehouse.warehouse_id,
    warehouse_name: warehouse.warehouse_name,
    routes: warehouse.routes.map((route) => {
      const decoded = route.polyline
        ? decode(route.polyline, POLYLINE_PRECISION)
        : [];

      const thinned = thinPathToSpacing(decoded, POINT_SPACING_KM);
      const path_coordinates = thinned.map(([lat, lng]) => ({
        latitude: lat,
        longitude: lng,
      }));

      if (route.polyline && decoded.length > 0) totalDecoded++;
      if (route.polyline && decoded.length === 0) totalSkipped++;

      return {
        distance_km: route.distance_km,
        duration_min: route.duration_min,
        polyline: route.polyline,
        path_coordinates,
      };
    }),
  };

  const chunk = JSON.stringify(entry, null, 2);
  await write(out, i === 0 ? chunk : ",\n" + chunk);

  if ((i + 1) % 100 === 0) console.log("  Processed", i + 1, "warehouses");
}

await write(out, "\n]\n");
out.end();

await new Promise((resolve, reject) => {
  out.on("finish", resolve);
  out.on("error", reject);
});

console.log("Done.");
console.log("  Decoded polylines:", totalDecoded);
console.log("  Empty/failed:", totalSkipped);
console.log("  Output:", OUTPUT_FILE);
