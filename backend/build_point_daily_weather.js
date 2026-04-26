/**
 * Build point_daily_weather.json from point_weather_cache (point -> station)
 * and station_daily_weather.jsonl (one line per station: station_id + daily_weather).
 * Each point gets day-wise weather for all 5 years from its nearest station.
 * Streams input/output to avoid loading 745MB into memory.
 */
import fs from "fs";
import readline from "readline";

const CACHE_PATH = "./cache/point_weather_cache.json";
const STATION_DAILY_JSONL = "./data/weather/station_daily_weather.jsonl";
const OUTPUT_PATH = "./data/weather/point_daily_weather.json";

function write(out, chunk) {
  return new Promise((resolve) => {
    const ok = out.write(chunk);
    if (ok) resolve();
    else out.once("drain", resolve);
  });
}

async function main() {
  console.log("Loading point cache (point -> nearest station)...");
  const cache = JSON.parse(fs.readFileSync(CACHE_PATH, "utf8"));
  const pointKeys = Object.keys(cache);

  // station_id -> list of point_keys that use this station
  const stationToPoints = new Map();
  for (const key of pointKeys) {
    const sid = cache[key]?.nearest_station_id;
    if (sid) {
      if (!stationToPoints.has(sid)) stationToPoints.set(sid, []);
      stationToPoints.get(sid).push(key);
    }
  }
  console.log("  Points:", pointKeys.length, "Stations referenced:", stationToPoints.size);

  const out = fs.createWriteStream(OUTPUT_PATH, { encoding: "utf8" });
  await write(out, "{\n");

  const fileStream = fs.createReadStream(STATION_DAILY_JSONL, { encoding: "utf8" });
  const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

  let first = true;
  let lineCount = 0;

  for await (const line of rl) {
    if (!line.trim()) continue;
    let row;
    try {
      row = JSON.parse(line);
    } catch (_) {
      continue;
    }
    const { station_id: stationId, daily_weather: dailyWeather } = row;
    const pointKeysForStation = stationToPoints.get(stationId);
    if (!pointKeysForStation || !dailyWeather) continue;

    const stationName = cache[pointKeysForStation[0]]?.nearest_station_name ?? null;

    for (const key of pointKeysForStation) {
      const value = {
        nearest_station_id: stationId,
        nearest_station_name: stationName,
        daily_weather: dailyWeather,
      };
      const valueStr = JSON.stringify(value, null, 2);
      const indented = valueStr.split("\n").map((line) => "  " + line).join("\n");
      const chunk = (first ? "" : ",\n") + "  " + JSON.stringify(key) + ": " + indented;
      await write(out, chunk);
      first = false;
    }

    lineCount++;
    if (lineCount % 200 === 0) console.log("  Processed", lineCount, "stations");
  }

  await write(out, "\n}\n");
  out.end();
  await new Promise((resolve, reject) => {
    out.on("finish", resolve);
    out.on("error", reject);
  });

  const size = fs.statSync(OUTPUT_PATH).size;
  console.log("Done.", OUTPUT_PATH, "~", (size / 1024 / 1024).toFixed(1), "MB");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
