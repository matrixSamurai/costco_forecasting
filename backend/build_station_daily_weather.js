/**
 * Build station_daily_weather.json from weather_data_2021_2025.csv.
 * One record per station per day (2021–2025). Used to build point_daily_weather.json.
 */
import fs from "fs";
import readline from "readline";

const CSV_PATH = "./data/source/weather_data_2021_2025.csv";
const OUTPUT_PATH = "./data/weather/station_daily_weather.jsonl";

const DAILY_FIELDS = [
  "temp_mean",
  "temp_max",
  "temp_min",
  "prcp_total",
  "snow_depth",
  "wind_speed_mean",
  "wind_speed_max",
  "wind_gust_max",
  "visibility",
];

async function build() {
  const daily = new Map(); // station_id -> array of { date, ...fields }

  const fileStream = fs.createReadStream(CSV_PATH, { encoding: "utf8" });
  const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

  let header = null;
  let lineNo = 0;

  for await (const line of rl) {
    lineNo++;
    if (lineNo === 1) {
      header = line.split(",").map((h) => h.trim());
      continue;
    }
    const cols = line.split(",");
    if (cols.length < 25) continue;

    const get = (name) => {
      const i = header.indexOf(name);
      return i >= 0 ? cols[i]?.trim() : null;
    };

    const stationId = get("station_id");
    const date = get("date");
    if (!stationId || !date) continue;

    const record = { date };
    DAILY_FIELDS.forEach((field) => {
      const v = get(field);
      const num = v === "" || v === null ? NaN : parseFloat(v);
      record[field] = isNaN(num) ? null : num;
    });

    if (!daily.has(stationId)) daily.set(stationId, []);
    daily.get(stationId).push(record);

    if (lineNo % 500000 === 0) console.log("  Rows:", lineNo);
  }

  console.log("Stations with daily data:", daily.size);
  console.log("Writing", OUTPUT_PATH, "...");

  const out = fs.createWriteStream(OUTPUT_PATH, { encoding: "utf8" });
  for (const [sid, arr] of daily) {
    out.write(JSON.stringify({ station_id: sid, daily_weather: arr }) + "\n");
  }
  out.end();
  await new Promise((resolve, reject) => {
    out.on("finish", resolve);
    out.on("error", reject);
  });

  console.log("Done.", OUTPUT_PATH);
}

build().catch((e) => {
  console.error(e);
  process.exit(1);
});
