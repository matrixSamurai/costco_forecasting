/**
 * One-time: stream weather_data_2021_2025.csv to build
 * 1) weather_stations.json - unique stations with lat/lng
 * 2) station_weekly_weather.json - per station, 52 weeks of averages (temp max/min, snow depth, etc.)
 */
import fs from "fs";
import readline from "readline";

const CSV_PATH = "./data/source/weather_data_2021_2025.csv";
const STATIONS_PATH = "./data/weather/weather_stations.json";
const WEEKLY_PATH = "./data/weather/station_weekly_weather.json";

const WEEK_FIELDS = [
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

function dayOfYear(year, month, day) {
  const days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  if (year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0)) days[2] = 29;
  let d = day;
  for (let m = 1; m < month; m++) d += days[m];
  return d;
}

function weekOfYear(year, month, day) {
  const d = dayOfYear(year, month, day);
  return Math.min(52, 1 + Math.floor((d - 1) / 7));
}

async function build() {
  const stations = new Map(); // station_id -> { station_id, latitude, longitude, station_name }
  const weekly = new Map();   // station_id -> { week -> { sum, count } per field }

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
    const lat = parseFloat(get("latitude"));
    const lng = parseFloat(get("longitude"));
    if (!stationId || isNaN(lat) || isNaN(lng)) continue;

    const year = parseInt(get("year"), 10);
    const month = parseInt(get("month"), 10);
    const day = parseInt(get("day"), 10);
    if (isNaN(year) || isNaN(month) || isNaN(day)) continue;

    const week = weekOfYear(year, month, day);

    if (!stations.has(stationId)) {
      stations.set(stationId, {
        station_id: stationId,
        latitude: lat,
        longitude: lng,
        station_name: get("station_name") || "",
      });
    }

    if (!weekly.has(stationId)) weekly.set(stationId, {});
    const st = weekly.get(stationId);
    if (!st[week]) {
      st[week] = {};
      WEEK_FIELDS.forEach((f) => {
        st[week][f] = { sum: 0, count: 0 };
      });
    }

    WEEK_FIELDS.forEach((field) => {
      const v = get(field);
      const num = v === "" || v === null ? NaN : parseFloat(v);
      if (!isNaN(num)) {
        st[week][field].sum += num;
        st[week][field].count += 1;
      }
    });
  }

  const stationList = Array.from(stations.values());
  const weeklyAgg = {};
  for (const [sid, weeks] of weekly) {
    weeklyAgg[sid] = {};
    for (let w = 1; w <= 52; w++) {
      const data = weeks[w];
      if (!data) {
        weeklyAgg[sid][w] = null;
        continue;
      }
      const out = {};
      WEEK_FIELDS.forEach((f) => {
        const { sum, count } = data[f];
        out[f] = count > 0 ? sum / count : null;
      });
      weeklyAgg[sid][w] = out;
    }
  }

  fs.writeFileSync(STATIONS_PATH, JSON.stringify(stationList, null, 0), "utf8");
  fs.writeFileSync(WEEKLY_PATH, JSON.stringify(weeklyAgg, null, 0), "utf8");
  console.log("Stations:", stationList.length);
  console.log("Written:", STATIONS_PATH, WEEKLY_PATH);
}

build().catch((e) => {
  console.error(e);
  process.exit(1);
});
