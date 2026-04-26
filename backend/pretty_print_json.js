/**
 * Pretty-print minified JSON files with newlines and indentation (human readable).
 * Uses streaming write for large files to avoid memory limits.
 */
import fs from "fs";

function indent(str, spaces = 2) {
  const prefix = " ".repeat(spaces);
  return str
    .split("\n")
    .map((line) => prefix + line)
    .join("\n");
}

function write(out, chunk) {
  return new Promise((resolve) => {
    const ok = out.write(chunk);
    if (ok) resolve();
    else out.once("drain", resolve);
  });
}

async function prettyPrintObjectToStream(obj, outPath) {
  const out = fs.createWriteStream(outPath, { encoding: "utf8" });
  const keys = Object.keys(obj);
  await write(out, "{\n");

  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    const valueStr = JSON.stringify(obj[k], null, 2);
    const indented = indent(valueStr);
    const line =
      (i === 0 ? "" : ",\n") + "  " + JSON.stringify(k) + ": " + indented;
    await write(out, line);
    if ((i + 1) % 500 === 0) console.log("    ", i + 1, "/", keys.length);
  }

  await write(out, "\n}\n");
  out.end();
  await new Promise((resolve, reject) => {
    out.on("finish", resolve);
    out.on("error", reject);
  });
}

async function main() {
  const files = [
    { path: "data/weather/weather_stations.json", stream: false },
    { path: "data/weather/station_weekly_weather.json", stream: true },
    { path: "cache/point_weather_cache.json", stream: true },
  ];

  for (const { path, stream } of files) {
    if (!fs.existsSync(path)) {
      console.log("Skip (missing):", path);
      continue;
    }
    console.log("Pretty-printing", path, "...");
    const obj = JSON.parse(fs.readFileSync(path, "utf8"));

    if (stream) {
      const tmpPath = path + ".tmp";
      await prettyPrintObjectToStream(obj, tmpPath);
      fs.renameSync(tmpPath, path);
    } else {
      fs.writeFileSync(path, JSON.stringify(obj, null, 2), "utf8");
    }
    console.log("  Done.");
  }

  // point_weekly_weather.json = same content as point_weather_cache (copy after pretty-print)
  if (fs.existsSync("cache/point_weather_cache.json")) {
    console.log("Copying cache/point_weather_cache.json → data/weather/point_weekly_weather.json ...");
    fs.copyFileSync("cache/point_weather_cache.json", "data/weather/point_weekly_weather.json");
    console.log("  Done.");
  }

  console.log("All done.");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
