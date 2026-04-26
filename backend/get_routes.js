import fs from "fs";
import dotenv from "dotenv";
import pLimit from "p-limit";

dotenv.config();

const API_KEY = process.env.ROUTES_API_KEY;

// concurrency limit to avoid quota errors
const limit = pLimit(5);

// depot location
const DEPOT = {
  latitude: 37.7397,
  longitude: -121.4252
};

// load warehouse list
const warehouses = JSON.parse(
  fs.readFileSync("./data/source/costco_warehouses_full.json", "utf8")
);

// output results
const results = [];

/* ----------------------------------------
   ROUTE CALL FUNCTION
---------------------------------------- */
async function getRoute(destination){

  try{

    const response = await fetch(
      "https://routes.googleapis.com/directions/v2:computeRoutes",
      {
        method:"POST",
        headers:{
          "Content-Type":"application/json",
          "X-Goog-Api-Key": API_KEY,
          "X-Goog-FieldMask":
            "routes.distanceMeters,routes.duration,routes.polyline"
        },
        body: JSON.stringify({
          origin:{ location:{ latLng: DEPOT } },
          destination:{ location:{ latLng: destination } },
          travelMode:"DRIVE",
          routingPreference:"TRAFFIC_AWARE",
          computeAlternativeRoutes:true
        })
      }
    );

    const data = await response.json();

    if(!data.routes){
      console.log("No route found");
      return null;
    }

    return data.routes.map(r=>({
      distance_km: r.distanceMeters/1000,
      duration_min: parseInt(r.duration)/60,
      polyline: r.polyline?.encodedPolyline || null
    }));

  }catch(err){
    console.error("Route error:",err);
    return null;
  }
}

/* ----------------------------------------
   MAIN LOOP
---------------------------------------- */

async function run(){

  console.log(`Starting routes for ${warehouses.length} warehouses`);

  const jobs = warehouses.map(w => limit(async()=>{

    console.log("Routing →", w.name);

    const routes = await getRoute({
      latitude:w.lat,
      longitude:w.lng
    });

    if(routes){
      results.push({
        warehouse_id: w.id,
        warehouse_name: w.name,
        routes
      });
    }

  }));

  await Promise.all(jobs);

  fs.writeFileSync(
    "data/routes/routes_output.json",
    JSON.stringify(results,null,2)
  );

  console.log("Done. Saved data/routes/routes_output.json");
}

run();
