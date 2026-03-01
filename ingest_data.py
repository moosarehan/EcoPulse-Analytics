import requests
import json
import time
import pandas as pd
import os
from datetime import datetime
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("OPENAQ_API_KEY")
BASE_URL = "https://api.openaq.org/v3"
DATA_DIR = "air_quality_data"

headers = {
    "X-API-Key": API_KEY
}

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def fetch_hourly_data(sensor_id, date_from, date_to):
    url = f"{BASE_URL}/sensors/{sensor_id}/hours"
    params = {
        "date_from": date_from,
        "date_to": date_to,
        "limit": 1000
    }
    
    retries = 3
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 200:
                results = response.json().get("results", [])
                time.sleep(0.4) # Safe sleep between requests
                return results
            elif response.status_code == 429:
                print(f"Rate limited (429). Sleeping 60s...")
                time.sleep(60)
            elif response.status_code in [408, 500]:
                print(f"Error {response.status_code}. Retrying {attempt+1}/{retries}...")
                if attempt == retries - 1:
                    print(f"  Final failure response for sensor {sensor_id}: {response.text}")
                time.sleep(10)
            else:
                print(f"Error {response.status_code} for sensor {sensor_id}: {response.text}")
                break
        except Exception as e:
            print(f"Exception for sensor {sensor_id}: {e}")
            time.sleep(10)
            
    return None # Indicate failure after retries

def main():
    with open("found_locations.json", "r") as f:
        locations = json.load(f)
    
    months = [
        ("2025-01-01T00:00:00Z", "2025-01-31T23:59:59Z"),
        ("2025-02-01T00:00:00Z", "2025-02-28T23:59:59Z"),
        ("2025-03-01T00:00:00Z", "2025-03-31T23:59:59Z"),
        ("2025-04-01T00:00:00Z", "2025-04-30T23:59:59Z"),
        ("2025-05-01T00:00:00Z", "2025-05-31T23:59:59Z"),
        ("2025-06-01T00:00:00Z", "2025-06-30T23:59:59Z"),
        ("2025-07-01T00:00:00Z", "2025-07-31T23:59:59Z"),
        ("2025-08-01T00:00:00Z", "2025-08-31T23:59:59Z"),
        ("2025-09-01T00:00:00Z", "2025-09-30T23:59:59Z"),
        ("2025-10-01T00:00:00Z", "2025-10-31T23:59:59Z"),
        ("2025-11-01T00:00:00Z", "2025-11-30T23:59:59Z"),
        ("2025-12-01T00:00:00Z", "2025-12-31T23:59:59Z"),
    ]

    progress_file = "ingestion_progress.json"
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = json.load(f)
    else:
        progress = {"last_location_idx": 0}

    for i in range(len(locations)):
        loc = locations[i]
        loc_id = loc["id"]
        loc_name = loc["name"]
        
        output_file = f"{DATA_DIR}/loc_{loc_id}.parquet"
        if os.path.exists(output_file):
            continue

        print(f"Processing location {i+1}/100: {loc_name} (ID: {loc_id})")
        
        sensors = loc.get("sensors", [])
        sensor_map = {}
        for s in sensors:
            name = s["parameter"]["name"]
            if name in ["pm25", "pm10", "no2", "o3", "temperature", "humidity", "relativehumidity"]:
                if name not in sensor_map:
                    sensor_map[name] = []
                sensor_map[name].append(s["id"])
        
        all_data = []
        for param, sensor_ids in sensor_map.items():
            print(f"  Fetching {param} (tried {len(sensor_ids)} sensors)...")
            param_success = False
            for sensor_id in sensor_ids:
                if param_success:
                    break
                    
                sensor_data = []
                sensor_failed = False
                for m_start, m_end in months:
                    results = fetch_hourly_data(sensor_id, m_start, m_end)
                    if results is None:
                        print(f"    Sensor {sensor_id} failed completely. Trying fallback if available...")
                        sensor_failed = True
                        break
                    
                    for res in results:
                        sensor_data.append({
                            "location_id": loc_id,
                            "location_name": loc_name,
                            "parameter": param,
                            "value": res["value"],
                            "timestamp": res["period"]["datetimeFrom"]["utc"],
                            "unit": res["parameter"]["units"]
                        })
                
                if not sensor_failed:
                    all_data.extend(sensor_data)
                    param_success = True
            
            if not param_success:
                print(f"  FAILED to fetch {param} after trying all available sensors.")
        
        if all_data:
            df = pd.DataFrame(all_data)
            output_file = f"{DATA_DIR}/loc_{loc_id}.parquet"
            df.to_parquet(output_file, index=False)
            print(f"  Saved {len(all_data)} records to {output_file}")
        
        progress["last_location_idx"] = i + 1
        with open(progress_file, "w") as f:
            json.dump(progress, f)
        
        time.sleep(1) # Extra gap between locations

if __name__ == "__main__":
    main()
