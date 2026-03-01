import requests
import json
import time

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("OPENAQ_API_KEY")
BASE_URL = "https://api.openaq.org/v3"

headers = {
    "X-API-Key": API_KEY
}

# Desired parameters
# PM2.5 (2), PM10 (1), NO2 (7 or 5), O3 (10 or 3), Temp (100), Humidity (134 or 98)

def find_locations(limit=100):
    locations = []
    page = 1
    found_count = 0
    
    while found_count < limit:
        print(f"Fetching locations page {page}... (Found {found_count}/{limit})")
        params = {
            "limit": 100,
            "page": page,
            "parameters_id": [2] # Filter by PM2.5 to narrow down
        }
        
        try:
            response = requests.get(f"{BASE_URL}/locations", headers=headers, params=params, timeout=30)
            if response.status_code != 200:
                print(f"Error fetching locations: {response.status_code}")
                break
                
            data = response.json()
            results = data.get("results", [])
            if not results:
                print("No more results.")
                break
                
            for loc in results:
                sensors = loc.get("sensors", [])
                sensor_params = {s['parameter']['name'] for s in sensors}
                
                # Check for all 6
                has_pm25 = "pm25" in sensor_params
                has_pm10 = "pm10" in sensor_params
                has_no2 = "no2" in sensor_params
                has_o3 = "o3" in sensor_params
                has_temp = "temperature" in sensor_params
                has_hum = "humidity" in sensor_params or "relativehumidity" in sensor_params
                
                if all([has_pm25, has_pm10, has_no2, has_o3, has_temp, has_hum]):
                    locations.append(loc)
                    found_count += 1
                    if found_count >= limit:
                        break
            
            page += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"Request failed: {e}")
            time.sleep(5)
            continue
            
    return locations

locations = find_locations(100)
with open("found_locations.json", "w") as f:
    json.dump(locations, f, indent=4)

print(f"Finished. Found {len(locations)} locations.")
