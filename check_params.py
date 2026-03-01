import requests
import json

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

def get_parameters():
    response = requests.get(f"{BASE_URL}/parameters", headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

params = get_parameters()
if params:
    with open("parameters.json", "w") as f:
        json.dump(params, f, indent=4)
    print("Parameters saved to parameters.json")
