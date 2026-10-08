import requests
import time
import json
import os

ENGINES = ["trellis", "hunyuan3d", "triposr", "ai_depth"]
URL = "http://localhost:8000/api/reconstruct"
FILE_PATH = r"C:\Users\Jolly InfoTech\Downloads\Picture1.png"

results = {}

for engine in ENGINES:
    print(f"Testing engine: {engine}")
    try:
        with open(FILE_PATH, 'rb') as f:
            files = {'file': ('Picture1.png', f, 'image/png')}
            params = {
                'engine': engine,
                'object_type': 'organic',
                'texture': 'true',
                'quality': 'high',
                'refine': 'true',
                'printable': 'false'
            }
            resp = requests.post(URL, params=params, files=files)
            print(resp.status_code)
            try:
                data = resp.json()
                print(json.dumps(data, indent=2))
                results[engine] = data
            except:
                print(resp.text)
                results[engine] = {"error": resp.text}
    except Exception as e:
        print(f"Exception for {engine}: {e}")

with open("test_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Finished testing all engines.")
