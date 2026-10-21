# 3D SCANNER companion

Camera plan: ESP32-CAM connected with the Arduino Uno turntable. Phone-as-webcam is reserved for later. Neither camera is active in this release. The companion never pretends a depth scan succeeded.

## Start

Install Python dependencies using `python -m pip install -r requirements.txt`, then run `python bridge.py`. The service binds only to 127.0.0.1:8787 and prints a temporary pairing token. Enter that token in Device setup. Nothing opens or moves on startup.

## Optional Arduino

Upload the included turntable firmware in the Arduino IDE (retain a copy of any old sketch first). Wire ULN2003 IN1–IN4 to Uno D8–D11, with the appropriate power supply and common ground. Connect the Uno by USB, pair the companion, select its port and press Connect Arduino. Connecting checks the firmware handshake and does not move the motor. Camera capture and calibration are intentionally unavailable until the ESP32-CAM adapter is integrated.

## Real image-to-3D

The AI path runs the official local TripoSR engine, not the embossed relief algorithm.

1. Install https://github.com/VAST-AI-Research/TripoSR with its documented Python/PyTorch/CUDA dependencies. The upstream project documents its hardware requirements; higher mesh extraction resolution can require more memory.
2. Set the local environment variable `TRIPOSR_ROOT` to the installed repository folder. Set `TRIPOSR_PYTHON` to that environment's Python executable. If omitted, the bridge uses the Python interpreter running it.
3. Start the companion and pair it from Device setup. Choose Full 3D object in Workspace and upload an image.
4. Generate. The bridge invokes run.py with GLB output and the selected 256/384 marching-cubes extraction resolution. The first invocation may download pretrained weights. Processing stays on the user's machine, apart from those engine downloads.
5. The browser only displays the resulting self-contained GLB after real completion. Engine errors remain errors; there is no substituted relief or fake progress.

Generated files and engine logs are kept under companion/jobs. TripoSR inference has not been run in the authoring environment because the engine and weights are not installed. The integration requires local engine setup and a successful inference test on the scanner computer. A single image cannot recover unseen surfaces exactly.

## Local website fallback

Extract the website production ZIP into a `web` folder next to bridge.py, then open http://127.0.0.1:8787/workspace.html if the browser blocks hosted-page access to localhost. All API requests require the pairing token and permitted origin.

The published private website uses its platform's real sign-in flow. Local browser projects remain on the current device; account login does not synchronise them.
