export function deviceMarkup(config,escape){return `
<div class="heading"><div><p class="eyebrow">YOUR SCANNER CONNECTIONS</p><h1>Connect. <em>Create.</em></h1><p class="intro">Arduino turntable control, a planned ESP32-CAM, and an optional local AI engine.</p></div><a class="btn" href="workspace.html">Open workspace ↗</a></div>
<div class="setup-grid"><section class="setup-card"><span class="eyebrow">01 / LOCAL COMPANION</span><h2>One local <em>connection.</em></h2>
<label class="field">Companion address<input id="bridge-url" type="text" value="${escape(config.url)}"></label>
<label class="field">Pairing token<input id="bridge-token" type="password" value="${escape(config.token)}" placeholder="Shown when bridge.py starts" autocomplete="off"></label>
<div class="project-actions"><button class="btn primary" id="pair">Connect companion ↗</button><button class="btn" id="disconnect">Disconnect</button></div>
<div class="status" id="status" role="status" aria-live="polite">No connection made yet.</div>
<div id="hardware-settings" hidden><label class="field">Arduino Uno port<select id="serial-port"></select></label><button id="connect-hardware" class="btn">Connect Arduino ↗</button><p class="note">Connecting the controller does not start the motor.</p></div>
<a class="btn full" href="./downloads/3d-scanner-companion.zip" download style="margin-top:24px">Download companion ↓</a></section>
<div><section class="setup-card"><span class="eyebrow">02 / CAMERA PLAN</span><h2>ESP32-CAM <em>first.</em></h2>
<div class="scan-card"><span class="planned">PLANNED / NOT ACTIVE</span><h3>ESP32-CAM + Arduino Uno</h3><p>The ESP32-CAM will capture views while the Uno controls the platter. Frame capture and calibration will be enabled after hardware integration.</p></div>
<div class="scan-card"><span class="planned">COMING LATER</span><h3>Phone camera as a webcam</h3><p>Reserved for a future capture mode. It is not used in this build yet.</p></div>
<p class="note">No camera opens automatically. Scan controls stay disabled until a working camera adapter is available.</p></section>
<section class="setup-card" id="ai" style="margin-top:22px"><span class="eyebrow">03 / FULL IMAGE-TO-3D</span><h2>A real object, <em>not a plaque.</em></h2>
<p class="note">AI mode uses TripoSR to reconstruct a mesh with sides and a back. It is separate from the embossed-relief tool.</p>
<ol><li>Install TripoSR and its dependencies using the official project instructions.</li><li>Set <code>TRIPOSR_ROOT</code> to its folder and <code>TRIPOSR_PYTHON</code> to that environment’s Python executable.</li><li>Start <code>python bridge.py</code>, pair the companion, then choose Full 3D object in Workspace.</li></ol>
<a class="btn" href="https://github.com/VAST-AI-Research/TripoSR" target="_blank" rel="noreferrer">TripoSR setup instructions ↗</a>
<p class="note">First use may download model weights. A single photo cannot reveal unseen geometry exactly. Local setup is required before AI generation can run.</p></section></div></div>`}
