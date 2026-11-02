// Phone 3D Scanner AI - Frontend Controller



let selectedFiles = [];

let currentJobId = null;

let pollInterval = null;

let currentMode = 'ai_depth'; // Default to single-image generation. Advanced: 'turntable', 'diffusion', 'multi'



// Auth State

let currentUser = null;

let authToken = localStorage.getItem('authToken');



// Three.js State

let scene, camera, renderer, controls;

let currentMesh = null;

let loadingPointCloud = null;

let isWireframe = false;

let isTextured = false;

let isClay = true;

let hasVertexColors = false;

let hasTextureMaps = false;



document.addEventListener('DOMContentLoaded', () => {

    initThreeJS();

    initHomeAnimation();

    initAuth();

    initDropzone();

    initControls();

    initAdvancedModes();

    loadEngineStatus();

    initAiTrainingControls();

    initGallery();

    checkAndLoadLatestScan();

    loadDatasetInventory();

});



// Auto-load latest completed scan if available

async function checkAndLoadLatestScan() {

    try {

        const res = await fetch('/api/scan/latest');

        if (res.ok) {

            const data = await res.json();

            if (data.job_id) {

                currentJobId = data.job_id;

                logConsole(`Restored latest 3DINVENZA 3D scan session: ${currentJobId}`, 'system');

                loadScanModel(currentJobId);

                fetchTopologyReport(currentJobId);

                fetchScanQualityReport(currentJobId);

                setupExportButtons(currentJobId);

            }

        }

    } catch (e) {

        console.log('No previous scan to auto-load:', e);

    }

}



// Initialize Three.js 3D Viewport

function initThreeJS() {

    const container = document.getElementById('canvasContainer');

    const width = container.clientWidth || 800;

    const height = container.clientHeight || 500;



    scene = new THREE.Scene();

    scene.background = null; // Transparent for radial gradient



    // Camera

    camera = new THREE.PerspectiveCamera(45, width / height, 0.05, 1000);

    camera.position.set(0, 1.2, 2.8);



    // Renderer

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });

    renderer.setSize(width, height);

    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputEncoding = THREE.sRGBEncoding;

    renderer.shadowMap.enabled = true;

    container.appendChild(renderer.domElement);



    // Controls

    controls = new THREE.OrbitControls(camera, renderer.domElement);

    controls.enableDamping = true;

    controls.dampingFactor = 0.05;



    // Studio 3-Point Balanced Lighting (Key, Fill, Rim)

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.35);

    scene.add(ambientLight);



    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x475569, 0.45);

    hemiLight.position.set(0, 20, 0);

    scene.add(hemiLight);



    const keyLight = new THREE.DirectionalLight(0xffffff, 0.55);

    keyLight.position.set(5, 8, 6);

    scene.add(keyLight);



    const fillLight = new THREE.DirectionalLight(0xd5d1c8, 0.2);

    fillLight.position.set(-6, 3, -6);

    scene.add(fillLight);



    const rimLight = new THREE.DirectionalLight(0x8faeb0, 0.15);

    rimLight.position.set(0, 8, -6);

    scene.add(rimLight);



    // Grid Floor

    const grid = new THREE.GridHelper(10, 20, 0xc8a486, 0x34515b);

    grid.position.y = -1.05; // Models are centered and normalized to a two-unit bounding box.

    scene.add(grid);



    // Resize Handler

    window.addEventListener('resize', onWindowResize);



    animate();

}



function animate() {

    requestAnimationFrame(animate);

    if (controls) controls.update();

    

    if (loadingPointCloud && loadingPointCloud.visible) {

        loadingPointCloud.material.opacity = .75 + Math.sin(performance.now() / 400) * .15;

        // Real 3D points retain world orientation while the camera orbits.

    }

    

    if (renderer && scene && camera) renderer.render(scene, camera);

}



// Loading Point Cloud Animation

function createLoadingPointCloud() {

    if (loadingPointCloud) {

        scene.remove(loadingPointCloud);

        if (loadingPointCloud.geometry) loadingPointCloud.geometry.dispose();

        if (loadingPointCloud.material) loadingPointCloud.material.dispose();

        loadingPointCloud = null;

    }

    

    const particleCount = 20000;

    const geometry = new THREE.BufferGeometry();

    const positions = new Float32Array(particleCount * 3);

    const colors = new Float32Array(particleCount * 3);



    for (let i = 0; i < particleCount; i++) {

        const radius = 1.0 * Math.random();

        const angle = Math.random() * 2 * Math.PI;

        const height = (Math.random() - 0.5) * 2.5;



        positions[i * 3] = radius * Math.cos(angle) + (Math.random() - 0.5) * 0.2;

        positions[i * 3 + 1] = height + (Math.random() - 0.5) * 0.2;

        positions[i * 3 + 2] = radius * Math.sin(angle) + (Math.random() - 0.5) * 0.2;



        const colorVal = 0.4 + Math.random() * 0.6;

        colors[i * 3] = colorVal;

        colors[i * 3 + 1] = colorVal;

        colors[i * 3 + 2] = colorVal;

    }



    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));



    const material = new THREE.PointsMaterial({

        size: 0.015,

        vertexColors: true,

        transparent: true,

        opacity: 0.7,

        sizeAttenuation: true

    });



    loadingPointCloud = new THREE.Points(geometry, material);

    scene.add(loadingPointCloud);

}



function startLoadingAnimation() {

    createLoadingPointCloud();

    loadingPointCloud.visible = true;

    if (currentMesh) {

        scene.remove(currentMesh);

        currentMesh = null;

    }

    const placeholder = document.getElementById('viewportPlaceholder');

    if (placeholder) placeholder.style.display = 'none';

    const overlay = document.getElementById('generatingOverlay');

    if (overlay) overlay.style.display = 'flex';

}



function stopLoadingAnimation() {

    if (loadingPointCloud) {

        loadingPointCloud.visible = false;

    }

    const overlay = document.getElementById('generatingOverlay');

    if (overlay) overlay.style.display = 'none';

}



function onWindowResize() {

    const container = document.getElementById('canvasContainer');

    if (!container || !renderer || !camera) return;

    const width = container.clientWidth;

    const height = container.clientHeight;

    camera.aspect = width / height;

    camera.updateProjectionMatrix();

    renderer.setSize(width, height);

}



// Drag & Drop Handling

function initDropzone() {

    const dropzone = document.getElementById('dropzone');

    const fileInput = document.getElementById('fileInput');



    if (dropzone) {

        dropzone.addEventListener('dragover', (e) => {

            e.preventDefault();

            dropzone.classList.add('dragover');

        });



        dropzone.addEventListener('dragleave', () => {

            dropzone.classList.remove('dragover');

        });



        dropzone.addEventListener('drop', (e) => {

            e.preventDefault();

            dropzone.classList.remove('dragover');

            if (e.dataTransfer.files.length > 0) {

                handleFiles(Array.from(e.dataTransfer.files));

            }

        });

    }



    if (fileInput) {

        fileInput.addEventListener('change', (e) => {

            if (e.target.files.length > 0) {

                handleFiles(Array.from(e.target.files));

            }

        });

    }



    const clearBtn = document.getElementById('clearFilesBtn');

    if (clearBtn) {

        clearBtn.addEventListener('click', () => {

            selectedFiles = [];

            document.getElementById('filePreviewBox').style.display = 'none';

            document.getElementById('dropzone').style.display = 'block';

            logConsole('Cleared file selections.', 'system');

        });

    }



    const startBtn = document.getElementById('startScanBtn');

    if (startBtn) startBtn.addEventListener('click', startPipeline);

    

    const startBtnEngine = document.getElementById('startScanBtnEngine');

    if (startBtnEngine) startBtnEngine.addEventListener('click', startPipeline);

}



function handleFiles(files) {

    selectedFiles = files;

    document.getElementById('filePreviewBox').style.display = 'block';

    document.getElementById('fileCountText').innerHTML = `<i class="fa-solid fa-check-circle"></i> ${selectedFiles.length} file(s) selected`;

    logConsole(`Selected ${selectedFiles.length} file(s) for 3D scan.`, 'system');



    // Show image preview for single-image modes

    if (currentMode === 'ai_depth' || currentMode === 'diffusion') {

        const preview = document.getElementById('unifiedPreview');

        const previewImg = document.getElementById('unifiedPreviewImg');

        const dropzone = document.getElementById('dropzone');

        if (preview && previewImg && files.length > 0) {

            const reader = new FileReader();

            reader.onload = (e) => {

                previewImg.src = e.target.result;

                preview.style.display = 'block';

                dropzone.style.display = 'none';

            };

            reader.readAsDataURL(files[0]);

        }

    } else {

        document.getElementById('dropzone').style.display = 'none';

    }

}



// UI Controls & Navigation

function initControls() {

    // Top Navigation Tabs

    const navHome = document.getElementById('navHome');

    const navWorkspace = document.getElementById('navWorkspace');

    const navGallery = document.getElementById('navGallery');

    const navProfile = document.getElementById('navProfile');

    const navAuth = document.getElementById('navAuth');

    const navAdmin = document.getElementById('navAdmin');

    

    const homeView = document.getElementById('homeView');

    const workspaceView = document.getElementById('workspaceView');

    const galleryView = document.getElementById('galleryView');

    const profileView = document.getElementById('profileView');

    const authView = document.getElementById('authView');

    const adminView = document.getElementById('adminView');



    function switchView(activeNav, activeView) {

        [navHome, navWorkspace, navGallery, navProfile, navAuth, navAdmin].forEach(n => {

            if (n) n.classList.remove('active');

        });

        [homeView, workspaceView, galleryView, profileView, authView, adminView].forEach(v => {

            if (v) v.style.display = 'none';

        });

        if (activeNav) activeNav.classList.add('active');

        if (activeView) activeView.style.display = activeView.id === 'workspaceView' ? 'grid' : 'flex';

        

        if (activeView === workspaceView) onWindowResize();

    }



    if (navHome) navHome.addEventListener('click', () => switchView(navHome, homeView));

    if (navWorkspace) navWorkspace.addEventListener('click', () => switchView(navWorkspace, workspaceView));

    if (navGallery) {

        navGallery.addEventListener('click', () => {

            switchView(navGallery, galleryView);

            loadGallery();

        });

    }

    if (navProfile) navProfile.addEventListener('click', () => switchView(navProfile, profileView));

    if (navAuth) navAuth.addEventListener('click', () => switchView(navAuth, authView));

    if (navAdmin) {

        navAdmin.addEventListener('click', () => {

            switchView(navAdmin, adminView);

            loadAdminDashboard();

        });

    }



    const btnHomeToWorkspace = document.getElementById('btnHomeToWorkspace');

    if (btnHomeToWorkspace) {

        btnHomeToWorkspace.addEventListener('click', () => {

            if (navWorkspace) navWorkspace.click();

        });

    }



    // Advanced Mode switcher buttons (inside collapsed panel)

    const modeTurntableBtn = document.getElementById('modeTurntableBtn');

    const modeDiffusionBtn = document.getElementById('modeDiffusionBtn');

    const modeMultiBtn = document.getElementById('modeMultiBtn');

    const fileInput = document.getElementById('fileInput');



    function resetModes() {

        if (modeTurntableBtn) modeTurntableBtn.classList.remove('active');

        if (modeDiffusionBtn) modeDiffusionBtn.classList.remove('active');

        if (modeMultiBtn) modeMultiBtn.classList.remove('active');

    }



    function switchToSingleImageMode() {

        currentMode = 'ai_depth';

        resetModes();

        if (fileInput) fileInput.removeAttribute('multiple');

        document.getElementById('dropzoneTitle').innerText = 'Drop your image here';

        document.getElementById('dropzoneDesc').innerText = 'Upload a photo to generate a 3D model';

        const scannerCfg = document.getElementById('scannerConfigBar');

        if (scannerCfg) scannerCfg.style.display = 'none';

    }



    if (modeTurntableBtn) {

        modeTurntableBtn.addEventListener('click', () => {

            currentMode = 'turntable';

            resetModes();

            modeTurntableBtn.classList.add('active');

            if (fileInput) fileInput.setAttribute('multiple', 'true');

            document.getElementById('dropzoneTitle').innerText = 'Upload 24–72 Turntable Frames';

            document.getElementById('dropzoneDesc').innerText = 'Select rotational photos or a .ZIP for 360° reconstruction';

            document.getElementById('scannerConfigBar').style.display = 'flex';

            // Reset single-image preview

            clearUnifiedPreview();

            logConsole('Switched to Calibrated Turntable Scanner mode.', 'system');

        });

    }



    if (modeDiffusionBtn) {

        modeDiffusionBtn.addEventListener('click', () => {

            currentMode = 'diffusion';

            resetModes();

            modeDiffusionBtn.classList.add('active');

            if (fileInput) fileInput.removeAttribute('multiple');

            document.getElementById('dropzoneTitle').innerText = 'Upload Photo for 360° Diffusion';

            document.getElementById('dropzoneDesc').innerText = 'Synthesizes 360° novel views to resolve unseen geometry';

            document.getElementById('scannerConfigBar').style.display = 'none';

            logConsole('Switched to 360° AI Multi-View Diffusion mode.', 'system');

        });

    }



    if (modeMultiBtn) {

        modeMultiBtn.addEventListener('click', () => {

            currentMode = 'multi';

            resetModes();

            modeMultiBtn.classList.add('active');

            if (fileInput) fileInput.setAttribute('multiple', 'true');

            document.getElementById('dropzoneTitle').innerText = 'Drag & Drop Sequence Frames';

            document.getElementById('dropzoneDesc').innerText = 'Upload sequence images or a .zip file';

            document.getElementById('scannerConfigBar').style.display = 'flex';

            clearUnifiedPreview();

            logConsole('Switched to Unconstrained Multi-View Sequence mode.', 'system');

        });

    }



    // AI Engine (model selector) change handler

    const aiEngineSelect = document.getElementById('aiEngineSelect');

    if (aiEngineSelect) {

        aiEngineSelect.addEventListener('change', () => {

            const key = aiEngineSelect.value;

            updateEngineStatusBadge(key);

            logConsole(`AI Engine set to ${getSelectedEngineLabel()}.`, 'system');

        });

    }



    // Unified preview remove button

    const previewRemoveBtn = document.getElementById('unifiedPreviewRemove');

    if (previewRemoveBtn) {

        previewRemoveBtn.addEventListener('click', () => {

            clearUnifiedPreview();

            selectedFiles = [];

            document.getElementById('filePreviewBox').style.display = 'none';

            logConsole('Cleared file selection.', 'system');

        });

    }



    // Viewport Tool Buttons

    const btnClay = document.getElementById('btnClay');

    if (btnClay) {

        btnClay.addEventListener('click', () => {

            isClay = true;

            isWireframe = false;

            isTextured = false;

            applyClayMode();

            setActiveToolBtn('btnClay');

        });

    }



    document.getElementById('btnSolid').addEventListener('click', () => {

        isClay = false;

        isWireframe = false;

        isTextured = false;

        applySolidColorMode(0xc8a486);

        setActiveToolBtn('btnSolid');

    });



    document.getElementById('btnWireframe').addEventListener('click', () => {

        isWireframe = true;

        setMeshWireframe(true);

        setActiveToolBtn('btnWireframe');

    });



    document.getElementById('btnResetCam').addEventListener('click', () => {

        if (camera && controls) {

            camera.position.set(0, 1.2, 2.8);

            controls.target.set(0, 0, 0);

            controls.update();

        }

    });



    document.getElementById('btnTextured').addEventListener('click', () => {

        isTextured = !isTextured;

        isClay = false;

        const btn = document.getElementById('btnTextured');

        if (isTextured) {

            btn.classList.add('active');

            setActiveToolBtn('btnTextured');

        } else {

            btn.classList.remove('active');

        }

        applyTexturedMode(isTextured);

    });

}



// Advanced Modes Toggle

function initAdvancedModes() {

    const toggleBtn = document.getElementById('advancedModesToggle');

    const panel = document.getElementById('advancedModesPanel');

    if (toggleBtn && panel) {

        toggleBtn.addEventListener('click', () => {

            toggleBtn.classList.toggle('open');

            panel.classList.toggle('open');

        });

    }

}



// Clear unified image preview

function clearUnifiedPreview() {

    const preview = document.getElementById('unifiedPreview');

    const previewImg = document.getElementById('unifiedPreviewImg');

    const dropzone = document.getElementById('dropzone');

    if (preview) preview.style.display = 'none';

    if (previewImg) previewImg.src = '';

    if (dropzone) dropzone.style.display = 'block';

}



// =========================================================================

//  AI Engine Model Selector

// =========================================================================

let engineCatalog = null;



function getSelectedEngine() {

    const sel = document.getElementById('aiEngineSelect');

    return sel ? sel.value : 'trellis';

}



function getSelectedEngineLabel() {

    const sel = document.getElementById('aiEngineSelect');

    if (!sel) return 'TripoSR';

    return sel.options[sel.selectedIndex].text;

}



function updateEngineStatusBadge(key) {

    const badge = document.getElementById('engineStatusBadge');

    if (!badge) return;



    if (!engineCatalog) {

        badge.innerHTML = '<i class="fa-solid fa-circle-question"></i> Status unknown';

        return;

    }



    const info = engineCatalog.engines.find(e => e.key === key);

    if (!info) {

        badge.innerHTML = '<i class="fa-solid fa-circle-question"></i> Status unknown';

        return;

    }



    if (info.available) {

        badge.innerHTML = '<i class="fa-solid fa-circle-check"></i> Package detected';

        badge.title = `${info.label} import check passed: ${info.detail}`;

        badge.style.color = 'var(--status-green)';

    } else {

        const chain = (info.fallbacks && info.fallbacks.length)

            ? ` Will fall back to: ${info.fallbacks.join(' → ')}.`

            : ' No fallback available.';

        badge.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Setup not ready';

        badge.title = `${info.label} readiness check did not pass.${chain}`;

        badge.style.color = 'var(--status-amber)';

    }

}



async function loadEngineStatus() {

    try {

        const res = await fetch('/api/engines');

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        engineCatalog = await res.json();



        const sel = document.getElementById('aiEngineSelect');

        if (sel) {

            const chosen = engineCatalog.engines.find(info => info.key === sel.value);
            if (!chosen?.available) {
                const ready = ['trellis','hunyuan3d','triposr'].map(key=>engineCatalog.engines.find(info=>info.key===key)).find(info=>info?.available);
                if (ready && Array.from(sel.options).some(option => option.value === ready.key)) sel.value = ready.key;
            }
            engineCatalog.engines.forEach(info => {

                const opt = Array.from(sel.options).find(o => o.value === info.key);

                if (opt && !info.available && !opt.text.includes('⚠')) {

                    opt.text = `${opt.text} ⚠ setup not ready`;

                }

            });

        }



        updateEngineStatusBadge(getSelectedEngine());

    } catch (err) {

        console.warn(`Could not load engine catalog: ${err.message}`);

    }

}



function setActiveToolBtn(id) {

    document.querySelectorAll('.view-toggles .toggle-btn').forEach(btn => btn.classList.remove('active'));

    document.getElementById(id).classList.add('active');

}



function setMeshWireframe(wireframe) {

    if (currentMesh) {

        currentMesh.traverse((child) => {

            if (child.isMesh) child.material.wireframe = wireframe;

        });

    }

}



function applyClayMode() {

    if (!currentMesh) return;

    currentMesh.traverse((child) => {

        if (child.isMesh) {

            child.material.vertexColors = false;

            child.material.map = null;

            child.material.wireframe = false;

            child.material.color.set(0x9ca3af); // Gray

            child.material.roughness = 0.8;

            child.material.metalness = 0.1;

            child.material.needsUpdate = true;

        }

    });

}



function applySolidColorMode(colorHex = 0xc8a486) {

    if (!currentMesh) return;

    currentMesh.traverse((child) => {

        if (child.isMesh) {

            child.material.vertexColors = false;

            child.material.map = null;

            child.material.wireframe = false;

            child.material.color.set(colorHex);

            child.material.roughness = 0.4;

            child.material.metalness = 0.2;

            child.material.needsUpdate = true;

        }

    });

}



function applyTexturedMode(textured) {

    if (!currentMesh) return;

    currentMesh.traverse((child) => {

        if (child.isMesh) {

            child.material.map = textured ? (child.userData.nativeTexture || null) : null;

            child.material.wireframe = false;

            child.material.roughness = 0.65;

            child.material.metalness = 0;

            if (textured && (hasVertexColors || hasTextureMaps)) {

                child.material.vertexColors = !!child.geometry.attributes.color;

                child.material.color.set(0xffffff);

            } else {

                child.material.vertexColors = false;

                child.material.color.set(0x9ca3af);

            }

            child.material.needsUpdate = true;

        }

    });

}



// Console Logging

function logConsole(message, type = 'normal') {

    const consoleBody = document.getElementById('consoleLogs');

    if (!consoleBody) return;

    const entry = document.createElement('p');

    entry.className = `log-entry ${type}`;

    entry.innerText = `> ${message}`;

    consoleBody.appendChild(entry);

    consoleBody.scrollTop = consoleBody.scrollHeight;

    

    // Also update bottom status bar

    const statusBarText = document.getElementById('statusBarText');

    if (statusBarText) statusBarText.innerText = message;

}



// Pipeline Execution

async function startPipeline() {

    if (selectedFiles.length === 0) {

        logConsole('Please upload an image first.', 'error');

        return;

    }



    const startBtn = document.getElementById('startScanBtn');

    const startBtnEngine = document.getElementById('startScanBtnEngine');

    

    if (startBtn) {

        startBtn.disabled = true;

        startBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Initializing...';

    }

    if (startBtnEngine) {

        startBtnEngine.disabled = true;

        startBtnEngine.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating...';

    }



    startLoadingAnimation();



    let endpoint;

    const stepSelect = document.getElementById('turntableStepSelect');

    const step = stepSelect ? parseFloat(stepSelect.value) : 15.0;

    const expected = Math.round(360.0 / step);



    if (currentMode === 'turntable') {

        endpoint = `/api/scan/turntable-calibrated?rotation_step=${step}&expected_frames=${expected}`;

    } else if (currentMode === 'diffusion') {

        endpoint = '/api/scan/diffusion-3d';

    } else if (currentMode === 'multi') {

        endpoint = '/api/scan/upload';

    } else {

        // Default: unified single-image generation via /api/reconstruct

        const engine = getSelectedEngine();

        const texture = true;

        const quality = document.getElementById('qualitySelect')?.value || 'high';

        const objectType = document.getElementById('objectTypeSelect')?.value || 'organic';

        const refine = document.getElementById('refineToggle')?.checked ?? true;

        const printable = document.getElementById('printableToggle')?.checked ?? false;

        endpoint = engine === 'ai_depth' ? '/api/scan/ai-depth' : '/api/reconstruct';

    }



    const formData = new FormData();

    if (!['turntable', 'diffusion', 'multi'].includes(currentMode)) {

        formData.append('engine', getSelectedEngine());

        formData.append('object_type', document.getElementById('objectTypeSelect')?.value || 'organic');

        formData.append('quality', document.getElementById('qualitySelect')?.value || 'high');

        formData.append('texture', 'true');
        formData.append('photo_detail',String(document.getElementById('photoDetailToggle')?.checked ?? false));

        formData.append('refine', String(document.getElementById('refineToggle')?.checked ?? true));

        formData.append('printable', String(document.getElementById('printableToggle')?.checked ?? false));

    }



    if (currentMode === 'turntable' || currentMode === 'multi') {

        selectedFiles.forEach(file => formData.append('files', file));

    } else {

        formData.append('file', selectedFiles[0]);

    }



    try {

        const response = await fetch(endpoint, { method: 'POST', body: formData });

        const data = await response.json();



        if (response.ok && data.job_id) {

            currentJobId = data.job_id;

            logConsole(`Job queued with ID: ${currentJobId}`, 'system');

            startPolling(currentJobId);

        } else {

            throw new Error(data.detail || 'Upload failed');

        }

    } catch (err) {

        logConsole(`Upload error: ${err.message}`, 'error');

        resetStartButtons();

        stopLoadingAnimation();

    }

}



function resetStartButtons() {

    const startBtn = document.getElementById('startScanBtn');

    const startBtnEngine = document.getElementById('startScanBtnEngine');

    if (startBtn) {

        startBtn.disabled = false;

        startBtn.innerHTML = '<i class="fa-solid fa-play"></i> Start Capture';

    }

    if (startBtnEngine) {

        startBtnEngine.disabled = false;

        startBtnEngine.innerHTML = '<i class="fa-solid fa-rocket"></i> Generate 3D Model';

    }

}



// Status Polling

function startPolling(jobId) {
    refreshSourcePointCloud(jobId);

    document.getElementById('pipelineProgressSection').style.display = 'block';



    pollInterval = setInterval(async () => {

        try {

            const res = await fetch(`/api/scan/status/${jobId}`);

            if (!res.ok) return;



            const job = await res.json();

            updateProgressUI(job);



            if (job.status === 'completed') {

                clearInterval(pollInterval);

                logConsole('Pipeline finished successfully! Loading 3D model...', 'system');

                stopLoadingAnimation();

                loadScanModel(jobId);

                fetchTopologyReport(jobId);

                fetchScanQualityReport(jobId);

                setupExportButtons(jobId);

                resetStartButtons();

                // Refresh gallery silently

                loadGallery();

            } else if (job.status === 'failed') {

                clearInterval(pollInterval);

                logConsole(`Pipeline failed: ${job.error}`, 'error');

                stopLoadingAnimation();

                resetStartButtons();

            }

        } catch (err) {

            console.error('Status poll error:', err);

        }

    }, 1500);

}



function updateProgressUI(job) {
    const timing=document.getElementById('generationProgressDetails');
    if(timing){const seconds=Math.max(0,Math.floor(job.elapsed_seconds||0));timing.textContent=`${job.step||job.status} · Elapsed ${Math.floor(seconds/60)}m ${seconds%60}s`;}


    const progressBar = document.getElementById('progressBar');

    if (progressBar) progressBar.style.width = `${job.progress}%`;



    const stages = ['stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6'];

    let activeIdx = 0;



    if (job.progress >= 15 && job.progress < 35) activeIdx = 0;

    else if (job.progress >= 35 && job.progress < 55) activeIdx = 1;

    else if (job.progress >= 55 && job.progress < 75) activeIdx = 2;

    else if (job.progress >= 75 && job.progress < 88) activeIdx = 3;

    else if (job.progress >= 88 && job.progress < 98) activeIdx = 4;

    else if (job.progress >= 98) activeIdx = 5;



    stages.forEach((stgId, idx) => {

        const el = document.getElementById(stgId);

        if (!el) return;

        

        // BUGFIX: Set class directly to pipe-item active/completed

        if (idx < activeIdx || job.progress === 100) {

            el.className = 'pipe-item completed';

            const icon = el.querySelector('.pipe-icon');

            if(icon) icon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';

        } else if (idx === activeIdx && job.progress < 100) {

            el.className = 'pipe-item active';

            const icon = el.querySelector('.pipe-icon');

            if(icon) icon.innerHTML = '<i class="fa-solid fa-circle-dot"></i>';

        } else {

            el.className = 'pipe-item';

            const icon = el.querySelector('.pipe-icon');

            if(icon) icon.innerHTML = '<i class="fa-regular fa-circle"></i>';

        }

    });



    if (job.logs && job.logs.length > 0) {

        const latest = job.logs[job.logs.length - 1];

        const statusBarText = document.getElementById('statusBarText');

        if (statusBarText) statusBarText.innerText = latest;

    }

}



async function fetchScanQualityReport(jobId) {
    const validationStatus=document.getElementById('modelValidationStatus');
    if(validationStatus) validationStatus.textContent='Checking mesh…';

    try {

        const res = await fetch(`/api/scan/quality/${jobId}`);

        if (!res.ok) {if(validationStatus)validationStatus.textContent='Not checked';return;}

        const data = await res.json();



        if(validationStatus){validationStatus.textContent=data.passed===true?'Mesh checks passed':(data.status || 'Review mesh');validationStatus.title='Mesh checks do not measure similarity to the source image.';}
        if (data.metrics) {
            const metrics = data.metrics;
            for (const [id, text] of Object.entries({
                valManifold: metrics.non_manifold_edges >= 0 ? `Manifold: ${metrics.non_manifold_edges === 0 ? 'yes' : 'needs repair'}` : 'Manifold: not checked',
                valSelfIntersection: metrics.self_intersections_reported >= 0 ? `Self-intersections: ${metrics.self_intersections_reported}` : 'Self-intersections: not checked',
                valNormals: `Normals: ${metrics.winding_consistent ? 'consistent' : 'needs repair'}`
            })) { const element = document.getElementById(id); if (element) element.innerText = text; }
            const overall = document.getElementById('validationOverall');
            if (overall) overall.innerText = data.status || 'NOT CHECKED';
        }
        const card = document.getElementById('scanQualityCard');

        if (card) card.style.display = 'block';



        const badge = document.getElementById('scanQualityScoreBadge');

        if (badge) {

            badge.innerText = Number.isFinite(data.overall_score) ? `Score: ${data.overall_score}%` : (data.status || "Quality pending");

            badge.title = [...(data.warnings || []), ...(data.errors || [])].join("; ");

            if (data.passed === true || data.overall_score >= 80) {

                badge.style.color = 'var(--status-green)';

            } else {

                badge.style.color = 'var(--status-amber)';

            }

        }

    } catch (e) {

        if(validationStatus)validationStatus.textContent='Check unavailable';
        console.log('Scan quality fetch notice:', e);

    }

}



async function fetchTopologyReport(jobId) {

    try {

        const res = await fetch(`/api/scan/topology/${jobId}`);

        if (!res.ok) return;

        const data = await res.json();



        const dimX = document.getElementById('dimX');

        const dimY = document.getElementById('dimY');

        const dimZ = document.getElementById('dimZ');

        const relative = data.coordinate_space === 'normalized';

        const qualityCard = document.getElementById('scanQualityCard');
        if (qualityCard) {
            qualityCard.querySelector('.section-title').textContent = relative ? 'MESH QUALITY' : 'METROLOGY';
            qualityCard.querySelectorAll('.stat-row').forEach((row,index) => {if(index < 2) row.style.display = relative ? 'none' : '';});
        }
        const calibration = document.querySelector('.calibration-card')?.closest('.section-block');
        if (calibration) calibration.style.display = relative ? 'none' : '';
        const dimensions = data.dimensions_units || data.dimensions_mm || {};

        const x = relative ? dimensions.x : (data.length_mm ?? dimensions.depth);

        const y = relative ? dimensions.y : (data.width_mm ?? dimensions.width);

        const z = relative ? dimensions.z : (data.height_mm ?? dimensions.height);

        if (dimX) dimX.innerText = x ?? '--';

        if (dimY) dimY.innerText = y ?? '--';

        if (dimZ) dimZ.innerText = z ?? '--';

        const obbEl = document.getElementById('topoObbDimensions');

        if (obbEl) {

            obbEl.innerText = `${x ?? '--'} x ${y ?? '--'} x ${z ?? '--'} ${relative ? 'relative units' : 'mm'}`;

            obbEl.title = relative ? 'Set the physical scale before printing. A single image does not measure real dimensions.' : '';

        }

        const volEl = document.getElementById('topoVolume');

        const volume = relative ? data.volume_units3 : data.volume_mm3;

        if (volEl) volEl.innerText = Number.isFinite(volume) ? `${volume.toLocaleString()} ${relative ? 'units^3' : 'mm^3'}` : '--';



        const wtEl = document.getElementById('topoWatertight');

        if (wtEl) {

            wtEl.innerHTML = data.is_watertight ? '<i class="fa-solid fa-circle-check"></i>' : '<i class="fa-solid fa-triangle-exclamation"></i>';

            wtEl.style.color = data.is_watertight ? 'var(--status-green)' : 'var(--status-amber)';

        }



        const facesEl = document.getElementById('topoFaces');

        if (facesEl) facesEl.innerText = `${data.faces.toLocaleString()}`;

        

        const vertsEl = document.getElementById('statVerts');

        if (vertsEl) vertsEl.innerText = `${data.vertices.toLocaleString()}`;

        

        const statFaces = document.getElementById('statFaces');

        if (statFaces) statFaces.innerText = `${data.faces.toLocaleString()}`;



    } catch (err) {

        console.error('Topology report error:', err);

    }

}



// Load 3D Model into Viewport

function loadScanModel(jobId) {
    if(typeof loadGenerationSummary==='function')loadGenerationSummary(jobId);
    clearPointView();

    const glbUrl = `/api/scan/download/${jobId}/glb`;

    const stlUrl = `/api/scan/download/${jobId}/stl`;



    const placeholder = document.getElementById('viewportPlaceholder');

    if (placeholder) placeholder.style.display = 'none';



    // BUGFIX: GLTFLoader and STLLoader now exist because CDN tags added to index.html

    const gltfLoader = new THREE.GLTFLoader();

    gltfLoader.load(glbUrl, (gltf) => {

        if (currentMesh) scene.remove(currentMesh);



        currentMesh = gltf.scene;
        prepareScannerAppearance(currentMesh);

        hasVertexColors = false;

        hasTextureMaps = false;



        currentMesh.traverse((child) => {

            if (child.isMesh) {

                child.userData.nativeTexture = child.material.map || null;

                hasTextureMaps = hasTextureMaps || !!child.material.map;

                child.material.side = THREE.DoubleSide;

                child.castShadow = true;

                child.receiveShadow = true;

                if (child.geometry && child.geometry.attributes && child.geometry.attributes.color) {

                    hasVertexColors = true;

                    child.material.vertexColors = true;

                    child.material.color.set(0xffffff);

                } else if (!child.material.map) {

                    child.material.color.set(0x9ca3af);

                    child.material.roughness = 0.8;

                    child.material.metalness = 0.1;

                }

                child.material.wireframe = isWireframe;

                child.material.needsUpdate = true;

            }

        });



        // Compute Bounding Box & Center

        const box = new THREE.Box3().setFromObject(currentMesh);

        const center = new THREE.Vector3();

        box.getCenter(center);

        const size = new THREE.Vector3();

        box.getSize(size);

        const maxDim = Math.max(size.x, size.y, size.z) || 1.0;

        const scale = 2.0 / maxDim;



        currentMesh.scale.set(scale, scale, scale);

        currentMesh.position.set(-center.x * scale, -center.y * scale, -center.z * scale);



        scene.add(currentMesh);



        camera.position.set(2.4, 1.3, -3.3);
        if (currentMesh.userData.source_view) camera.position.fromArray(currentMesh.userData.source_view).multiplyScalar(3.3);

        camera.lookAt(0, 0, 0);

        controls.target.set(0, 0, 0);

        controls.update();



        const texturedBtn = document.getElementById('btnTextured');

        if (hasVertexColors) {

            isTextured = true;

            if (texturedBtn) texturedBtn.classList.add('active');

        } else {

            if (texturedBtn) texturedBtn.classList.remove('active');

        }

        

        isWireframe = false;

        isTextured = hasVertexColors || hasTextureMaps;

        if (isTextured) {

            applyTexturedMode(true);

            setActiveToolBtn('btnTextured');

        } else {

            applySolidColorMode(0xc8a486);

            setActiveToolBtn('btnSolid');

        }



    }, undefined, (gltfError) => {

        console.warn('GLB load failed, falling back to STL:', gltfError);



        const stlLoader = new THREE.STLLoader();

        stlLoader.load(stlUrl, (geometry) => {

            if (currentMesh) scene.remove(currentMesh);



            geometry.computeVertexNormals();



            // BUGFIX: Compute scale for STL fallback so it doesn't crash

            geometry.computeBoundingBox();

            const box = geometry.boundingBox;

            const center = new THREE.Vector3();

            box.getCenter(center);

            const size = new THREE.Vector3();

            box.getSize(size);

            const maxDim = Math.max(size.x, size.y, size.z) || 1.0;

            const scale = 2.0 / maxDim;



            const material = new THREE.MeshStandardMaterial({

                color: 0xc8a486, // Accent color

                roughness: 0.4,

                metalness: 0.2,

                side: THREE.DoubleSide

            });



            currentMesh = new THREE.Mesh(geometry, material);

            currentMesh.scale.set(scale, scale, scale);

            // Center it

            currentMesh.position.set(-center.x * scale, -center.y * scale, -center.z * scale);

            

            scene.add(currentMesh);



            camera.position.set(0, 1.0, 2.5);

            camera.lookAt(0, 0, 0);

            controls.target.set(0, 0, 0);

            controls.update();

            

            applySolidColorMode(0xc8a486);

            setActiveToolBtn('btnSolid');

            

        }, undefined, (stlErr) => {

            console.error('Failed to load STL model:', stlErr);

        });

    });

}



function setupExportButtons(jobId) {

    const exportBtn = document.getElementById('downloadStlBtn');

    const fmtSelect = document.getElementById('exportFormatSelect');

    

    if (exportBtn && fmtSelect) {

        exportBtn.onclick = () => {

            const fmt = fmtSelect.value;

            window.open(`/api/scan/download/${jobId}/${fmt}`, '_blank');

        };

    }

}



// =========================================================================

//  Gallery Controller

// =========================================================================



function initGallery() {

    const btnScanExisting = document.getElementById('btnScanExisting');

    const btnRefreshGallery = document.getElementById('btnRefreshGallery');

    const btnToWorkspace = document.getElementById('btnGalleryToWorkspace');

    

    if (btnScanExisting) {

        btnScanExisting.addEventListener('click', async () => {

            btnScanExisting.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Scanning...';

            try {

                const res = await fetch('/api/models/scan-existing', { method: 'POST' });

                const data = await res.json();

                console.log(`Scanned ${data.scanned} jobs, added ${data.added} to DB`);

                loadGallery();

            } catch (e) {

                console.error("Scan existing error:", e);

            }

            btnScanExisting.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i> Import Existing';

        });

    }

    

    if (btnRefreshGallery) btnRefreshGallery.addEventListener('click', loadGallery);

    

    if (btnToWorkspace) {

        btnToWorkspace.addEventListener('click', () => {

            document.getElementById('navWorkspace').click();

        });

    }

    

    loadGallery();

}



async function loadGallery() {

    try {

        const res = await fetch('/api/models');

        if (!res.ok) return;

        const data = await res.json();

        

        const grid = document.getElementById('galleryGrid');

        const empty = document.getElementById('galleryEmpty');

        const badge = document.getElementById('galleryCountBadge');

        const total = document.getElementById('galleryTotal');

        

        if (badge) badge.innerText = data.total;

        if (total) total.innerText = `${data.total} models`;

        

        // Clear grid except for the empty state

        Array.from(grid.children).forEach(child => {

            if (child.id !== 'galleryEmpty') grid.removeChild(child);

        });

        

        if (data.models.length === 0) {

            empty.style.display = 'flex';

        } else {

            empty.style.display = 'none';

            data.models.forEach(model => {

                grid.appendChild(createGalleryCard(model));

            });

        }

    } catch (e) {

        console.error("Failed to load gallery:", e);

    }

}



function createGalleryCard(model) {

    const card = document.createElement('div');

    card.className = 'gallery-card glass-card';

    

    const dateStr = new Date(model.created_at * 1000).toLocaleDateString() + ' ' + 

                    new Date(model.created_at * 1000).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});

                    

    const escapeText=value=>String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const modelName=escapeText(model.name || 'Untitled model');
    const engineLabel = escapeText(model.engine_label || model.engine || '3D Scan');

    

    const facesStr = model.faces > 0 ? model.faces.toLocaleString() : '--';

    let scaleMetadata={};
    try{scaleMetadata=typeof model.extra_json==='string'?JSON.parse(model.extra_json):(model.extra||{});}catch(error){}
    const calibrated=scaleMetadata?.scale_calibrated===true||scaleMetadata?.quality?.metrics?.scale_calibrated===true;
    const dimStr=calibrated&&model.length_mm>0?`${model.length_mm} x ${model.width_mm} mm`:'Not calibrated';

    

    // Unique ID for the thumbnail canvas

    const thumbId = `thumb_${Date.now()}_${Math.random().toString(36).slice(2)}`;

    

    card.innerHTML = `

        <div class="gallery-card-thumb" id="${thumbId}">

            <div class="thumb-loading">

                <div class="thumb-spinner"></div>

                <span>Loading 3D preview...</span>

            </div>

            <div class="gallery-card-badge">${engineLabel}</div>

        </div>

        <div class="gallery-card-body">

            <h3 class="gallery-card-title" title="${modelName}">${modelName}</h3>

            <span class="gallery-card-date">${dateStr}</span>

            <div class="gallery-card-stats">

                <div class="gallery-card-stat">

                    <span>Faces</span>

                    <span>${facesStr}</span>

                </div>

                <div class="gallery-card-stat">

                    <span>Scale</span>

                    <span>${dimStr}</span>

                </div>

            </div>

        </div>

        <div class="gallery-card-actions">

            <button class="btn btn-primary btn-sm btn-open"><i class="fa-solid fa-folder-open"></i> Open</button>

            <button class="btn btn-danger btn-sm btn-delete"><i class="fa-solid fa-trash"></i></button>

        </div>

    `;

    

    // Attach event listeners

    const btnOpen = card.querySelector('.btn-open');

    btnOpen.addEventListener('click', () => {

        currentJobId = model.job_id;

        document.getElementById('navWorkspace').click(); // Switch to workspace view

        loadScanModel(model.job_id);

        fetchTopologyReport(model.job_id);

        fetchScanQualityReport(model.job_id);

        setupExportButtons(model.job_id);

        logConsole(`Opened model: ${model.name}`, 'system');

    });

    

    const btnDelete = card.querySelector('.btn-delete');
    btnDelete.setAttribute('aria-label', `Delete ${model.name || 'model'}`);

    btnDelete.addEventListener('click', async () => {

        if (confirm(`Are you sure you want to delete ${model.name}?`)) {

            try {

                btnDelete.disabled=true;
                const response=await fetch(`/api/models/${encodeURIComponent(model.id)}`, { method: 'DELETE' });
                if(!response.ok) throw new Error('Delete failed. Your model is still saved.');

                loadGallery(); // Refresh

            } catch (e) {

                btnDelete.disabled=false;
                logConsole("Could not delete model. Please try again.", "error");
                console.error("Failed to delete model:", e);

            }

        }

    });

    

    // Load 3D thumbnail after card is rendered

    requestAnimationFrame(() => {

        loadGalleryThumbnail(thumbId, model.job_id);

    });

    

    return card;

}



// =========================================================================

//  Gallery 3D Thumbnail Renderer

// =========================================================================

// Gallery thumbnails are rendered in scanner-improvements.js.

// AI Lab & Dataset Loaders

async function loadDatasetInventory() {

    try {

        const res = await fetch('/api/dataset/readiness');

        if (res.ok) {

            const data = await res.json();

            // Just silence the error if badges aren't present

        }

    } catch (e) { }

}



function initAiTrainingControls() {

    const btnTrain = document.getElementById('btnStartTraining');

    if (btnTrain) {

        btnTrain.addEventListener('click', async () => {

            btnTrain.disabled = true;

            btnTrain.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Starting...';

            try {

                await fetch(`/api/train/start`, { method: 'POST' });

                document.getElementById('trainingTracker').style.display = 'block';

            } catch (e) {}

        });

    }

}



// =========================================================================

//  Auth & Profile

// =========================================================================

function initAuth() {

    const tabLogin = document.getElementById('tabLogin');

    const tabRegister = document.getElementById('tabRegister');

    const loginForm = document.getElementById('loginForm');

    const registerForm = document.getElementById('registerForm');

    

    if(tabLogin && tabRegister) {

        tabLogin.addEventListener('click', () => {

            tabLogin.classList.add('active');

            tabRegister.classList.remove('active');

            loginForm.style.display = 'block';

            registerForm.style.display = 'none';

            clearAuthMessage();

        });

        tabRegister.addEventListener('click', () => {

            tabRegister.classList.add('active');

            tabLogin.classList.remove('active');

            registerForm.style.display = 'block';

            loginForm.style.display = 'none';

            clearAuthMessage();

        });

    }



    const btnSubmitLogin = document.getElementById('btnSubmitLogin');

    if(btnSubmitLogin) {

        btnSubmitLogin.addEventListener('click', async () => {

            clearAuthMessage();

            const user = document.getElementById('loginUser').value.trim();

            const pass = document.getElementById('loginPass').value;

            

            // Validation

            if (!user) {

                showAuthMessage('Please enter your username.', 'error');

                document.getElementById('loginUser').classList.add('input-error');

                return;

            }

            if (!pass) {

                showAuthMessage('Please enter your password.', 'error');

                document.getElementById('loginPass').classList.add('input-error');

                return;

            }

            

            btnSubmitLogin.disabled = true;

            btnSubmitLogin.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Logging in...';

            

            try {

                const res = await fetch('/api/auth/login', {

                    method: 'POST',

                    headers: {'Content-Type': 'application/json'},

                    body: JSON.stringify({username: user, password: pass})

                });

                if(res.ok) {

                    const data = await res.json();

                    authToken = data.token;

                    localStorage.setItem('authToken', authToken);

                    showAuthMessage('Login successful! Redirecting...', 'success');

                    setTimeout(() => {

                        checkProfile();

                        document.getElementById('navHome').click();

                    }, 800);

                } else {

                    const errData = await res.json().catch(() => ({}));

                    showAuthMessage(errData.detail || 'Invalid username or password.', 'error');

                }

            } catch(e) {

                showAuthMessage('Connection error. Please check your server.', 'error');

                console.error(e);

            }

            

            btnSubmitLogin.disabled = false;

            btnSubmitLogin.innerHTML = '<i class="fa-solid fa-right-to-bracket"></i> Login';

        });

    }

    

    const btnSubmitRegister = document.getElementById('btnSubmitRegister');

    if(btnSubmitRegister) {

        btnSubmitRegister.addEventListener('click', async () => {

            clearAuthMessage();

            clearInputErrors();

            

            const user = document.getElementById('regUser').value.trim();

            const email = document.getElementById('regEmail').value.trim();

            const pass = document.getElementById('regPass').value;

            const passConfirm = document.getElementById('regPassConfirm').value;

            

            // Client-side validation

            let hasError = false;

            

            if (!user || user.length < 3) {

                showAuthMessage('Username must be at least 3 characters.', 'error');

                document.getElementById('regUser').classList.add('input-error');

                hasError = true;

            }

            

            if (!email || !email.includes('@') || !email.includes('.')) {

                if (!hasError) showAuthMessage('Please enter a valid email address.', 'error');

                document.getElementById('regEmail').classList.add('input-error');

                hasError = true;

            }

            

            if (!pass || pass.length < 6) {

                if (!hasError) showAuthMessage('Password must be at least 6 characters.', 'error');

                document.getElementById('regPass').classList.add('input-error');

                hasError = true;

            }

            

            if (pass !== passConfirm) {

                if (!hasError) showAuthMessage('Passwords do not match.', 'error');

                document.getElementById('regPassConfirm').classList.add('input-error');

                hasError = true;

            }

            

            if (hasError) return;

            

            btnSubmitRegister.disabled = true;

            btnSubmitRegister.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Creating account...';

            

            try {

                const res = await fetch('/api/auth/register', {

                    method: 'POST',

                    headers: {'Content-Type': 'application/json'},

                    body: JSON.stringify({username: user, email: email, password: pass})

                });

                if(res.ok) {

                    showAuthMessage('Account created successfully! Please login.', 'success');

                    // Clear form fields

                    document.getElementById('regUser').value = '';

                    document.getElementById('regEmail').value = '';

                    document.getElementById('regPass').value = '';

                    document.getElementById('regPassConfirm').value = '';

                    // Auto-switch to login tab after a delay

                    setTimeout(() => {

                        if (tabLogin) tabLogin.click();

                    }, 1500);

                } else {

                    const errData = await res.json().catch(() => ({}));

                    showAuthMessage(errData.detail || 'Registration failed. Username or email may already exist.', 'error');

                }

            } catch(e) {

                showAuthMessage('Connection error. Please check your server.', 'error');

                console.error(e);

            }

            

            btnSubmitRegister.disabled = false;

            btnSubmitRegister.innerHTML = '<i class="fa-solid fa-user-plus"></i> Create Account';

        });

    }



    const btnLogout = document.getElementById('btnLogout');

    if(btnLogout) {

        btnLogout.addEventListener('click', async () => {

            if(authToken) {

                await fetch('/api/auth/logout', {

                    method: 'POST',

                    headers: {'Authorization': `Bearer ${authToken}`}

                });

            }

            authToken = null;

            localStorage.removeItem('authToken');

            currentUser = null;

            updateAuthUI();

            document.getElementById('navHome').click();

        });

    }



    const btnSaveSettings = document.getElementById('btnSaveSettings');

    if(btnSaveSettings) {

        btnSaveSettings.addEventListener('click', async () => {

            if(!authToken) return;

            const theme = document.getElementById('prefTheme').value;

            const engine = document.getElementById('prefEngine').value;

            const notifs = document.getElementById('prefNotifs').value === "1" ? 1 : 0;

            

            try {

                const res = await fetch('/api/user/settings', {

                    method: 'POST',

                    headers: {

                        'Content-Type': 'application/json',

                        'Authorization': `Bearer ${authToken}`

                    },

                    body: JSON.stringify({theme: theme, default_engine: engine, notifications: notifs})

                });

                if(res.ok) {

                    showAuthMessage('Settings saved!', 'success');

                }

            } catch(e) {}

        });

    }



    // Remove focus-error on input focus

    document.querySelectorAll('.auth-card .input-text').forEach(input => {

        input.addEventListener('focus', () => {

            input.classList.remove('input-error');

        });

    });



    if(authToken) {

        checkProfile();

    } else {

        updateAuthUI();

        const navHome = document.getElementById('navHome');

        if(navHome) navHome.click();

    }

}



// Auth message helpers

function showAuthMessage(msg, type) {

    const el = document.getElementById('authMessage');

    if (!el) return;

    const icon = type === 'error'

        ? '<i class="fa-solid fa-circle-exclamation"></i>'

        : '<i class="fa-solid fa-circle-check"></i>';

    el.innerHTML = `${icon} ${msg}`;

    el.className = `auth-message ${type}`;

}



function clearAuthMessage() {

    const el = document.getElementById('authMessage');

    if (el) {

        el.innerHTML = '';

        el.className = 'auth-message';

    }

}



function clearInputErrors() {

    document.querySelectorAll('.auth-card .input-error').forEach(el => {

        el.classList.remove('input-error');

    });

}



async function checkProfile() {

    try {

        const res = await fetch('/api/user/profile', {

            headers: {'Authorization': `Bearer ${authToken}`}

        });

        if(res.ok) {

            const data = await res.json();

            currentUser = data.user;

            document.getElementById('profUsername').innerText = currentUser.username;

            document.getElementById('profEmail').innerText = currentUser.email;

            

            if(data.settings) {

                document.getElementById('prefTheme').value = data.settings.theme;

                document.getElementById('prefEngine').value = data.settings.default_engine;

                document.getElementById('prefNotifs').value = data.settings.notifications ? "1" : "0";

            }

            updateAuthUI();

        } else {

            authToken = null;

            localStorage.removeItem('authToken');

            updateAuthUI();

        }

    } catch(e) { console.error(e); }

}



function updateAuthUI() {

    const navProfile = document.getElementById('navProfile');

    const navAuth = document.getElementById('navAuth');

    const navAdmin = document.getElementById('navAdmin');

    if(currentUser) {

        if(navProfile) navProfile.style.display = 'inline-flex';

        if(navAuth) navAuth.style.display = 'none';

        if(navAdmin && currentUser.is_admin === 1) navAdmin.style.display = 'inline-flex';

        else if (navAdmin) navAdmin.style.display = 'none';

    } else {

        if(navProfile) navProfile.style.display = 'none';

        if(navAuth) navAuth.style.display = 'inline-flex';

        if(navAdmin) navAdmin.style.display = 'none';

    }

}



// =========================================================================

//  Admin Dashboard

// =========================================================================

async function loadAdminDashboard() {

    if (!authToken || !currentUser || currentUser.is_admin !== 1) return;

    

    // Load Stats

    try {

        const resStats = await fetch('/api/admin/stats', {

            headers: {'Authorization': `Bearer ${authToken}`}

        });

        if (resStats.ok) {

            const data = await resStats.json();

            document.getElementById('adminTotalUsers').innerText = data.total_users || 0;

            document.getElementById('adminTotalModels').innerText = data.total_models || 0;

        }

    } catch(e) { console.error("Admin stats fetch failed", e); }

    

    // Load Users

    try {

        const resUsers = await fetch('/api/admin/users', {

            headers: {'Authorization': `Bearer ${authToken}`}

        });

        if (resUsers.ok) {

            const data = await resUsers.json();

            const tbody = document.getElementById('adminUsersTableBody');

            tbody.innerHTML = '';

            

            if (data.users && data.users.length > 0) {

                data.users.forEach(u => {

                    const tr = document.createElement('tr');

                    const roleBadge = u.is_admin === 1 

                        ? `<span class="role-badge admin">ADMIN</span>` 

                        : `<span class="role-badge user">USER</span>`;

                    const joinedDate = new Date(u.created_at * 1000).toLocaleDateString();

                    

                    tr.innerHTML = `

                        <td>${u.id}</td>

                        <td><strong>${u.username}</strong></td>

                        <td>${u.email}</td>

                        <td>${roleBadge}</td>

                        <td>${joinedDate}</td>

                    `;

                    tbody.appendChild(tr);

                });

            } else {

                tbody.innerHTML = '<tr><td colspan="5">No users found.</td></tr>';

            }

        }

    } catch(e) { console.error("Admin users fetch failed", e); }

}



const btnAdminRefresh = document.getElementById('btnAdminRefresh');

if(btnAdminRefresh) btnAdminRefresh.addEventListener('click', loadAdminDashboard);



// =========================================================================

//  Home Animation

// =========================================================================

let homeScene, homeCamera, homeRenderer, homeMesh;

function initHomeAnimation() {

    if (document.getElementById("scannerHomeFrame")) return; // The assembly scene owns its renderer and scroll timeline.

    const container = document.getElementById('homeCanvasContainer');

    if(!container) return;

    

    homeScene = new THREE.Scene();

    homeCamera = new THREE.PerspectiveCamera(45, container.clientWidth / (container.clientHeight || 400), 0.1, 100);

    homeCamera.position.z = 5;

    

    homeRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });

    // Handle hidden container height

    let cWidth = container.clientWidth || 600;

    let cHeight = container.clientHeight || 400;

    homeRenderer.setSize(cWidth, cHeight);

    container.appendChild(homeRenderer.domElement);

    

    const geometry = new THREE.IcosahedronGeometry(1.5, 2);

    const material = new THREE.MeshStandardMaterial({ 

        color: 0xc8a486, 

        wireframe: true,

        roughness: 0.1,

        metalness: 0.8

    });

    

    homeMesh = new THREE.Mesh(geometry, material);

    homeScene.add(homeMesh);

    

    const light1 = new THREE.PointLight(0x8faeb0, 2);

    light1.position.set(5, 5, 5);

    homeScene.add(light1);

    

    const light2 = new THREE.PointLight(0xc8a486, 2);

    light2.position.set(-5, -5, -5);

    homeScene.add(light2);

    

    function animateHome() {

        requestAnimationFrame(animateHome);

        if(container.offsetParent === null) return; // Don't render if hidden

        

        homeMesh.rotation.x += 0.005;

        homeMesh.rotation.y += 0.01;

        

        const time = Date.now() * 0.001;

        homeMesh.scale.x = 1 + Math.sin(time) * 0.1;

        homeMesh.scale.y = 1 + Math.cos(time) * 0.1;

        

        homeRenderer.render(homeScene, homeCamera);

    }

    animateHome();

    

    window.addEventListener('resize', () => {

        if(container.clientWidth > 0 && container.clientHeight > 0) {

            homeCamera.aspect = container.clientWidth / container.clientHeight;

            homeCamera.updateProjectionMatrix();

            homeRenderer.setSize(container.clientWidth, container.clientHeight);

        }

    });



    const homeUpload = document.getElementById('homePreviewUpload');

    if(homeUpload) {

        homeUpload.addEventListener('change', (e) => {

            if(e.target.files.length > 0) {

                const reader = new FileReader();

                reader.onload = function(event) {

                    const textureLoader = new THREE.TextureLoader();

                    textureLoader.load(event.target.result, (tex) => {

                        homeMesh.material.wireframe = false;

                        homeMesh.material.map = tex;

                        homeMesh.material.color.set(0xffffff);

                        homeMesh.material.needsUpdate = true;

                    });

                }

                reader.readAsDataURL(e.target.files[0]);

                

                const wsInput = document.getElementById('fileInput');

                if(wsInput) {

                    wsInput.files = e.target.files;

                    const aiBtn = document.getElementById('modeAiDepthBtn');

                    if(aiBtn) aiBtn.click();

                    handleFiles(Array.from(e.target.files));

                }

            }

        });

    }

}

