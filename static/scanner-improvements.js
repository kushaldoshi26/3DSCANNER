// TripoSR exports sRGB vertex samples; Three.js shades in linear space.
function prepareScannerAppearance(root) {
    root.traverse(child=>{
        if(!child.isMesh)return;
        const attr=child.geometry.attributes.color;
        if(attr && !child.geometry.userData.scannerLinearColors){
            const channels=attr.itemSize, converted=new Float32Array(attr.count*channels);
            const byte=attr.array instanceof Uint8Array;
            for(let i=0;i<attr.count;i++)for(let c=0;c<channels;c++){
                const value=attr.array[i*channels+c]/(byte?255:1);
                converted[i*channels+c]=c===3?value:(value<=.04045?value/12.92:Math.pow((value+.055)/1.055,2.4));
            }
            child.geometry.setAttribute('color',new THREE.BufferAttribute(converted,channels));
            child.geometry.userData.scannerLinearColors=true;
        }
        for(const mat of [].concat(child.material)){if('metalness' in mat)mat.metalness=0;if('roughness' in mat)mat.roughness=.85;}
    });
}
// Shared controls and lightweight gallery snapshots.
let previewGeneration = 0;
createLoadingPointCloud = function () {
    ++previewGeneration;
    clearPointView(false);
    if (loadingPointCloud) {
        scene.remove(loadingPointCloud);
        loadingPointCloud.geometry.dispose(); loadingPointCloud.material.dispose();
    }
    loadingPointCloud = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({size:.012, vertexColors:true, transparent:true, opacity:.85}));
    scene.add(loadingPointCloud);
};

let thumbnailQueue = Promise.resolve();
loadGalleryThumbnail = function(containerId, jobId) {
    thumbnailQueue = thumbnailQueue.catch(()=>{}).then(()=>new Promise(resolve => {
        const container=document.getElementById(containerId);
        if(!container) {resolve(); return;}
        const miniScene=new THREE.Scene(); miniScene.background=new THREE.Color(0x102630);
        miniScene.add(new THREE.HemisphereLight(0xffffff,0x64747a,1.1));
        const light=new THREE.DirectionalLight(0xffffff,.8);light.position.set(2,4,-3);miniScene.add(light);
        new THREE.GLTFLoader().load(`/api/scan/download/${encodeURIComponent(jobId)}/glb`, gltf=>{
            let renderer;
            try {
                const model=gltf.scene;prepareScannerAppearance(model);
                model.traverse(child=>{if(child.isMesh)for(const mat of [].concat(child.material)){if('metalness' in mat)mat.metalness=.05;if('roughness' in mat)mat.roughness=.7;}});
                const box=new THREE.Box3().setFromObject(model), center=box.getCenter(new THREE.Vector3()), size=box.getSize(new THREE.Vector3());
                const factor=2/(Math.max(size.x,size.y,size.z)||1);
                model.scale.setScalar(factor);model.position.copy(center.multiplyScalar(-factor));miniScene.add(model);
                const cam=new THREE.PerspectiveCamera(38,1.6,.01,100);cam.position.set(2,1.1,-3.7);if(model.userData.source_view)cam.position.fromArray(model.userData.source_view).multiplyScalar(4);cam.lookAt(0,0,0);
                renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.outputEncoding=THREE.sRGBEncoding;renderer.setSize(480,300);renderer.render(miniScene,cam);
                const img=new Image();img.alt='3D preview of saved model';img.src=renderer.domElement.toDataURL('image/png');
                container.querySelector('.thumb-loading')?.remove();container.prepend(img);
            } catch(error) { const loading=container.querySelector('.thumb-loading');if(loading)loading.textContent='Preview unavailable — open model'; }
            finally {
                gltf.scene.traverse(child=>{if(child.isMesh){child.geometry.dispose();for(const mat of [].concat(child.material)){mat.map?.dispose();mat.dispose();}}});
                renderer?.dispose();renderer?.forceContextLoss();resolve();
            }
        },undefined,()=>{const loading=container.querySelector('.thumb-loading');if(loading)loading.textContent='Model file unavailable';resolve();});
    }));
};

document.addEventListener('DOMContentLoaded',()=>{
    document.getElementById('modeAiDepthBtn').onclick=()=>{
        currentMode='ai_depth'; selectedFiles=[]; clearUnifiedPreview();
        const input=document.getElementById('fileInput');input.removeAttribute('multiple');input.value='';
        document.getElementById('filePreviewBox').style.display='none';
        document.getElementById('scannerConfigBar').style.display='none';
        document.getElementById('dropzoneTitle').textContent='Drop your image here';
        document.getElementById('dropzoneDesc').textContent='Upload a photo to generate a 3D model';
        document.querySelectorAll('.mode-btn').forEach(b=>b.classList.toggle('active',b.id==='modeAiDepthBtn'));
    };
    for(const id of ['modeTurntableBtn','modeMultiBtn','modeDiffusionBtn']) document.getElementById(id).addEventListener('click',()=>document.getElementById('modeAiDepthBtn').classList.remove('active'));
    const zoom=factor=>{if(!camera||!controls)return;const delta=camera.position.clone().sub(controls.target);delta.multiplyScalar(factor);delta.clampLength(.35,40);camera.position.copy(controls.target).add(delta);controls.update();};
    document.getElementById('zoomIn').onclick=()=>zoom(.8);
    document.getElementById('zoomOut').onclick=()=>zoom(1.25);
    document.getElementById('flipUpright').onclick=()=>{
        if(!currentMesh)return;
        // Rotate about the displayed centre, including translated GLTF roots.
        currentMesh.position.y*=-1; currentMesh.position.z*=-1;
        currentMesh.rotateOnWorldAxis(new THREE.Vector3(1,0,0),Math.PI);
    };
    document.getElementById('btnResetCam').addEventListener('click',()=>{camera.position.set(2.4,1.3,-3.3);if(currentMesh?.userData.source_view)camera.position.fromArray(currentMesh.userData.source_view).multiplyScalar(3.3);controls.update();});
    const hint=document.getElementById('singleImageHint');
    for(const id of ['modeTurntableBtn','modeMultiBtn','modeAiDepthBtn','modeDiffusionBtn']) document.getElementById(id)?.addEventListener('click',()=>{hint.textContent=['turntable','multi'].includes(currentMode)?'Multiple views for a complete scan':'Single image only';});
    document.querySelectorAll('[data-oauth]').forEach(button=>button.onclick=async()=>{
        const status=await fetch('/api/auth/oauth/providers').then(r=>r.json()).catch(()=>({}));
        if(!status[button.dataset.oauth]){showAuthMessage('This provider needs app credentials configured on the server.','error');return;}
        location.assign(`/api/auth/oauth/${button.dataset.oauth}/start`);
    });
    const params=new URLSearchParams(location.search);
    if(params.has('oauth_error')){document.getElementById('navAuth').click();showAuthMessage('Social login could not finish. Please try again.','error');history.replaceState(null,'','/');}
    if(params.has('oauth_success')) fetch('/api/auth/oauth/session',{method:'POST'}).then(r=>r.ok?r.json():Promise.reject()).then(data=>{authToken=data.token;localStorage.setItem('authToken',authToken);checkProfile();history.replaceState(null,'','/');}).catch(()=>{document.getElementById('navAuth').click();showAuthMessage('Login expired. Please try again.','error');});
});

let modelPointCloud=null;
function clearPointView(showMesh=true) {
    if(modelPointCloud){scene.remove(modelPointCloud);modelPointCloud.geometry.dispose();modelPointCloud.material.dispose();modelPointCloud=null;}
    if(currentMesh&&showMesh)currentMesh.visible=true;
    document.getElementById('btnPointCloud')?.classList.remove('active');
}
function showModelPointCloud() {
    if(!currentMesh)return;
    if(modelPointCloud){clearPointView();return;}
    currentMesh.updateMatrixWorld(true);
    const positions=[],colors=[],point=new THREE.Vector3(),textureData=new Map();
    let total=0;currentMesh.traverse(child=>{if(child.isMesh)total+=child.geometry.attributes.position.count;});
    const step=Math.max(1,Math.ceil(total/45000));
    currentMesh.traverse(child=>{
        if(!child.isMesh)return;
        const geom=child.geometry,p=geom.attributes.position,c=geom.attributes.color,uv=geom.attributes.uv,map=child.userData.nativeTexture;
        let tex=null;
        if(map?.image&&uv){
            if(!textureData.has(map)){
                const canvas=document.createElement('canvas');canvas.width=map.image.width;canvas.height=map.image.height;
                const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(map.image,0,0);textureData.set(map,{data:ctx.getImageData(0,0,canvas.width,canvas.height).data,w:canvas.width,h:canvas.height});
            }
            tex=textureData.get(map);
        }
        for(let i=0;i<p.count;i+=step){
            point.fromBufferAttribute(p,i).applyMatrix4(child.matrixWorld);positions.push(point.x,point.y,point.z);
            if(tex){
                const x=Math.max(0,Math.min(tex.w-1,Math.floor(uv.getX(i)*tex.w))),v=map.flipY?1-uv.getY(i):uv.getY(i);
                const y=Math.max(0,Math.min(tex.h-1,Math.floor(v*tex.h))),offset=(y*tex.w+x)*4;
                for(let k=0;k<3;k++){const s=tex.data[offset+k]/255;colors.push(s<=.04045?s/12.92:Math.pow((s+.055)/1.055,2.4));}
            }else if(c)colors.push(c.getX(i),c.getY(i),c.getZ(i));
            else colors.push(.6,.45,.3);
        }
    });
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
    modelPointCloud=new THREE.Points(geometry,new THREE.PointsMaterial({size:.008,vertexColors:true}));scene.add(modelPointCloud);currentMesh.visible=false;
    document.getElementById('btnPointCloud').classList.add('active');
    logConsole('3D surface point cloud — orbit to inspect depth and unseen inferred surfaces.','system');
}

async function refreshSourcePointCloud(jobId) {
    const cloud=loadingPointCloud,version=previewGeneration;
    for(let attempt=0;attempt<240;attempt++){
        if(!cloud?.visible||cloud!==loadingPointCloud||version!==previewGeneration)return;
        try{
            const response=await fetch(`/api/scan/point-cloud/${encodeURIComponent(jobId)}`,{cache:'no-store'});
            if(response.ok){
                const data=await response.json();if(!cloud.visible||cloud!==loadingPointCloud)return;
                const pts=data.positions,colors=data.colors,low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
                for(let i=0;i<pts.length;i++) {const k=i%3;low[k]=Math.min(low[k],pts[i]);high[k]=Math.max(high[k],pts[i]);}
                const scale=2/Math.max(...high.map((x,k)=>x-low[k]),.001);
                for(let i=0;i<pts.length;i++){const k=i%3;pts[i]=(pts[i]-(low[k]+high[k])/2)*scale;const v=colors[i]/255;colors[i]=v<=.04045?v/12.92:Math.pow((v+.055)/1.055,2.4);}
                cloud.geometry.setAttribute('position',new THREE.Float32BufferAttribute(pts,3));cloud.geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
                cloud.quaternion.identity();
                const label=document.querySelector('#generatingOverlay small');if(label)label.textContent=data.kind==='reconstructed_surface'?'Reconstructed 3D surface points · orbit to inspect':'Estimated depth points · unseen surfaces are unknown';
                return;
            }
            if(![404,503].includes(response.status))return;
        }catch(error){}
        await new Promise(resolve=>setTimeout(resolve,1500));
    }
}

document.addEventListener('DOMContentLoaded',()=>{
    document.getElementById('btnPointCloud').onclick=showModelPointCloud;
    for(const id of ['btnSolid','btnWireframe','btnClay','btnTextured'])document.getElementById(id).addEventListener('click',()=>{if(id==='btnTextured'&&modelPointCloud)isTextured=false;clearPointView();},true);
    const updateDepth=()=>{
        const depth=document.getElementById('aiEngineSelect').value==='ai_depth',printable=document.getElementById('printableToggle');
        printable.disabled=depth;if(depth)printable.checked=false;
        printable.title=depth?'Depth mode produces an open visible surface. Use a 360° generator for printable geometry.':'';
    };
    document.getElementById('aiEngineSelect').addEventListener('change',updateDepth);updateDepth();
    fetch('/api/dataset/readiness').then(r=>r.json()).then(data=>{
        document.getElementById('trainingReadiness').textContent=`${(data.geometry_valid_npz ?? data.geometry_npz).toLocaleString()} geometry sets · ${data.photo_pairs} rendered RGB/depth pairs. ${data.custom_checkpoint.ready?'Depth checkpoint qualified.':'Depth checkpoint not qualified for real photos.'}`;
    }).catch(()=>{document.getElementById('trainingReadiness').textContent='Dataset status unavailable';});
});
