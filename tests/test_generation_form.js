const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');
const controls = {};
const document = {
    addEventListener() {},
    getElementById(id) { return controls[id] || null; },
    querySelectorAll() { return []; },
};
let submission;
const context = vm.createContext({document,console,FormData,Blob,URL,localStorage:{getItem(){return null;}},
    window:{addEventListener(){}},setInterval,clearInterval,
    fetch:async (url,options) => {submission={url,options};return {ok:false,json:async()=>({detail:'test captured'})};}});
vm.runInContext(fs.readFileSync('static/app.js','utf8'),context);
Object.assign(controls,{aiEngineSelect:{value:'triposr'},qualitySelect:{value:'ultra'},objectTypeSelect:{value:'hard_surface'},refineToggle:{checked:false},printableToggle:{checked:true},photoDetailToggle:{checked:true}});
vm.runInContext("selectedFiles=[new Blob(['test'])]; startLoadingAnimation=()=>{}; stopLoadingAnimation=()=>{}; resetStartButtons=()=>{}; logConsole=()=>{};",context);
(async()=>{
    await vm.runInContext('startPipeline()',context);
    assert.equal(submission.url,'/api/reconstruct');
    const form=submission.options.body;
    for(const [key,value] of Object.entries({engine:'triposr',quality:'ultra',object_type:'hard_surface',texture:'true',refine:'false',printable:'true',photo_detail:'true'}))assert.equal(form.get(key),value,key);
    assert.ok(form.get('file'));
    vm.runInContext(`
        const child = {isMesh:true,userData:{nativeTexture:{name:'native'}},geometry:{attributes:{}},material:{color:{set(){}},vertexColors:false}};
        currentMesh={traverse(fn){fn(child);}};hasVertexColors=false;hasTextureMaps=true;
        applySolidColorMode();
        if(child.material.map!==null)throw Error('Solid mode retained the texture');
        applyTexturedMode(true);
        if(child.material.map!==child.userData.nativeTexture)throw Error('Native texture was not restored');
    `,context);
    console.log('UI engine, quality, object type, refinement and printable settings reach multipart form correctly.');
    console.log('Native texture survives switching between shaded and textured views.');
    controls.aiEngineSelect.value='ai_depth';
    await vm.runInContext('startPipeline()',context);
    assert.equal(submission.url,'/api/scan/ai-depth');
    console.log('Explicit depth-preview selection reaches its dedicated endpoint.');
})().catch(error=>{console.error(error);process.exitCode=1;});
