const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
function element(){return {innerHTML:'',className:'',querySelector(){return {addEventListener(){},setAttribute(){}};}};}
const context=vm.createContext({document:{addEventListener(){},getElementById(){return null;},createElement:element},window:{addEventListener(){}},localStorage:{getItem(){return null;}},requestAnimationFrame(){},console});
vm.runInContext(fs.readFileSync('static/app.js','utf8'),context);
const model={name:'<img src=x onerror=bad()>',engine_label:'TripoSR',created_at:0,faces:100,length_mm:2,width_mm:3,job_id:'test',id:1};
context.model=model;
let card=vm.runInContext('createGalleryCard(model)',context);
assert.ok(card.innerHTML.includes('Not calibrated'));
assert.ok(!card.innerHTML.includes('2 x 3 mm'));
assert.ok(card.innerHTML.includes('&lt;img'));
model.extra_json=JSON.stringify({scale_calibrated:true});
card=vm.runInContext('createGalleryCard(model)',context);assert.ok(card.innerHTML.includes('2 x 3 mm'));
model.extra_json='{broken';card=vm.runInContext('createGalleryCard(model)',context);assert.ok(card.innerHTML.includes('Not calibrated'));
console.log('Gallery scale requires explicit calibration; malformed metadata and model names are handled safely.');
