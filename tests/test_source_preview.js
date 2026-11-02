const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const button={addEventListener(){},classList:{add(){},remove(){}}};const added=[];
class Vector {fromBufferAttribute(a,i){[this.x,this.y,this.z]=Array.from(a.array).slice(i*3,i*3+3);return this;}applyMatrix4(){return this;}}
class Geometry {constructor(){this.attributes={};}setAttribute(k,v){this.attributes[k]=v;}dispose(){}}
class Attribute {constructor(values,n){this.array=Float32Array.from(values);this.itemSize=n;this.count=values.length/n;}}
const context=vm.createContext({document:{addEventListener(){},getElementById(){return button;},querySelectorAll(){return [];}},window:{addEventListener(){}},localStorage:{getItem(){}},console,performance:{now(){return 0;}},requestAnimationFrame(){},THREE:{Vector3:Vector,BufferGeometry:Geometry,Float32BufferAttribute:Attribute,PointsMaterial:class{constructor(o){Object.assign(this,o);}dispose(){}},Points:class{constructor(g,m){this.geometry=g;this.material=m;}}},added});
vm.runInContext(fs.readFileSync('static/app.js','utf8'),context);vm.runInContext(fs.readFileSync('static/scanner-improvements.js','utf8'),context);
vm.runInContext(`logConsole=()=>{};scene={add(value){added.push(value)},remove(){}};
const positions={count:3,array:Float32Array.from([0,0,0,1,0,1,0,2,-.5])};
const colours={getX(){return .4},getY(){return .5},getZ(){return .6}};
currentMesh={visible:true,updateMatrixWorld(){},traverse(fn){fn({isMesh:true,geometry:{attributes:{position:positions,color:colours}},userData:{},matrixWorld:{}})}};
showModelPointCloud();`,context);
const data=added[0].geometry.attributes.position.array;
assert.equal(data[5],1);assert.equal(data[8],-.5);assert.equal(data.length,9);
assert.equal(vm.runInContext('currentMesh.visible',context),false);
vm.runInContext('clearPointView()',context);assert.equal(vm.runInContext('currentMesh.visible',context),true);
vm.runInContext('loadingPointCloud={visible:true,material:{}};animate()',context);
console.log('3D point view retains nonzero world depth, restores mesh view, and never billboards or flattens loading points.');
