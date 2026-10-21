import * as THREE from 'three';

// An illustrative turntable capture; no camera or hardware connection is used.
export function createScanDemo(assembly, platter, requestRender, resumeMotion) {
  const object = new THREE.Group(); object.position.y = .145; platter.add(object);
  const profile = [[.22,0],[.29,.03],[.36,.15],[.42,.4],[.39,.65],[.28,.9],[.18,1.08],[.18,1.32],[.23,1.4],[.23,1.45],[.17,1.45],[.14,1.32],[.14,1.14],[.12,1.08]];
  const geometry = new THREE.LatheGeometry(profile.map(([r,y]) => new THREE.Vector2(r,y)),96);
  const ceramic = new THREE.MeshStandardMaterial({color:'#cda582',roughness:.36,metalness:.12,side:THREE.DoubleSide});
  const vase = new THREE.Mesh(geometry,ceramic); vase.castShadow=true; vase.receiveShadow=true; object.add(vase);
  const inlayPoints=profile.slice(1,9).map(([r,y])=>new THREE.Vector3(r*.22,y,r*.98));
  const inlay=new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(inlayPoints),64,.016,8,false),new THREE.MeshStandardMaterial({color:'#285d60',roughness:.4}));vase.add(inlay);
  const base = new THREE.Mesh(new THREE.CylinderGeometry(.22,.22,.025,64),ceramic); base.position.y=.0125;object.add(base);
  // A real three-dimensional surface sample, revealed progressively by azimuth.
  const vertices=[];
  for(let a=0;a<180;a++) for(let j=0;j<64;j++) {
    const h=j/63*1.44;let k=0;while(k<8&&profile[k+1][1]<h)k++;
    const [r0,y0]=profile[k], [r1,y1]=profile[k+1];const r=THREE.MathUtils.lerp(r0,r1,THREE.MathUtils.clamp((h-y0)/(y1-y0),0,1));
    vertices.push(Math.cos(a/180*Math.PI*2)*r*1.015,h,Math.sin(a/180*Math.PI*2)*r*1.015);
  }
  const cloudGeometry=new THREE.BufferGeometry(); cloudGeometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));
  const cloud=new THREE.Points(cloudGeometry,new THREE.PointsMaterial({color:'#92e3d5',size:.022,transparent:true,opacity:.9}));object.add(cloud);cloud.visible=false;
  const beamGeometry=new THREE.BufferGeometry();beamGeometry.setAttribute('position',new THREE.Float32BufferAttribute(new Float32Array(18),3));
  const beam=new THREE.LineSegments(beamGeometry,new THREE.LineBasicMaterial({color:'#8edacb',transparent:true,opacity:.55,depthWrite:false}));assembly.add(beam);beam.visible=false;
  const sweep=new THREE.Mesh(new THREE.TorusGeometry(.45,.009,6,100),new THREE.MeshBasicMaterial({color:'#a5f5df',transparent:true,opacity:.9}));sweep.rotation.x=Math.PI/2;assembly.add(sweep);sweep.visible=false;
  const panel=document.querySelector('#scan-demo-panel'),button=document.querySelector('#scan-demo-start');
  let start=null,lastPercent=-1,lastTime=0;
  function stop(){start=null;panel.hidden=true;button.textContent='Watch a sample scan ↗';cloud.visible=beam.visible=sweep.visible=false;vase.visible=base.visible=true;requestRender();}
  button.addEventListener('click',()=>{if(!button.disabled){start=lastTime;lastPercent=-1;panel.hidden=false;button.textContent='Replay sample scan ↻';window.scrollTo({top:0,behavior:'smooth'});resumeMotion();requestRender();}});
  document.querySelector('#scan-demo-close').addEventListener('click',stop);
  return {
    angle(time){return start===null?time*.28:start*.28+Math.min(time-start,12)/12*Math.PI*2;},
    update(time,show){
      lastTime=time;object.visible=show;
      if(!show){if(start!==null)stop();return;}
      if(start===null)return;
      const seconds=Math.max(0,time-start),percent=Math.min(100,Math.floor(seconds/18*100));
      const capturing=seconds<12,rebuilding=seconds>=12&&seconds<18;
      beam.visible=sweep.visible=capturing;cloud.visible=rebuilding;vase.visible=base.visible=!rebuilding;
      cloudGeometry.setDrawRange(0,Math.floor(vertices.length/3*THREE.MathUtils.clamp((seconds-12)/6,0,1)));
      const height=.18+(Math.sin(seconds*2)*.5+.5)*1.4;sweep.position.set(0,-.785+height,0);
      const positions=beamGeometry.attributes.position;
      [[-.4,height,0],[.4,height,0],[0,height+.15,.3]].forEach((p,i)=>{positions.setXYZ(i*2,5.7,2.65,0);positions.setXYZ(i*2+1,p[0],-.785+p[1],p[2]);});positions.needsUpdate=true;
      if(percent!==lastPercent){
        lastPercent=percent;document.querySelector('#scan-demo-progress').style.width=`${percent}%`;
        document.querySelector('#scan-demo-count').textContent=`${percent}%`;
        document.querySelector('#scan-demo-status').textContent=capturing?`01 / Capture · ${Math.min(36,Math.floor(seconds/12*36))} of 36 views`:rebuilding?'02 / Reconstruct · views become a 3D surface':'03 / Complete · sample 3D model ready';
        document.querySelector('#scan-demo-explain').textContent=capturing?'The object rotates so the camera can capture every side.':rebuilding?'Matching details across views builds a point cloud and surface.':'Inspect the resulting model, then save or export it in the workspace.';
      }
    }
  };
}
