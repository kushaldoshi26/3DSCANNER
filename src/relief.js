import * as THREE from 'three';

// Closed, millimetre-scale height field: front, flat back, and four stitched walls.
// Brightness represents height, not estimated scene depth.
export function buildRelief(pixels, columns, rows, width, depth, invert=false, options={}) {
  if(columns<2||rows<2||pixels.length!==columns*rows*4||width<=0||depth<=0)throw new Error('Invalid relief dimensions.');
  const height=width*(rows-1)/(columns-1), base=2, count=columns*rows;
  let heights=new Float32Array(count);
  const contrast=options.contrast??1;
  for(let i=0;i<count;i++){
    const p=i*4,alpha=pixels[p+3]/255;
    let l=(.2126*pixels[p]+.7152*pixels[p+1]+.0722*pixels[p+2])/255;
    l=Math.max(0,Math.min(1,(l-.5)*contrast+.5));
    heights[i]=base+depth*(invert?1-l:l)*alpha;
  }
  // Range-weighted filtering suppresses photo noise while preserving sharp edges.
  for(let pass=0;pass<(options.smoothing??0);pass++){
    const next=new Float32Array(count),range=Math.max(.01,depth*.045);
    for(let y=0;y<rows;y++)for(let x=0;x<columns;x++){
      const i=y*columns+x,centre=heights[i];let sum=0,weights=0;
      for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
        const v=heights[Math.max(0,Math.min(rows-1,y+dy))*columns+Math.max(0,Math.min(columns-1,x+dx))];
        const difference=(v-centre)/range,weight=(dx===0?2:1)*(dy===0?2:1)/(1+difference*difference);
        sum+=v*weight;weights+=weight;
      }
      next[i]=sum/weights;
    }
    heights=next;
  }
  const edge=[];
  for(let x=0;x<columns;x++)edge.push(x);
  for(let y=1;y<rows;y++)edge.push(y*columns+columns-1);
  for(let x=columns-2;x>=0;x--)edge.push((rows-1)*columns+x);
  for(let y=rows-2;y>0;y--)edge.push(y*columns);
  // A flat back needs a boundary fan, not a second dense image grid.
  const boundary=edge.length,vertexCount=count+boundary+1+boundary*4;
  const positions=new Float32Array(vertexCount*3),normals=new Float32Array(vertexCount*3);
  const indices=new Uint32Array(((columns-1)*(rows-1)*2+boundary*3)*3);
  const dx=width/(columns-1),dy=height/(rows-1);
  const point=(i,x,y,z,nx,ny,nz)=>{positions.set([x,y,z],i*3);normals.set([nx,ny,nz],i*3)};
  for(let y=0;y<rows;y++)for(let x=0;x<columns;x++){
    const i=y*columns+x;
    const sx=(heights[y*columns+Math.min(columns-1,x+1)]-heights[y*columns+Math.max(0,x-1)])/((x===0||x===columns-1?1:2)*dx);
    const sy=(heights[Math.max(0,y-1)*columns+x]-heights[Math.min(rows-1,y+1)*columns+x])/((y===0||y===rows-1?1:2)*dy);
    const length=Math.hypot(sx,sy,1);
    point(i,x*dx-width/2,height/2-y*dy,heights[i],-sx/length,-sy/length,1/length);
  }
  const centre=count+boundary;point(centre,0,0,0,0,0,-1);
  for(let k=0;k<boundary;k++){const i=edge[k];point(count+k,positions[i*3],positions[i*3+1],0,0,0,-1)}
  let offset=0;const triangle=(a,b,c)=>{indices[offset++]=a;indices[offset++]=b;indices[offset++]=c};
  for(let y=0;y<rows-1;y++)for(let x=0;x<columns-1;x++){
    const a=y*columns+x,b=a+1,c=a+columns,d=c+1;triangle(a,c,b);triangle(b,c,d);
  }
  for(let k=0;k<boundary;k++){
    const next=(k+1)%boundary,a=edge[k],b=edge[next],v=count+boundary+1+k*4;
    triangle(centre,count+k,count+next);
    const ax=positions[a*3],ay=positions[a*3+1],bx=positions[b*3],by=positions[b*3+1];
    const length=Math.hypot(bx-ax,by-ay),nx=-(by-ay)/length,ny=(bx-ax)/length;
    point(v,ax,ay,heights[a],nx,ny,0);point(v+1,bx,by,heights[b],nx,ny,0);
    point(v+2,ax,ay,0,nx,ny,0);point(v+3,bx,by,0,nx,ny,0);
    triangle(v,v+1,v+2);triangle(v+1,v+3,v+2);
  }
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(positions,3));g.setAttribute('normal',new THREE.BufferAttribute(normals,3));g.setIndex(new THREE.BufferAttribute(indices,1));
  return g;
}
