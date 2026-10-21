import { buildRelief } from './relief.js';
self.onmessage=({data})=>{
 try{
  const {pixels,columns,rows,width,depth,invert,options}=data;
  const g=buildRelief(pixels,columns,rows,width,depth,invert,options);
  const position=g.attributes.position.array,normal=g.attributes.normal.array,index=g.index.array;
  self.postMessage({position,normal,index},[position.buffer,normal.buffer,index.buffer]);
 }catch(error){self.postMessage({error:error.message})}
};
