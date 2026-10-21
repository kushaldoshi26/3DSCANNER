// The API returns binary GLB only, so imported models cannot fetch remote buffers.
export function validateGLB(buffer) {
 const bytes=new DataView(buffer);
 if(bytes.byteLength<20||bytes.getUint32(0,true)!==0x46546c67||bytes.getUint32(4,true)!==2||bytes.getUint32(8,true)!==bytes.byteLength)throw new Error('Choose a valid GLB 2.0 file.');
 const length=bytes.getUint32(12,true);
 if(length>bytes.byteLength-20||bytes.getUint32(16,true)!==0x4e4f534a)throw new Error('The GLB file is damaged.');
 const json=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,20,length)));
 for(const resource of [...(json.buffers||[]),...(json.images||[])])if(resource.uri&&!resource.uri.startsWith('data:'))throw new Error('Use a self-contained GLB with embedded resources.');
 return json;
}
