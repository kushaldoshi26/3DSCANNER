"""Sequential Hunyuan3D inference; preserve geometry and report real failures."""
from pathlib import Path
import gc,os,json
import torch
import trimesh
from PIL import Image

SHAPE_PIPELINE=None
PAINT_PIPELINE=None

def _release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()

def _native_imports():
    import sys
    os.environ.setdefault('HY3DGEN_MODELS','D:/3dSCANNER/engines/models')
    source=os.environ.get('SCANNER_HUNYUAN_SOURCE','D:/3dSCANNER/engines/Hunyuan3D-2-main')
    if (Path(source)/'hy3dgen').is_dir() and source not in sys.path:sys.path.insert(0,source)
    packages=str(Path(__file__).resolve().parent.parent/'engines/python-packages')
    if packages not in sys.path:sys.path.insert(0,packages)

def generate_hunyuan3d_mesh(image_path,output_stl_path,output_glb_path=None,paint_texture=True,quality='high',seed=42,paint_face_limit=100000,progress_cb=None):
    global SHAPE_PIPELINE,PAINT_PIPELINE
    _native_imports()
    def progress(stage,percent,message):
        if progress_cb:progress_cb(stage,percent,message)
    if os.name=='nt':
        import ctypes
        class Status(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
        status=Status();status.length=ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):raise ctypes.WinError()
        if status.avail_page<12*1024**3:raise RuntimeError('Insufficient system memory for Hunyuan; save and close memory-heavy apps before retrying.')
    if not torch.cuda.is_available():raise RuntimeError('Hunyuan3D requires the configured CUDA GPU.')
    if torch.cuda.mem_get_info()[0]<8*1024**3:raise RuntimeError('Insufficient free GPU memory for Hunyuan; another GPU workload may be active.')
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    settings={'fast':(20,256),'high':(30,384),'ultra':(50,512)}
    steps,resolution=settings.get(quality,settings['high'])
    out=Path(output_stl_path);out.parent.mkdir(parents=True,exist_ok=True)
    glb=Path(output_glb_path) if output_glb_path else out.with_suffix('.glb')
    glb.parent.mkdir(parents=True,exist_ok=True)
    progress('Loading shape model',5,'Loading local Hunyuan shape weights...')
    image=Image.open(image_path).convert('RGBA')
    try:
        SHAPE_PIPELINE=Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2',use_safetensors=True)
        # Shape and texture execute sequentially; shape fits the configured 16GB GPU.
        mesh=SHAPE_PIPELINE(image=image,num_inference_steps=steps,octree_resolution=resolution,
                            callback=lambda i,t,o:progress('Generating shape',10+int(35*(i+1)/steps),f'Shape sampling {i+1}/{steps}'),callback_steps=1,num_chunks=8000,generator=torch.Generator(device='cuda').manual_seed(seed))[0]
        if not isinstance(mesh,trimesh.Trimesh) or mesh.is_empty:raise RuntimeError('Shape engine returned no valid mesh')
    finally:
        SHAPE_PIPELINE=None;_release()
    # No destructive smoothing, topology reduction or second-network deformation.
    mesh.remove_unreferenced_vertices();mesh.fix_normals()
    print(f'Hunyuan shape ready: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces',flush=True)
    metadata={'engine_used':'hunyuan3d','weights_finetuned':False,'unseen_surfaces':'inferred','source_view':[0,0,1],'seed':seed,'sampling_steps':steps,'octree_resolution':resolution}
    if paint_texture:
        full_shape=mesh.copy()
        progress('Preparing texture preview',55,'Preserving the full shape and preparing bounded texture generation...')
        mesh.export(str(glb.with_name(glb.stem+'_shape_full.glb')))
        mesh.export(str(out.with_name(out.stem+'_shape_full.stl')))
        if paint_face_limit is not None and len(mesh.faces)>paint_face_limit:
            print(f'Texture preview LOD: {len(mesh.faces)} -> {paint_face_limit} faces; full shape preserved separately',flush=True)
            mesh=mesh.simplify_quadric_decimation(face_count=paint_face_limit)
        try:
            progress('Loading texture models',60,'Loading texture models with GPU offload...')
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            from hy3dgen.texgen.pipelines import Hunyuan3DTexGenConfig
            model_root=Path(os.environ['HY3DGEN_MODELS'])/'tencent/Hunyuan3D-2'
            config=Hunyuan3DTexGenConfig(str(model_root/'hunyuan3d-delight-v2-0'),str(model_root/'hunyuan3d-paint-v2-0-turbo'),'hunyuan3d-paint-v2-0-turbo')
            config.device='cpu'
            PAINT_PIPELINE=Hunyuan3DPaintPipeline(config)
            PAINT_PIPELINE.enable_model_cpu_offload()
            progress('Generating texture',65,'Generating inferred multi-view appearance...')
            painted=PAINT_PIPELINE(mesh,image=image)
            mesh=painted[0] if isinstance(painted,(list,tuple)) else painted
            if not hasattr(mesh,'export'):raise RuntimeError('Paint stage returned no exportable mesh')
        except Exception as error:
            raise RuntimeError('Hunyuan texture stage failed; no untextured success was reported: '+str(error)) from error
        finally:
            PAINT_PIPELINE=None;_release()
    if paint_texture:
        from pipeline.appearance_transfer import transfer_preview_appearance
        progress('Restoring full geometry',80,'Transferring appearance to the preserved full-resolution shape...')
        mesh,transfer=transfer_preview_appearance(full_shape,mesh,progress=lambda ratio:progress('Restoring full geometry',80+int(15*ratio),'Preserving full geometry while transferring appearance...'))
        metadata.update(transfer)
    glb.with_suffix('.generation.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    progress('Exporting model',98,'Writing full-resolution GLB and STL...')
    mesh.export(str(glb));mesh.export(str(out))
    if not glb.exists() or not out.exists():raise RuntimeError('Generation export is incomplete')
    return str(out)
