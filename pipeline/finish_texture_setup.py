from pathlib import Path
import sys,os,json,time,subprocess,traceback
ROOT=Path('D:/3dSCANNER');out=ROOT/'data/software_checks/hunyuan_texture_setup';out.mkdir(parents=True,exist_ok=True)
tools=ROOT/'engines/toolchains';started=time.time();vs=tools/'VS2022/VC/Auxiliary/Build/vcvars64.bat';cuda=tools/'cuda-12.8'
def record(stage,**extra):
    p=out/'setup.tmp';p.write_text(json.dumps({'stage':stage,'elapsed_seconds':time.time()-started,'updated':time.time(),**extra},indent=2));os.replace(p,out/'setup.json')
try:
    while not(vs.exists() and (tools/'cuda-setup.json').exists()):
        record('waiting_for_compilers',msvc_ready=vs.exists(),cuda_ready=(tools/'cuda-setup.json').exists())
        if time.time()-started>3*3600:raise RuntimeError('Compiler setup timeout; inspect official installer logs')
        time.sleep(10)
    record('compiling_renderer')
    package=ROOT/'engines/Hunyuan3D-2-main/hy3dgen/texgen/custom_rasterizer';dest=ROOT/'engines/python-packages';dest.mkdir(exist_ok=True)
    env=os.environ.copy();env.update(CUDA_HOME=str(cuda),CUDA_PATH=str(cuda),TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2',PYTHONIOENCODING='utf-8',DISTUTILS_USE_SDK='1')
    env['PATH']=str(cuda/'bin')+os.pathsep+env['PATH']
    env['PYTORCH_NVCC']=(cuda/'bin/nvcc.exe').as_posix()
    command=out/'compile_renderer.cmd'
    command.write_text('@echo off\ncall "'+str(vs)+'"\nif errorlevel 1 exit /b 1\n"'+sys.executable+'" -m pip install "'+str(package)+'" --no-build-isolation --no-deps --target "'+str(dest)+'"\n',encoding='utf-8')
    with (out/'compiler.log').open('w',encoding='utf-8') as log:code=subprocess.call(['cmd.exe','/d','/c',str(command)],env=env,stdout=log,stderr=log)
    if code:raise RuntimeError('Renderer compilation failed; inspect compiler.log')
    record('waiting_for_texture_weights')
    while True:
        p=out/'download.json';d=json.loads(p.read_text()) if p.exists() else {}
        if d.get('stage')=='completed':break
        if d.get('stage')=='failed':raise RuntimeError('Texture download failed: '+d.get('error','unknown'))
        if time.time()-started>12*3600:raise RuntimeError('Texture weight download timeout')
        time.sleep(15)
    record('testing_portrait_texture')
    env['PYTHONPATH']=str(dest)+os.pathsep+str(ROOT)
    with (out/'texture-test.log').open('w',encoding='utf-8') as log:code=subprocess.call([sys.executable,str(ROOT/'pipeline/test_hunyuan_texture.py')],env=env,stdout=log,stderr=log)
    if code:raise RuntimeError('Portrait texture test failed; inspect texture-test.log')
    record('completed')
except Exception as error:record('failed',error=str(error));traceback.print_exc();sys.exit(1)
