from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from pipeline.hunyuan3d_inference import generate_hunyuan3d_mesh

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--image',required=True)
    parser.add_argument('--stl',required=True);parser.add_argument('--glb',required=True)
    parser.add_argument('--quality',choices=['fast','high','ultra'],default='high');parser.add_argument('--texture',action='store_true')
    args=parser.parse_args()
    def progress(stage,percent,message):
        print(json.dumps({'event':'progress','stage':stage,'percent':percent,'message':message}),flush=True)
    generate_hunyuan3d_mesh(args.image,args.stl,args.glb,paint_texture=args.texture,quality=args.quality,
                           paint_face_limit=100000,progress_cb=progress)

if __name__=='__main__':main()
