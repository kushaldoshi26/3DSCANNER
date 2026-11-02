"""Download official runtime weights; no training dataset or credentials copied."""
import argparse, os, json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--revision', required=True, help='Official Hugging Face commit SHA to pin')
    parser.add_argument('--accept-license', action='store_true')
    args=parser.parse_args()
    if not args.accept_license:
        parser.error('Review Tencent Hunyuan3D-2 LICENSE first, then pass --accept-license')
    if len(args.revision)!=40 or any(c not in '0123456789abcdef' for c in args.revision.lower()):
        parser.error('--revision must be a 40-character commit SHA')
    from huggingface_hub import snapshot_download
    base=Path(os.environ.get('HY3DGEN_MODELS', Path(__file__).resolve().parents[1]/'engines/models'))
    target=base/'tencent/Hunyuan3D-2'
    snapshot_download(repo_id='tencent/Hunyuan3D-2', revision=args.revision, local_dir=str(target),
        allow_patterns=['hunyuan3d-dit-v2-0/*','hunyuan3d-delight-v2-0/*','hunyuan3d-paint-v2-0-turbo/*','README.md','LICENSE*'])
    (target/'scanner-download.json').write_text(json.dumps({'repo':'tencent/Hunyuan3D-2','revision':args.revision},indent=2),encoding='utf-8')
    print('Official model snapshot downloaded to',target)
if __name__=='__main__': main()
