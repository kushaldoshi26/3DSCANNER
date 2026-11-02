"""Official THuman4 download; requires explicit research terms acceptance."""
from pathlib import Path
import argparse,json,shutil,time,os
import gdown
LINKS={'00':('1KqA57I7KPvP9A7_hPiZd5ot2zu1jK5Dx',25),'01':('1ZBOJldrZPxGttzvX74IemVa9JMLjY9vh',58),'02':('1MsFx3xTYygdetfhc1kVs2W3pDSsAHKJg',36)}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--subject',choices=LINKS,default='00');parser.add_argument('--research-terms-accepted',action='store_true');parser.add_argument('--root',default='D:/3dSCANNER/data/datasets/thuman4');args=parser.parse_args()
    if not args.research_terms_accepted:raise SystemExit('Explicit acceptance of official non-commercial research terms is required before downloading.')
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    identifier,size_gb=LINKS[args.subject]
    if shutil.disk_usage(root).free<(size_gb+10)*1024**3:raise RuntimeError('Insufficient free space for archive plus working reserve')
    target=root/f'subject{args.subject}.7z';status=root/f'subject{args.subject}_status.json'
    def record(state,**extra):
        temporary=status.with_suffix('.tmp');temporary.write_text(json.dumps({'status':state,'subject':args.subject,'archive':str(target),'official_source':'https://github.com/ZhengZerong/THUman4.0-Dataset','research_terms_accepted':True,'updated':time.time(),**extra},indent=2));os.replace(temporary,status)
    record('downloading')
    try:
        result=gdown.download(id=identifier,output=str(target),quiet=False,resume=True,use_cookies=False)
        if not result:raise RuntimeError('Official Google Drive download did not return an archive; access or quota may be required')
        with target.open('rb') as stream:
            if stream.read(6)!=b'7z\xbc\xaf\x27\x1c':raise RuntimeError('Downloaded file is not a 7z archive')
        record('downloaded',bytes=target.stat().st_size)
    except Exception as error:record('failed',error=str(error));raise
if __name__=='__main__':main()
