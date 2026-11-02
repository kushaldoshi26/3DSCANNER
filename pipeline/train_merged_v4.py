from pathlib import Path
root=Path(__file__).resolve().parent.parent
source=(root/'pipeline/merge_and_train.py').read_text()
source=source.replace('merged_v3_candidate','merged_v4_candidate')
insertion="source=source.replace('lr=2e-4','lr=5e-5')"+chr(10)
source=source.replace('exec(compile(source,',insertion+'exec(compile(source,')
exec(compile(source,str(root/'pipeline/merge_and_train.py'),'exec'),{'__name__':'__main__','__file__':str(root/'pipeline/merge_and_train.py')})
