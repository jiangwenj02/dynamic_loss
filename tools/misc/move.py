import os,shutil,glob,tqdm
 
sourcefile='/data3/zzhang/ours/ILSVRC2012/train'
#new_file=str(glob.glob(os.path.join(sourcefile,'*.txt')))
Txt=glob.glob(os.path.join(sourcefile,'*.JPEG'))
for file_path in tqdm.tqdm(Txt):
    new_dir=file_path.split('_')[0]
    try:
        os.makedirs(os.path.join(sourcefile,new_dir))
    except:
        pass
 
    shutil.move(file_path,os.path.join(sourcefile,new_dir))