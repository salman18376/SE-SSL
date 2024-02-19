import numpy as np
from natsort import natsorted
import os
from tools.compute_metrics import compute_metrics
from utils import *
import torchaudio
import soundfile as sf
import argparse
import math
from tqdm import tqdm
import pandas as pd

def get_paths(root_dir):
    ret = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.wav'):
                ret.append(os.path.join(root, f))
    return sorted(ret)

def check_path(path):
    if not os.path.isdir(path): 
        os.makedirs(path)
        
def check_folder(path):
    path_n = '/'.join(path.split('/')[:-1])
    check_path(path_n)
    
def get_score(x):
    for i, s in enumerate(x):
        if math.isnan(s):
            print(i)
    return np.mean(np.array(x))

def evaluation(enhanced, clean):

    pesq, csig, cbak, covl, ssnr = [], [], [], [], []
    stois = []
    filenames = []

    [clean_paths, enhanced_paths] = map(get_paths, [clean, enhanced])
    num = len(clean_paths)

    for enhanced_path,clean_path in tqdm(zip(enhanced_paths,clean_paths)):

        clean_audio, sr = sf.read(clean_path)
        est_audio, sr = sf.read(enhanced_path)
        assert sr == 16000
        metrics = compute_metrics(clean_audio, est_audio, sr, 0)
        metrics = np.array(metrics)
        
        filenames.append(enhanced_path.split('/')[-1])
        pesq.append(metrics[0])
        csig.append(metrics[1])
        cbak.append(metrics[2])
        covl.append(metrics[3])
        ssnr.append(metrics[4])
        stois.append(metrics[5])
        
    filenames.append('Average')
    stois.append(get_score(stois))
    pesq.append(get_score(pesq))
    csig.append(get_score(csig))
    cbak.append(get_score(cbak))
    covl.append(get_score(covl))
    ssnr.append(get_score(ssnr))
        
    return filenames,pesq,csig,cbak,covl,ssnr,stois

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c_path', type=str, default='./clean_folder')
    parser.add_argument('--e_path', type=str, default='./enhanced_folder')
    parser.add_argument('--csv_path', type=str, default=None)  
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_args()
    filenames,pesq,csig,cbak,covl,ssnr,stois = evaluation(args.e_path, args.c_path)
    
    # dictionary of lists  
    dict = {'filename':filenames , 'PESQ': pesq, 'CSIG': csig, 'CBAK': cbak, 'COVL': covl, 'STOI':stois, 'SSNR': ssnr}  
    df = pd.DataFrame(dict) 

    print(f"------------------------------------")
    print(f"results for {args.e_path.split('/')[-1]}")
    print(f"------------------------------------")
    
    # saving the dataframe 
    if not args.csv_path:
        args.csv_path = os.path.join('Result_py',f"{args.csv_path.split('/')[-1]}.csv")
    check_folder(args.csv_path)
    df.to_csv(args.csv_path,index=False) 
    print(args.e_path, args.c_path, pesq[-1], csig[-1], cbak[-1], covl[-1], stois[-1], ssnr[-1])