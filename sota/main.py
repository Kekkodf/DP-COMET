import DPMLM
import ir_datasets
import pandas as pd
from tqdm import tqdm
import os
import time
tqdm.pandas()

if __name__ == "__main__":
    t_0 = time.time()
    collection = 'robust04'
    dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
    num_rep = 50
    df_main = pd.DataFrame(dataset.queries_iter())
    df_main = df_main.rename(columns={"title": "text"})
    df_main = df_main.rename(columns={"query_id": "id"})
    df_main.drop(columns=['description', 'narrative'], inplace=True)

    
    epsilons = [1, 5, 10, 12.5, 15, 17.5, 20, 25, 30, 50]
    mech = DPMLM.DPMLM()
    for epsilon in epsilons:
        df = df_main.copy()
        df = pd.concat([df]*num_rep, ignore_index=True).reset_index(drop=True)
        df.sort_values(by=['id'], inplace=True)
        results = [val[0] for val in df['text'].progress_apply(mech.dpmlm_rewrite, args=(epsilon,)).values]
        df['obfuscatedText'] = results
        #expand 
        df['obfuscatedText'] = df['obfuscatedText'].astype(str)
        #make a column with mechanism name
        df['mechanism'] = 'DPMLM'
        #make a column with epsilon
        df['epsilon'] = epsilon
    #save to csv
        os.makedirs(f"./sota/{collection}/", exist_ok=True)
        df.to_csv(f"./sota/{collection}/obfuscatedText_DPMLM_{epsilon}.csv", index=False)
    t_robust = time.time() - t_0
    
    
    collection = 'medline04'
    dataset = ir_datasets.load("medline/2004/trec-genomics-2004")
    num_rep = 50
    df_main = pd.DataFrame(dataset.queries_iter())
    df_main = df_main.rename(columns={"title": "text"})
    df_main = df_main.rename(columns={"query_id": "id"})
    df_main.drop(columns=['need', 'context'], inplace=True)

    for epsilon in epsilons:
        df = df_main.copy()
        df = pd.concat([df]*num_rep, ignore_index=True).reset_index(drop=True)
        df.sort_values(by=['id'], inplace=True)
        results = [val[0] for val in df['text'].progress_apply(mech.dpmlm_rewrite, args=(epsilon,)).values]
        df['obfuscatedText'] = results
        #expand 
        df['obfuscatedText'] = df['obfuscatedText'].astype(str)
        #make a column with mechanism name
        df['mechanism'] = 'DPMLM'
        #make a column with epsilon
        df['epsilon'] = epsilon
    #save to csv
        os.makedirs(f"./sota/{collection}/", exist_ok=True)
        df.to_csv(f"./sota/{collection}/obfuscatedText_DPMLM_{epsilon}.csv", index=False)
    
    print("Time taken robust04: ", t_robust)
    print("Time taken medline04: ", time.time() - t_robust)
    