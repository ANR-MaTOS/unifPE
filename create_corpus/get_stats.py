from post_process_utils import sent2doc_segid
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = '/home/peng/MaTOS/data_txt/data/TED-2017'
stat_dir_dict = {
    'TED-full': {
        'train': f"{DATA_DIR}/TED_doc/TED_train_doc_stat_NLLB_tokenizer.tsv",
        'dev':  f"{DATA_DIR}/TED_doc/TED_dev_doc_stat_NLLB_tokenizer.tsv",
    }, 
    'TED-distill':{
        'train': f"{DATA_DIR}/TED_pseudo_doc_len56-1024/TED_train_doc_stat_NLLB_tokenizer.tsv",
        'dev':f"{DATA_DIR}/TED_pseudo_doc_len56-1024/TED_dev_doc_stat_NLLB_tokenizer.tsv",
    },
    'TED-unif':{
        'train':  f"{DATA_DIR}/TED_pseudo_doc_len128-1024/TED_train_doc_stat_NLLB_tokenizer.tsv",
        'dev': f"{DATA_DIR}/TED_pseudo_doc_len128-1024/TED_dev_doc_stat_NLLB_tokenizer.tsv",
    },
    'TED-sent':{
        'train':  f"{DATA_DIR}/TED_sent/TED_train_sent_stat_NLLB_tokenizer.tsv",
        'dev': f"{DATA_DIR}/TED_sent/TED_dev_sent_stat_NLLB_tokenizer.tsv",
    }

}

res_table = {}
for key in stat_dir_dict.keys():
    # res_table[key] = {}
    for s in ['train', 'dev']:
        stat_df = pd.read_csv(stat_dir_dict[key][s], sep = '\t', index_col = 0)
    
        key1 = f"\\corpus{{{key}}}"
        res_table[(key1, s)] = {}
        for i in ['count','mean','min','max']:
            tmp_df = stat_df.describe().T[i]
            
            if i == 'count':
                res_table[(key1, s)][i] = int(tmp_df['en'])
                continue
            res_table[(key1, s)][i] = int(tmp_df['en']) #f"{int(tmp_df['en'])} /{int(tmp_df['fr']) }"
        

stat_df = pd.DataFrame(res_table)

print(stat_df.T[['count','mean','min','max']].to_markdown())

print(stat_df.T[['count','mean','min','max']].style.format(precision = 0).to_latex())