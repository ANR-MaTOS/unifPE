
import numpy as np
import pandas as pd
from pathlib import Path

from create_test import create_testset_with_stat

from transformers import AutoTokenizer


def get_mersenne_doc_path( lang = 'en'):
    
    return f"{DATA_DIR}/doc_auxiliaire/MERSENNE_article_sep.{lang}"


def main(root_store_dir):

    tokenizer = AutoTokenizer.from_pretrained("Unbabel/TowerBase-7B-v0.1")
    sep_tag = '<sep>' 
    max_n_list =  [ 1024, 1536, 2048, 3072, 4096]

    TEST_DATA_DIR = f"{root_store_dir}/pseudo_doc_MERSENNE-GEOS"
    Path(TEST_DATA_DIR).mkdir(parents=True, exist_ok=True)

    en_fpath = get_mersenne_doc_path( lang = 'en')
    fr_fpath = get_mersenne_doc_path( lang = 'fr')

    testset_stat_dict = create_testset_with_stat(
        tokenizer,  
        en_fpath,
        fr_fpath,
        TEST_DATA_DIR, 
        max_n_list, 
        sep_tag,
        )

    en_test_stat_df = pd.concat(
        testset_stat_dict[n][['count', 'mean', 'min', 'max']].loc[['en']].rename(index = {'en': n}) for n in max_n_list
    )
    print('\nen:')
    print(en_test_stat_df.astype(int).to_markdown())
        
    fr_test_stat_df = pd.concat(
        testset_stat_dict[n][['count', 'mean', 'min', 'max']].loc[['fr']].rename(index = {'fr': n}) for n in max_n_list
    )
    print('\nfr:')
    print(fr_test_stat_df.astype(int).to_markdown())


if __name__ == "__main__":
    DATA_DIR = '/home/peng/MaTOS/data_txt/data/STEP/testset/MERSENNE-GEOS'
    
    main(root_store_dir = DATA_DIR)



