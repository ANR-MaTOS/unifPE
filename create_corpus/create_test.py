import re
from transformers import NllbTokenizer

import numpy as np
import pandas as pd
from pathlib import Path

from create_train import (
    get_nllb_tokenizer,
    get_length_string,
    get_stat,
)


def pseudo_doc_with_max_len( 
        en_fpath, fr_fpath, tokenizer, store_path_prefix, max_n, sep_tag = '<sep>', to_store=True
        ):
    # en_fpath: doc path of doc with sep_tag for sentence boundaries

    with open(en_fpath, encoding="utf-8") as f:
        en_txt = f.read().splitlines()
    
    with open(fr_fpath, encoding="utf-8") as f:
        fr_txt = f.read().splitlines()
    
    en_pseudo_docs = []
    fr_pseudo_docs = []
    nb_doc_dict = {}
    
    for i in range(len(en_txt)):
        en_sents = [s.strip() for s in en_txt[i].split(sep_tag)]
        fr_sents = [s.strip() for s in fr_txt[i].split(sep_tag)]
        nb_doc_dict[i] = 0
        en_doc = []
        fr_doc = []
        doc_len = 0

        for j in range(len(en_sents)):
            sent_len = get_length_string(en_sents[j], tokenizer)
    
            if doc_len + sent_len >= max_n :
                if j == len(en_sents)-1:
                    print(f'last sent: {en_sents[j]}\ndoc_len = {doc_len}, last_sent_len = {sent_len}')
                    en_pseudo_docs.append(f"{sep_tag} ".join(en_doc + [en_sents[j]]))
                    fr_pseudo_docs.append(f"{sep_tag} ".join(fr_doc + [fr_sents[j]] ))
                    nb_doc_dict[i] +=1
                    # idx_list += [f"D{n_doc}.{l}" for l in range(len(en_doc + [en_sents[j]])) ]
                    # reset
                    doc_len = 0
                    en_doc = []
                    fr_doc = []                     
                else:  
                    en_pseudo_docs.append(f"{sep_tag} ".join(en_doc))
                    fr_pseudo_docs.append(f"{sep_tag} ".join(fr_doc))
                    nb_doc_dict[i] +=1
                    # reset
                    doc_len = sent_len
                    en_doc = [en_sents[j]]
                    fr_doc = [fr_sents[j]]
                    
            else:
                en_doc.append(en_sents[j])
                fr_doc.append(fr_sents[j])
                doc_len += sent_len
                

                if j == len(en_sents)-1 and len(en_pseudo_docs) ==0:
                    print(f'whole doc = {doc_len} < {max_n}')
                    en_pseudo_docs.append(f"{sep_tag} ".join(en_doc ))
                    fr_pseudo_docs.append(f"{sep_tag} ".join(fr_doc ))
                    nb_doc_dict[i] +=1
                    doc_len = 0
                    en_doc = []
                    fr_doc = []
                    

                if j == len(en_sents)-1 and len(en_doc) > 0: 
                    new_en_doc = f"{sep_tag} ".join( [en_pseudo_docs[-1]] + en_doc )
                    new_fr_doc = f"{sep_tag} ".join( [fr_pseudo_docs[-1]] + fr_doc )
                    if doc_len < 50:
                        print(f'doc_len = {doc_len} < 50, concatenate to the last doc')
                        en_pseudo_docs[-1] = new_en_doc
                        fr_pseudo_docs[-1] = new_fr_doc
                         
                    else:
                        print(f'doc_len = {doc_len} < {max_n}, take the whole doc')
                        # new_en_sents = [ l.strip() for l in new_en_doc.split(sep_tag)]
                        # new_fr_sents = [ l.strip() for l in new_fr_doc.split(sep_tag)]
                        # nb_s = len(new_en_sents)//2

                        # en_pseudo_docs += [f"{sep_tag} ".join(new_en_sents[:nb_s ]), f"{sep_tag} ".join(new_en_sents[nb_s: ])]
                        # fr_pseudo_docs += [f"{sep_tag} ".join(new_fr_sents[:nb_s ]), f"{sep_tag} ".join(new_fr_sents[nb_s: ])]                        
                        
                        en_pseudo_docs.append(f"{sep_tag} ".join(en_doc ))
                        fr_pseudo_docs.append(f"{sep_tag} ".join(fr_doc ))
                        nb_doc_dict[i] +=1
                    # reset
                    doc_len = 0
                    en_doc = []
                    fr_doc = []
            
            
    pd.DataFrame(nb_doc_dict, index = ['nb-pseudo-doc']).T.to_csv(f"{store_path_prefix}.nb_doc.stat.tsv", sep = '\t')
    # print(len(en_pseudo_docs))
    if to_store:
        with open( f"{store_path_prefix}.en", 'w', encoding="utf-8" ) as f:
            f.write( re.sub( sep_tag, '',  '\n'.join( en_pseudo_docs) ) )
        
        with open( f"{store_path_prefix}.fr", 'w', encoding="utf-8" ) as f:
           f.write(re.sub(  sep_tag, '',  '\n'.join(fr_pseudo_docs) ) )
    
            
        with open( f"{store_path_prefix}_sep.en", 'w', encoding="utf-8" ) as f:
            f.write( '\n'.join( en_pseudo_docs) ) 
        
        with open( f"{store_path_prefix}_sep.fr", 'w', encoding="utf-8" ) as f:
           f.write( '\n'.join(fr_pseudo_docs) ) 


def create_testset_with_stat(
        tokenizer, 
        en_fpath,
        fr_fpath,
        store_dir, 
        max_n_list, 
        sep_tag = '<sep>'
        ):
    
    testset_stat_dict = {}
    
    for max_n in max_n_list :
        print(max_n)
        store_path_prefix = f"{store_dir}/pseudo_doc_max{max_n}"

        # construct test sets
        pseudo_doc_with_max_len(en_fpath, fr_fpath, tokenizer, store_path_prefix, max_n, sep_tag = sep_tag )

        res_dict  = get_stat(en_fpath = f"{store_path_prefix}.en", fr_fpath = f"{store_path_prefix}.fr", tokenizer = tokenizer)
        
        tmp_stat_df = pd.DataFrame(res_dict)
        testset_stat_dict[max_n] = tmp_stat_df.describe().T
    return testset_stat_dict



def get_ted_doc_path(year, lang = 'en'):
    return f"{DATA_DIR }/testset/doc/TED_test{year}_doc_sep.{lang}"

def main(root_store_dir):

    tokenizer = get_nllb_tokenizer()
    sep_tag = '<sep>' 
    max_n_list =  [ 256, 512,  768, 1024, 1200, 1600, 2048 ]

    for year in ['2014', '2015', '2016', '2017']:
        print(year)

        TEST_DATA_DIR = f"{root_store_dir}/testset/pseudo_doc_test{year}"
        Path(TEST_DATA_DIR).mkdir(parents=True, exist_ok=True)

        en_fpath = get_ted_doc_path(year, lang = 'en')
        fr_fpath = get_ted_doc_path(year, lang = 'fr')

        testset_stat_dict = create_testset_with_stat(
            tokenizer,  
            en_fpath,
            fr_fpath,
            TEST_DATA_DIR, 
            max_n_list, 
            sep_tag
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
    DATA_DIR = '/home/peng/MaTOS/data_txt/data/TED-2017'
    main(root_store_dir = DATA_DIR)
