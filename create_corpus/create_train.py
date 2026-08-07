from transformers import NllbTokenizer
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def get_nllb_tokenizer():

    source = 'eng_Latn' # English
    target = 'fra_Latn' # French
    tokenizer = NllbTokenizer.from_pretrained(
            'facebook/nllb-200-distilled-600M',
            src_lang=source,
            tgt_lang=target
        )
    return tokenizer 


def get_length_string(txt, tokenizer):
    assert isinstance(txt, str)
    return tokenizer( [txt] , return_length = True, add_special_tokens = False)['length'][0]


def get_stat(en_fpath, fr_fpath, tokenizer):
    
    with open(en_fpath, encoding="utf-8") as f:
        en_txt = f.read().splitlines()
    
    with open(fr_fpath, encoding="utf-8") as f:
        fr_txt = f.read().splitlines()

    assert len(en_txt) == len(fr_txt)
        
    corpus_len_en = tokenizer( en_txt, return_length = True, add_special_tokens=False, return_attention_mask=False)['length']
    corpus_len_fr = tokenizer( fr_txt, return_length = True, add_special_tokens=False, return_attention_mask=False)['length']

    res_dict = {}
    res_dict['en'] = {i: l for i, l in enumerate(corpus_len_en)}
    res_dict['fr'] = {i: l for i, l in enumerate(corpus_len_fr)}
        
    return pd.DataFrame(res_dict)



def pseudo_doc_with_max_len_range(en_fpath, fr_fpath, tokenizer, store_path_prefix, max_n, min_n, sep_tag = '<sep>' ):
    # en_fpath: doc path of doc with sep_tag for sentence boundaries
    with open(en_fpath, encoding="utf-8") as f:
        en_txt = f.read().splitlines()

    with open(fr_fpath, encoding="utf-8") as f:
        fr_txt = f.read().splitlines()
        
    en_pseudo_docs = []
    fr_pseudo_docs = []
    for i in range(len(en_txt)):
        en_sents = [s.strip() for s in en_txt[i].split(sep_tag)]
        fr_sents = [s.strip() for s in fr_txt[i].split(sep_tag)]
        
        doc_len = 0
        en_doc = []
        fr_doc = []
        max_len = np.random.randint(min_n, max_n, 1)[0]
        for j in range(len(en_sents)):
            sent_len = get_length_string(en_sents[j], tokenizer)

            if doc_len + sent_len >= max_len :
                en_pseudo_docs.append(f"{sep_tag} ".join(en_doc))
                fr_pseudo_docs.append(f"{sep_tag} ".join(fr_doc))
                en_doc = [en_sents[j]]
                fr_doc = [fr_sents[j]]
                doc_len = sent_len
                # update max_len for next pseudo_doc
                max_len = np.random.randint(min_n, max_n, 1)[0]
            else:
                en_doc.append(en_sents[j])
                fr_doc.append(fr_sents[j])
                doc_len += sent_len
            # end of doc
            if j == len(en_sents)-1 and len(en_doc) > 0:
                en_pseudo_docs.append(f"{sep_tag} ".join(en_doc))
                fr_pseudo_docs.append(f"{sep_tag} ".join(fr_doc))
                en_doc = []
                fr_doc = []
                doc_len = 0
                
    # store
    # print(len(en_pseudo_docs))
    with open( f"{store_path_prefix}.en", 'w', encoding="utf-8" ) as f:
        f.write( re.sub( sep_tag, '',  '\n'.join( en_pseudo_docs) ) )
    
    with open( f"{store_path_prefix}.fr", 'w', encoding="utf-8" ) as f:
       f.write(re.sub(  sep_tag, '',  '\n'.join(fr_pseudo_docs) ) )

        
    with open( f"{store_path_prefix}_sep.en", 'w', encoding="utf-8" ) as f:
        f.write( '\n'.join( en_pseudo_docs) ) 
    
    with open( f"{store_path_prefix}_sep.fr", 'w', encoding="utf-8" ) as f:
       f.write( '\n'.join(fr_pseudo_docs) ) 


def pseudo_doc_with_max_len_range_custom(
    en_fpath, 
    fr_fpath, 
    tokenizer, 
    store_path_prefix, 
    max_n, 
    min_n, 
    train_stat_df, 
    idx_to_keep = [],
    sep_tag = '<sep>'
):
    # en_fpath: doc path of doc with sep_tag for sentence boundaries

    with open(en_fpath, encoding="utf-8") as f:
        en_txt = f.read().splitlines()

    with open(fr_fpath, encoding="utf-8") as f:
        fr_txt = f.read().splitlines()
    
    hist, bin_edges = np.histogram(train_stat_df['en'], density = True, bins = max_n - min_n)
    prob_list = hist*(bin_edges[1]- bin_edges[0]) 
    
    en_pseudo_docs = []
    fr_pseudo_docs = []
    
    for i in range(len(en_txt)):
        if i in idx_to_keep:
            en_pseudo_docs.append(en_txt[i])
            fr_pseudo_docs.append(fr_txt[i])
            continue
            
        en_sents = [s.strip() for s in en_txt[i].split(sep_tag)]
        fr_sents = [s.strip() for s in fr_txt[i].split(sep_tag)]
        
        doc_len = 0
        en_doc = []
        fr_doc = []
        max_len = np.random.choice( np.arange(min_n, max_n), p=prob_list )
        # tmp_list.append(max_len)
        for j in range(len(en_sents)):
            sent_len = get_length_string(en_sents[j], tokenizer)
            
            if doc_len + sent_len >= max_len :
                en_pseudo_docs.append(f"{sep_tag} ".join(en_doc))
                fr_pseudo_docs.append(f"{sep_tag} ".join(fr_doc))
                en_doc = [en_sents[j]]
                fr_doc = [fr_sents[j]]
                doc_len = sent_len
                # update max_len for next pseudo_doc
                max_len = np.random.choice( np.arange(min_n, max_n), p=prob_list )
            else:
                en_doc.append(en_sents[j])
                fr_doc.append(fr_sents[j])
                doc_len += sent_len
            # end of doc
            if j == len(en_sents)-1 and len(en_doc) > 0:
                en_pseudo_docs.append(f"{sep_tag} ".join(en_doc))
                fr_pseudo_docs.append(f"{sep_tag} ".join(fr_doc))
                en_doc = []
                fr_doc = []
                doc_len = 0
                
    # store
    # print(len(en_pseudo_docs))
    with open( f"{store_path_prefix}.en", 'w', encoding="utf-8" ) as f:
        f.write( re.sub( sep_tag, '',  '\n'.join( en_pseudo_docs) ) )
    
    with open( f"{store_path_prefix}.fr", 'w', encoding="utf-8" ) as f:
       f.write(re.sub(  sep_tag, '',  '\n'.join(fr_pseudo_docs) ) )

        
    with open( f"{store_path_prefix}_sep.en", 'w', encoding="utf-8" ) as f:
        f.write( '\n'.join( en_pseudo_docs) ) 
    
    with open( f"{store_path_prefix}_sep.fr", 'w', encoding="utf-8" ) as f:
       f.write( '\n'.join(fr_pseudo_docs) ) 
    # return tmp_list
        


def create_corpus_TED_train(sep_tag = '<sep>'    ):
    
    tokenizer = get_nllb_tokenizer()

    # length distribution
    ted_train_stat_df = pd.read_csv(f"{DATA_DIR}/TED_doc/TED_train_doc_stat_NLLB_tokenizer.tsv", sep = '\t', index_col = 0)
    MIN_N = min(ted_train_stat_df['en'])
    MAX_N = 1024

    STORE_DIR = f"{DATA_DIR}/TED_pseudo_doc_len{MIN_N}-{MAX_N}"
    Path(STORE_DIR).mkdir(parents=True, exist_ok=True)


    en_fpath = f"{DATA_DIR }/TED_doc/TED_train_doc_sep.en"
    fr_fpath = f"{DATA_DIR }/TED_doc/TED_train_doc_sep.fr"


    store_path_prefix = f"{STORE_DIR}/TED_train_doc"

    tmp_list = pseudo_doc_with_max_len_range_custom(
        en_fpath, 
        fr_fpath, 
        tokenizer, 
        store_path_prefix, 
        max_n = MAX_N,
        min_n = MIN_N, 
        sep_tag = sep_tag,
        idx_to_keep = ted_train_stat_df[ted_train_stat_df['en'] < MAX_N].index,
        train_stat_df = ted_train_stat_df,
    )

    res_dict  = get_stat(en_fpath = f"{store_path_prefix}.en", fr_fpath = f"{store_path_prefix}.fr", tokenizer = tokenizer)

    print(pd.DataFrame(res_dict).sum(axis = 0))
    pd.DataFrame(res_dict).to_csv(f"{DATA_DIR}/TED_pseudo_doc_len{MIN_N}-{MAX_N}/TED_train_doc_stat_NLLB_tokenizer.tsv", sep = '\t')
    # pd.DataFrame(res_dict)[['en']].plot(kind = 'hist')


def create_corpus_TED_dev(tokenizer, sep_tag = '<sep>' ):
    
    # length distribution
    ted_train_stat_df = pd.read_csv(f"{DATA_DIR}/TED_doc/TED_train_doc_stat_NLLB_tokenizer.tsv", sep = '\t', index_col = 0)
    MIN_N = min(ted_train_stat_df['en'])
    MAX_N = 1024

    STORE_DIR = f"{DATA_DIR}/TED_pseudo_doc_len{MIN_N}-{MAX_N}"
    Path(STORE_DIR).mkdir(parents=True, exist_ok=True)


    en_fpath = f"{DATA_DIR }/TED_doc/TED_train_doc_sep.en"
    fr_fpath = f"{DATA_DIR }/TED_doc/TED_train_doc_sep.fr"


    store_path_prefix = f"{STORE_DIR}/TED_train_doc"

    tmp_list = pseudo_doc_with_max_len_range_custom(
        en_fpath, 
        fr_fpath, 
        tokenizer, 
        store_path_prefix, 
        max_n = MAX_N,
        min_n = MIN_N, 
        sep_tag = sep_tag,
        idx_to_keep = ted_train_stat_df[ted_train_stat_df['en'] < MAX_N].index,
        train_stat_df = ted_train_stat_df,
    )

    res_dict  = get_stat(en_fpath = f"{store_path_prefix}.en", fr_fpath = f"{store_path_prefix}.fr", tokenizer = tokenizer)
    # display(pd.DataFrame(res_dict).describe().T)
    print(pd.DataFrame(res_dict).sum(axis = 0))
    pd.DataFrame(res_dict)[['en']].plot(kind = 'hist')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_n", type=int, required=True, help="minimum length of the target corpus")
    parser.add_argument("--max_n", type=int, required=True, help="maximum length of the target corpus")

    args = parser.parse_args()
            
    min_n = args.min_n
    max_n = args.max_n

    DATA_DIR = '/home/peng/MaTOS/data_txt/data/TED-2017'


