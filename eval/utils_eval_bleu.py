# compute BLEU scores
import re
import os
import pandas as pd
from pathlib import Path

from utils_post_process import re_align_doc2sent_file_idx

#pattern for sacrebleu signature
BP_PATTERN = re.compile(r'.+BP = (\d+\.\d+).+')
BLEU_PATTERN = re.compile(r'.*BLEU = (\d+\.\d+) .+')
SBLEU_PATTERN = re.compile(r'.*(BLEU[^=]+) = (\d+\.\d+) .+')



def eval_sacrebleu( ref_fpath, sys_fpath, metric = 'bleu', store_fpath = None):
    end = f" > {store_fpath}" if store_fpath else ""
    os.system(f"sacrebleu {ref_fpath} < {sys_fpath} -m {metric}" + end)


def eval_doc2sent( sent_store_prefix , sys_fpath, ref_fpath,  store_fpath, sep_tok = '<sep>'):
    """
    sent_store_prefix = out_file_path
    ref_fpath with sep tag such as '<sep>' 
    """
    sys_sent_fpath = f"{sent_store_prefix}.sent.sys"
    ref_sent_fpath = f"{sent_store_prefix}.sent.ref"


    # doc 2 sent with edlib
    re_align_doc2sent_file_idx( 
        sys_fpath,  
        ref_fpath, 
        sys_store_path  = sys_sent_fpath, 
        ref_store_path = ref_sent_fpath, 
        eos_tag = sep_tok.strip()
        )
    
    eval_sacrebleu( 
        ref_sent_fpath, 
        sys_sent_fpath, 
        metric = 'bleu', 
        store_fpath = store_fpath,
        )

    
 # sentence-BLEU   
def get_sbleu_info(sbleu_path):
    """
    sbleu_path: output of sacrebleu --sentence-level ref < sys, 
    every line contains signature of sentence BLEU for a translation hypothesis
    """
    sbleu_info = [ l.strip() for l in open(sbleu_path, 'r').read().split('\n') if l.strip()]
    print(len(sbleu_info))
    signature = re.match(SBLEU_PATTERN, sbleu_info[0]).group(1)
    print( f"signature = {signature}")
    sbleu_dict = {
        'BLEU': {},
        'BP':{},
        'verbose_score': {},
    }

    for i in range(len(sbleu_info)):
        sbleu_dict['verbose_score'][i] = sbleu_info[i][len(signature):]
        sbleu_dict['BLEU'][i] = float(re.match(SBLEU_PATTERN, sbleu_info[i]).group(2).strip())
        sbleu_dict['BP'][i] = float(re.match(BP_PATTERN, sbleu_info[i]).group(1).strip())

    return pd.DataFrame(sbleu_dict)


def compute_bleu_scores_file(
        ref_fpath, 
        sys_fpath, 
        sent_store_prefix,
        ref_sep_fpath, # for realignment
        ref_sent_fpath,
        eval_store_path,
        sep_tok = '<sep>', # for realignment
        ):
    # TODO remove out_file_path as param
    bleu_fname_dict = {
        'vanilla-BLEU':f"{os.path.basename(sys_fpath)}.sent.bleu.json",
        's-BLEU': f"{os.path.basename(sys_fpath)}.s-bleu.tsv",
        'ds-BLEU': f"{os.path.basename(sys_fpath)}.ds-bleu.tsv",
        'd-BLEU':f"{os.path.basename(sys_fpath)}.d-bleu.json",
    }
    Path(eval_store_path).mkdir(parents=True, exist_ok=True)
    
    bleu_dict = {}
    # corpus BLEU  d-BLEU
    print('d-bleu')
    score_fpath = os.path.join( eval_store_path, bleu_fname_dict['d-BLEU'])
    eval_sacrebleu( 
        ref_fpath, 
        sys_fpath, 
        metric = 'bleu', 
        store_fpath = score_fpath
        )
    # record
    score_df = pd.read_json(score_fpath, typ = 'series').to_frame().rename(index = {'score':'BLEU'}).T
    score_df['BP'] = float(re.match(BP_PATTERN, score_df['verbose_score'].values[0] ).group(1).strip())
    bleu_dict['d-BLEU'] = { 'BLEU': score_df['BLEU'].values[0],
                            'BP':score_df['BP'].values[0]}
        
    # vanilla-BLEU
    print('vanilla-BLEU')
    # realigned sentences will be store at f"{sent_store_prefix}.sent.sys"
    try:
        store_fpath = os.path.join( eval_store_path, f"{os.path.basename(sys_fpath)}.sent.bleu.json")
        eval_doc2sent( sent_store_prefix , sys_fpath, ref_sep_fpath, store_fpath, sep_tok = sep_tok)
        # record
        score_df = pd.read_json(
            os.path.join( eval_store_path, bleu_fname_dict['vanilla-BLEU']),
            typ = 'series').to_frame().rename(index = {'score':'BLEU'}).T
        score_df['BP'] = float(re.match(BP_PATTERN, score_df['verbose_score'].values[0] ).group(1).strip())
        bleu_dict['vanilla-BLEU'] = { 'BLEU': score_df['BLEU'].values[0],
                                      'BP':score_df['BP'].values[0]}
    except:
        print("Cannot compute vanilla BLEU")

    # ds-BLEU
    print('ds-BLEU')
    sbleu_path = os.path.join( eval_store_path, f"{os.path.basename(sys_fpath)}.ds-bleu.txt")
    eval_sacrebleu( 
        ref_fpath, 
        sys_fpath, 
        metric = 'bleu --sentence-level', 
        store_fpath = sbleu_path
        )
    sbleu_df = get_sbleu_info(sbleu_path)
    sbleu_df.to_csv( os.path.join(eval_store_path , bleu_fname_dict['ds-BLEU']), sep = '\t')
    bleu_dict['ds-BLEU'] = sbleu_df[['BLEU', 'BP']].mean().to_dict()

    # s-BLEU
    print('s-BLEU')
    sbleu_path = os.path.join( eval_store_path, f"{os.path.basename(sys_fpath)}.s-bleu.txt")
    try:
        eval_sacrebleu( 
            ref_sent_fpath, 
            sys_fpath = f"{sent_store_prefix}.sent.sys", 
            metric = 'bleu --sentence-level', 
            store_fpath = sbleu_path
            )
        sbleu_df = get_sbleu_info(sbleu_path)
        sbleu_df.to_csv( os.path.join( eval_store_path, bleu_fname_dict['s-BLEU']), sep = '\t')
        bleu_dict['s-BLEU'] = sbleu_df[['BLEU', 'BP']].mean().to_dict()
    except:
        print('Error when evaluating s-BLEU')

    score_all_df = pd.DataFrame(bleu_dict)
    bleu_fname_dict['all'] = f"{os.path.basename(sys_fpath)}.bleu.all.tsv"
    score_all_df.to_csv( os.path.join(eval_store_path, bleu_fname_dict['all']), sep = '\t')
    return score_all_df, bleu_fname_dict


