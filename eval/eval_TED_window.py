from pathlib import Path
import os
import pandas as pd
import re

from scipy import stats
from utils_eval_bleu import (
    eval_sacrebleu,
    get_sbleu_info,
    BP_PATTERN, 
    # BLEU_PATTERN,
    )
import numpy as np

# from data_paths import (
#     get_stat_path,
#     get_begin_id_path,
#     get_testset_ref_path,
#     get_ref_sep_path,
# )


##############################################
def get_empty_align_stat(year_list, max_len_list, md_fname, get_testset_ref_path, get_realign_sys_path):
    # return statistics of empty alignment for realigned translations of TED pseudo doc
    # of year and context window size given in year_list and max_len_list
    # sentence-level reference : get_testset_ref_path(year=year, level = 'sent')
    # translation hypothesis path : get_realign_sys_path(md_fname, year, max_n)
    empty_align_dict = {}
    for year in year_list:
        empty_align_dict[year] = {}
        ref_fpath = get_testset_ref_path(year=year, level = 'sent')
        ref_txt =  open(ref_fpath, 'r').read().strip().split('\n')    
        
        for max_n in  max_len_list:
            sys_fpath = get_realign_sys_path(md_fname, year, max_n)
            sys_txt = open(sys_fpath, 'r').read().split('\n')
            if len(sys_txt) == len(ref_txt)+1 and sys_txt[-1] == '':
                # mwersegementer may add a new line at the end of sys file
                sys_txt = sys_txt[:-1]
            if len(ref_txt) != len(sys_txt):
                print(year, max_n, len(ref_txt), len(sys_txt))
                break
            empty_align_dict[year][max_n] = len([l for l in sys_txt if l.strip() == '' ])
    return empty_align_dict



def get_bleu_dict(year_list, max_len_list, key, get_testset_ref_path, get_realign_sys_path, store_fpath):
    # ZPTODO clean and test this function
    # compute BLEU score for realigned translations of TED window testsets
    bleu_dict = {}
    for year in year_list:
        bleu_dict[year] = {}
        ref_fpath = get_testset_ref_path(year=year, level = 'sent')
        
        for max_n in max_len_list:
            
            sys_fpath = get_realign_sys_path(key, year, max_n)
                
            print(year, max_n)
            # store_fpath = f"{res_dir}/{os.path.basename(sys_fpath)}.bleu.json"
            # print(store_fpath)
            eval_sacrebleu( ref_fpath, sys_fpath, metric = 'bleu', store_fpath = store_fpath)

            score_df = pd.read_json(
                store_fpath, typ = 'series').to_frame().rename(index = {'score':'BLEU'}).T
            
            score_df['BP'] = float(re.match(BP_PATTERN, score_df['verbose_score'].values[0] ).group(1).strip())
            # bleu_dict[key][year][max_n]= { 'BLEU': score_df['BLEU'].values[0],
            #                               'BP':score_df['BP'].values[0]}
            tmp_bleu = round( score_df['BLEU'].values[0], 1)
            tmp_bp = round(score_df['BP'].values[0], 2 )
            # bleu_dict[key][year][max_n]=  f"{tmp_bleu} \\footnotesize{{({tmp_bp})"
            
            # bleu_dict[key][year][max_n]=  f"{tmp_bleu} \\footnotesize{{({tmp_bp})}}"
            # ZPTODO define options to choose output format
            bleu_dict[key][year][max_n]=  f"{tmp_bleu} ({tmp_bp})"


##############################################
################ rewrite as recover_full_doc.py #############################

# recover full doc
# 1. full doc
def recover_doc(md_fname, year, max_n, get_stat_path, get_sys_path, get_store_full_doc_path  ):
    # get_store_full_doc_path is a function
    stat_df = pd.read_csv(get_stat_path(year, max_n), sep = '\t', index_col = 0)
    info_list = [0] + list(stat_df.cumsum().T.values[0])    
    fr_txt = open(get_sys_path(md_fname, year, max_n), 'r').read().strip().split('\n')
    # fr_txt = open(get_testset_ref_path(max_n, year, level = 'doc'), 'r').read().strip().split('\n')
    fr_doc_list = []

    for i in range(len(info_list)-1):
        id0=info_list[i]
        id1=info_list[i+1]
        fr_doc_list.append(' '.join(fr_txt[id0:id1]))
    
    store_fpath = get_store_full_doc_path(md_fname, year, max_n)
    # print(store_fpath)
    with open(store_fpath, 'w', encoding = 'utf-8') as f:
        f.write( '\n'.join(fr_doc_list))


# 1. full doc
def recover_sent2doc(md_fname, year, max_n, get_ref_sep_path, get_sys_path, get_store_full_doc_path, sep_tag ='<sep>' ):
    # get_store_full_doc_path is a function
    full_txt = open(get_ref_sep_path( year), 'r').read().strip().split('\n')
    info_list = [0]
    for l in full_txt:
        info_list.append(info_list[-1] + len(l.split(sep_tag)))
            
    fr_txt = open(get_sys_path(md_fname, year, max_n), 'r').read().strip().split('\n')
    # fr_txt = open(get_testset_ref_path(max_n, year, level = 'doc'), 'r').read().strip().split('\n')
    fr_doc_list = []

    for i in range(len(info_list)-1):
        id0=info_list[i]
        id1=info_list[i+1]
        fr_doc_list.append(' '.join(fr_txt[id0:id1]))
    
    store_fpath = get_store_full_doc_path(md_fname, year, max_n)
    # print(store_fpath)
    with open(store_fpath, 'w', encoding = 'utf-8') as f:
        f.write( '\n'.join(fr_doc_list))


##############################################
# 2. ds-bleu
def eval_ds_bleu(sys_fpath, ref_fpath, store_dir):
    sbleu_path = f"{store_dir}/{os.path.basename(sys_fpath)}.ds-bleu.txt"
    eval_sacrebleu( 
        ref_fpath, 
        sys_fpath, 
        metric = 'bleu --sentence-level', 
        store_fpath = sbleu_path,
        )
    sbleu_df = get_sbleu_info(sbleu_path)
    sbleu_csv_path = f"{store_dir}/{os.path.basename(sys_fpath)}.ds-bleu.tsv"
    sbleu_df.to_csv(sbleu_csv_path , sep = '\t')
    return sbleu_csv_path

def eval_ds_bleu_TEDwindows(
        year_list, max_len_list, sys_fname_dict, eval_dir, get_testset_ref_path, get_store_full_doc_path,
        ):
    for k, md_fname in sys_fname_dict.items():     
        store_dir = f'{eval_dir}/{md_fname}'
        Path(store_dir).mkdir(parents=True, exist_ok=True)   
        for year in year_list:
            ref_fpath = get_testset_ref_path( year = year, level = 'doc')
            
            for max_n in max_len_list:
                sys_fpath = get_store_full_doc_path(md_fname, year, max_n)
                eval_ds_bleu(sys_fpath, ref_fpath, store_dir)


def get_ds_bleu_corpus_dict(sys_fname_dict, year_list, max_len_list,  get_store_full_doc_path_func, eval_dir):
    # TODO add option to choose output format (latex, md,...)
    res_table = {}
    for k, md_fname in sys_fname_dict.items():
        store_dir = f'{eval_dir}/{md_fname}'
        Path(store_dir).mkdir(parents=True, exist_ok=True)

        res_table[k] = {}
        for year in year_list:    
            res_table[k][year] = {}
            for max_n in max_len_list:
                ds_bleu_path = get_ds_bleu_path(md_fname, year, max_n, store_dir, get_store_full_doc_path_func)
                tmp_df =  pd.read_csv(ds_bleu_path, sep = '\t', index_col = 0)[['BLEU', 'BP']] 
                tmp = tmp_df.mean().to_dict()
                res_table[k][year][max_n] =  f"{round(tmp['BLEU'], 1)} \\footnotesize{{({round(tmp['BP'], 2)})}}"
      
    return res_table


##############################################
# 3. paired t test
# stats.ttest_rel(rvs1, rvs3)
# max_len_list = [ 'sent', 256, 512, 768, 1024, 1200, 1600, 2048, 'doc' ]
# year_list = ['2014', '2015', '2016', '2017']

# def get_ds_bleu_path(md_fname, year, max_n, store_dir, get_store_path_func):
#     sys_fpath =  get_store_path_func(md_fname, year, max_n)    
#     return f"{store_dir}/{os.path.basename(sys_fpath)}.ds-bleu.tsv"

def get_ds_bleu_path(md_fname, year, max_n, store_dir, get_store_path_func):
    sys_fpath =  get_store_path_func(md_fname, year, max_n)   
    if 'Qwen3.5' in md_fname or 'EuroLLM-22B' in md_fname:
        return f"{store_dir}/{os.path.basename(sys_fpath)}.{year}.{max_n}.ds-bleu.tsv"

    return f"{store_dir}/{os.path.basename(sys_fpath)}.ds-bleu.tsv"

def get_ds_bleu_dict(sys_fname_dict, year_list, max_len_list,  get_store_full_doc_path_func, eval_dir):
    # TODO add option to choose output format (latex, md,...)
    res_table = {}
    for k, md_fname in sys_fname_dict.items():
        store_dir = f'{eval_dir}/{md_fname}'
        Path(store_dir).mkdir(parents=True, exist_ok=True)
        res_table[k] = {}
        
        for year in year_list:    
            res_table[k][year] = {}
            for max_n in max_len_list:
                ds_bleu_path = get_ds_bleu_path(md_fname, year, max_n, store_dir, get_store_full_doc_path_func)
                res_table[k][year][max_n] =  pd.read_csv(ds_bleu_path, sep = '\t', index_col = 0)[['BLEU', 'BP']]
    
    
    ds_bleu_dict = {}
    for k, md_fname in sys_fname_dict.items():
        ds_bleu_dict[k] = {}
        for max_n in max_len_list:    
            tmp_ls = []
            for year in year_list:
                tmp_ls += list(res_table[k][year][max_n]['BLEU'].values) 
            ds_bleu_dict[k][max_n] = tmp_ls
    return ds_bleu_dict 


def get_test( key, max_len_list, ds_bleu_dict ):
    # TODO add option to choose output format (latex, md,...)
    res_table ={}
    for i in range(len(max_len_list)-1):
        l1 = max_len_list[i]
        l2 = max_len_list[i+1]
        
        ex1= ds_bleu_dict[key][l1]
        ex2= ds_bleu_dict[key][l2]
        stat_res = stats.ttest_rel(ex1, ex2)
        mean_diff = np.mean(np.array(ex1)-np.array(ex2))
        assert( (mean_diff < 0 and stat_res.statistic < 0) or  (mean_diff > 0 and stat_res.statistic > 0)  )
        print(mean_diff, stat_res)
        # res_table['TED-v2'][max_n] = round(stat_res.pvalue, 4)
        # res_table[f"{l1}-{l2}"] =f"{round(mean_diff, 1)} ({round(stat_res.pvalue, 2)})"
        res_table[f"{l1}-{l2}"] =f"{round(mean_diff, 1)} \\footnotesize{{({round(stat_res.pvalue, 2)})}}"
        # res_table[f"{l1}-{l2}"] =f"{round(mean_diff, 1)}" \\footnotesize{{({tmp_bp})}}"
        
        if stat_res.pvalue > 0.05:
            # res_table[f"{l1}-{l2}"] ="-" #f"- ({round(stat_res.pvalue, 4)})"
            res_table[f"{l1}-{l2}"] =f"{round(mean_diff, 1)} \\footnotesize{{({round(stat_res.pvalue, 2)})}}"

    return res_table

def get_test_compare_system( md1, md2, max_len_list, ds_bleu_dict ):
    res_table ={}
    for max_n in max_len_list:        
        ex1= ds_bleu_dict[md1][max_n]
        ex2= ds_bleu_dict[md2][max_n]
        
        stat_res = stats.ttest_rel(ex1, ex2)
        mean_diff = np.mean(np.array(ex1)-np.array(ex2))
        assert( (mean_diff < 0 and stat_res.statistic < 0) or  (mean_diff > 0 and stat_res.statistic > 0)  )
        print(mean_diff, stat_res)

        # res_table[max_n] =f"{round(mean_diff, 1)} ({round(stat_res.pvalue, 2)})"
        res_table[max_n] =f"{round(mean_diff, 1)} \\footnotesize{{({round(stat_res.pvalue, 2)})}}"
        
        if stat_res.pvalue > 0.05:
            res_table[max_n] ="-" #f"- ({round(stat_res.pvalue, 4)})"
    return res_table


########################
# COMET
def store_comet_score( md_fname, year, max_len_list, store_dir, store_fpath, get_comet_path_func, get_sys_path_func = None, mode = None):
    # store_dir from which to read comet-score output
    # store_fpath to store the results by system and by year
    comet_ls = []
    for max_n in max_len_list:
        tmp_dict = {}
        res_df = pd.read_json(get_comet_path_func(md_fname, year, max_n, store_dir, get_sys_path_func,  mode))
        for i in range(len(res_df)):
            tmp_dict[i] = res_df.loc[i].values[0]['COMET']
        comet_ls.append( pd.DataFrame(tmp_dict, index = [max_n]) )

    comet_df = pd.concat(comet_ls).T

    comet_df.to_csv(store_fpath, sep = '\t')
    return comet_df


def get_ds_comet_and_store(md_fname, year, get_ref_sep_path, store_dir, get_comet_tsv_path_func, sep_tag ='<sep>' ):
    # get_store_full_doc_path is a function
    full_txt = open(get_ref_sep_path( year), 'r').read().strip().split('\n')
    info_list = [0]
    for l in full_txt:
        info_list.append(info_list[-1] + len(l.split(sep_tag)))

    comet_df = pd.read_csv(get_comet_tsv_path_func(md_fname, year, store_dir, level = 'sent'), sep = '\t', index_col = 0)

    ds_comet_ls = []
    for i in range(len(info_list)-1):
        id0=info_list[i]
        id1=info_list[i+1]
        ds_comet_ls.append(comet_df[id0:id1].mean(axis = 0).to_frame().rename(columns = {0: i}))

    store_fpath = get_comet_tsv_path_func(md_fname, year, store_dir, level = 'doc')
    res_df = pd.concat(ds_comet_ls, axis = 1)
    res_df.to_csv(store_fpath, sep = '\t')
    
    return res_df


def get_comet_sent_dict(sys_fname_dict,year_list, store_dir, get_comet_tsv_path_func ):
    comet_sent_dict = {}
    for k, md_fname in sys_fname_dict.items():
        tmp_ls = []
        for year in year_list:
            comet_df = pd.read_csv(get_comet_tsv_path_func(md_fname, year, store_dir, level = 'sent'), sep = '\t', index_col = 0)
            tmp_ls.append( comet_df.mean().to_frame().rename(columns = {0:year}) )
        comet_sent_dict[k] = pd.concat(tmp_ls, axis = 1)
    return comet_sent_dict


# paired t test
# TODO
def get_ds_comet_dict(sys_fname_dict, year_list, max_len_list,  get_comet_tsv_path_func, store_dir):
    res_table = {}
    for k, md_fname in sys_fname_dict.items():
        # store_dir = f'{eval_dir}/{md_fname}'
        # Path(store_dir).mkdir(parents=True, exist_ok=True)
        
        res_table[k] = {}
        for year in year_list:    
            res_table[k][year] = {}
            
            ds_comet_path = get_comet_tsv_path_func(md_fname, year, store_dir, level = 'doc')
            res_table[k][year] =  pd.read_csv(ds_comet_path, sep = '\t', index_col = 0)
    
    ds_comet_dict = {}
    for k, md_fname in sys_fname_dict.items():
        ds_comet_dict[k] = {}
        for max_n in max_len_list:
    
            tmp_ls = []
            for year in year_list:
                tmp_ls += list(res_table[k][year].loc[str(max_n)].values) 
            ds_comet_dict[k][max_n] = tmp_ls
    return ds_comet_dict 


def get_test_comet( key, max_len_list, ds_comet_dict ):
    res_table ={}
    for i in range(len(max_len_list)-1):
        l1 = max_len_list[i]
        l2 = max_len_list[i+1]
        
        ex1= ds_comet_dict[key][l1]
        ex2= ds_comet_dict[key][l2]
        stat_res = stats.ttest_rel(ex1, ex2)
        mean_diff = np.mean(np.array(ex1)-np.array(ex2))
        
        assert( (mean_diff < 0 and stat_res.statistic < 0) or  (mean_diff > 0 and stat_res.statistic > 0)  )
        print(round(mean_diff, 2), stat_res)
        # res_table['TED-v2'][max_n] = round(stat_res.pvalue, 4)
        # res_table[f"{l1}-{l2}"] =f"{round(100*mean_diff, 1)} ({round(stat_res.pvalue, 2)})"
        res_table[f"{l1}-{l2}"] =f"{round(100*mean_diff, 1)} \\footnotesize{{({round(stat_res.pvalue, 2)})}}"
        
        # res_table[f"{l1}-{l2}"] = f"{round(100*mean_diff, 1)}"
        
        if stat_res.pvalue > 0.05:
            res_table[f"{l1}-{l2}"] =f"{round(100*mean_diff, 1)} \\footnotesize{{({round(stat_res.pvalue, 2)})}}"
            # res_table[f"{l1}-{l2}"] ="-" #f"- ({round(stat_res.pvalue, 4)})"
    return res_table


def get_test_compare_system_comet( md1, md2, max_len_list, ds_comet_dict ):
    res_table ={}
    for max_n in max_len_list:
        
        ex1= ds_comet_dict[md1][max_n]
        ex2= ds_comet_dict[md2][max_n]
        
        stat_res = stats.ttest_rel(ex1, ex2)
        mean_diff = np.mean(np.array(ex1)-np.array(ex2))
        assert( (mean_diff < 0 and stat_res.statistic < 0) or  (mean_diff > 0 and stat_res.statistic > 0)  )
        print(mean_diff, stat_res)

        # res_table[max_n] =f"{round(mean_diff, 1)} ({round(stat_res.pvalue, 2)})"
        res_table[max_n] =f"{round(100*mean_diff, 1)} \\footnotesize{{({round(stat_res.pvalue, 2)})}}"
        # if stat_res.pvalue > 0.05:
        #     res_table[max_n] ="-" 
    return res_table

        
####################################
# position bias
# collection of begin id 

# get stat (amount of avalaible samples)
def get_begin_id_stat_testsets( year_list, max_len_list, get_begin_id_path_func ):
    res_dict = {}
    for year in year_list:
        res_dict[year] = {}    
        for max_n in max_len_list :
            if max_n == 'sent':
                continue
            begin_id_df = pd.read_csv(get_begin_id_path_func(year, max_n), sep = '\t', index_col = 0)
            
            tmp_dict =  begin_id_df.to_dict()['begin_id']
            for i in range(len(begin_id_df)):
                res_dict[year][i] =res_dict[year].get(i, []) + [tmp_dict[i]]
    return res_dict



def count_begin_id_stat_total( year_list, max_len_list, get_begin_id_path_func, threshold = 7 ):
    res_dict = get_begin_id_stat_testsets( year_list, max_len_list, get_begin_id_path_func )
    total=0
    begin_id_stat_dict = {}
    
    for year in year_list:
        begin_id_stat_dict[year] = {}
        for i in range(len(res_dict[year])):
            begin_id_stat_dict[year][i] = len( set(res_dict[year][i]) )
        print(np.max(pd.DataFrame(begin_id_stat_dict[year], index = ['len']).values) )
    
        tmp_df = pd.DataFrame(begin_id_stat_dict[year], index = ['len'])
        tmp_df1 = tmp_df[tmp_df== threshold].T.dropna()
        
        print(tmp_df1.T)
        total += len(tmp_df1)
        print(year, len(tmp_df1))
        
    return total

#########################################################
# extract value

def extract_begin_id_stat( year_list, max_len_list, get_begin_id_path_func, threshold = 7 ):
    res_dict = get_begin_id_stat_testsets( year_list, max_len_list, get_begin_id_path_func )
    total=0
    begin_id_dict = {}
    
    for year in year_list:
        begin_id_dict[year] = []
        # for each sentence 
        for i in range(len(res_dict[year])):
            if len( set(res_dict[year][i])) >=threshold: 
                begin_id_dict[year].append( i)
    return  begin_id_dict
    
# get value (the comet scores of each exact (src, hypothesis) sentence pair)
def extract_begin_id_stat_testsets( year_list, max_len_list, get_begin_id_path_func, sent_id_dict ):
    # sent_id_dict contains the sentence idx to extract for each year of IWSLT (i.e. tst2014, tst2015...)
    # depend on `extract_begin_id_stat`
    res_dict = {}
    for year in year_list:
        res_dict[year] = {}    
        
        for max_n in max_len_list :
            if max_n == 'sent':
                continue
            begin_id_df = pd.read_csv(get_begin_id_path_func(year, max_n), sep = '\t', index_col = 0)
            
            tmp_dict =  begin_id_df.to_dict()['begin_id']
            for i in sent_id_dict[year]:
                res_dict[year][i] =res_dict[year].get(i, {})
                res_dict[year][i][tmp_dict[i]] = max_n
    return res_dict
    

def extract_comet_by_begin_id(  md_fname, year_list, max_len_list, begin_id_stat_dict, get_comet_tsv_path_func, store_dir ):
    # store_dir : where we stored the comet-score tsv (sentcen-level)
    # depend on the output of `extract_begin_id_stat_testsets`
    comet_dict = {}

    for year in year_list:
        comet_df = pd.read_csv(get_comet_tsv_path_func(md_fname, year, store_dir, level = 'sent'), sep = '\t', index_col = 0)
        comet_dict[year] = {}

        for i in begin_id_stat_dict[year].keys():
            comet_dict[year][i] = {}

            for begin_id, max_n in begin_id_stat_dict[year][i].items():
                comet_dict[year][i][begin_id] = comet_df.loc[i][str(max_n)]  

        # sort by begin_id
        for i in begin_id_stat_dict[year].keys():
            comet_dict[year][i] = {begin_id: comet_dict[year][i][begin_id] for begin_id in sorted(comet_dict[year][i].keys())}
            
    return comet_dict 



def get_comet_by_intervall(year_list, comet_dict, threshold, verbose = True):
    # comet_dict is the output of `extract_comet_by_begin_id`, computed with teh same threshold
    comet_by_begin_id_dict = {}
    begin_id_avg_dict = {}
    
    for i in range(threshold):
        comet_by_begin_id_dict[i] = []
        tmp_ls = []
        
        for year in year_list :
            for idx, val in comet_dict[year].items():
                # val is a dict of {begin_id : comet-score}            
                assert(sorted(val.keys()) == list(val.keys()))
                begin_id = list(val.keys())[i]
                comet_by_begin_id_dict[i] += [val[begin_id]]
                tmp_ls.append(begin_id)
                
        begin_id_avg_dict[i] = np.mean(tmp_ls)    
        if verbose:
            print(f"group {i}, average beginning position: {begin_id_avg_dict[i]}")   
    return comet_by_begin_id_dict, begin_id_avg_dict



# the main function for one MT model
def get_ttest_by_begin_id(comet_by_begin_id_dict, begin_id_avg_dict, threshold, filter_pvalue = None):
    # filter_pvalue = 0.05
    res_table ={}
    for i in range(threshold -1):
        ex1 = comet_by_begin_id_dict[i]
        ex2 = comet_by_begin_id_dict[i+1]
        stat_res = stats.ttest_rel(ex1, ex2)
        mean_diff = np.mean(np.array(ex1)-np.array(ex2))
        
        assert( (mean_diff < 0 and stat_res.statistic < 0) or  (mean_diff > 0 and stat_res.statistic > 0)  )
        print(round(mean_diff, 2), stat_res)
        # res_table['TED-v2'][max_n] = round(stat_res.pvalue, 4)
        l1 = int(begin_id_avg_dict[i])
        l2 = int(begin_id_avg_dict[i+1])
        
        # res_table[f"{l1}-{l2}"] =f"{round(100*mean_diff, 1)} ({round(stat_res.pvalue, 2)})"
        res_table[f"{l1}-{l2}"] =f"{round(100*mean_diff, 1)} \\footnotesize{{({round(stat_res.pvalue, 2)})}}"
        
        if filter_pvalue is not None and stat_res.pvalue > filter_pvalue:
            # res_table[f"{l1}-{l2}"] =f"{round(100*mean_diff, 1)} \\footnotesize{{({round(stat_res.pvalue, 2)})}}"
            res_table[f"{l1}-{l2}"] = '-'
            
    return res_table


# the main function for test sets
def get_ttest_testsets(
        sys_fname_dict, 
        year_list, 
        max_len_list, 
        begin_id_stat_dict, 
        get_comet_tsv_path, 
        store_dir, 
        threshold, 
        filter_pvalue = None,
        verbose = True,
        ):
    # depend on `extract_comet_by_begin_id`, `get_comet_by_intervall` and `get_ttest_by_begin_id`
    res_dict = {}
    for key, md_fname in sys_fname_dict.items():  
        comet_dict = extract_comet_by_begin_id(  md_fname, year_list, max_len_list, begin_id_stat_dict, get_comet_tsv_path, store_dir )
        
        comet_by_begin_id_dict, begin_id_avg_dict = get_comet_by_intervall(year_list, comet_dict, threshold, verbose)        
        res_dict[key] = get_ttest_by_begin_id(comet_by_begin_id_dict, begin_id_avg_dict, threshold, filter_pvalue)
    
    res_df = pd.DataFrame(res_dict)#.rename(columns = rename_dict)
    return res_df


