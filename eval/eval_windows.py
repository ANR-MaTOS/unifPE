# ==========================================
# Compute the BLEU score and ds-BLEU
# cf. the paper:
# [Investigating Length Issues in Document-level Machine Translation](https://aclanthology.org/2025.mtsummit-1.3/) (Peng et al., MTSummit 2025)
# see results in eval_TED_window.ipynb
# ==========================================

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


from utils_eval_bleu import (
    eval_sacrebleu, 
    eval_doc2sent,
    get_sbleu_info, 
    BP_PATTERN,
)
from data_paths import ( 
    DatasetPaths,
    SystemPaths,
)
from recover_full_doc import RecoverFullDoc


# ==========================================
# BLEU and ds-BLEU
# ==========================================


def eval_ds_bleu(sys_fpath, ref_fpath, tsv_store_fpath):
        
    # sbleu_path = f"{store_dir}/{os.path.basename(sys_fpath)}.ds-bleu.txt"
    # sbleu_csv_path = f"{store_dir}/{os.path.basename(sys_fpath)}.ds-b leu.tsv"
    txt_store_fpath = os.path.splitext( tsv_store_fpath)[0] + ".txt"
    eval_sacrebleu( 
        ref_fpath, 
        sys_fpath, 
        metric = 'bleu --sentence-level', 
        store_fpath = txt_store_fpath,
        )
    sbleu_df = get_sbleu_info(txt_store_fpath)
    sbleu_df.to_csv(tsv_store_fpath , sep = '\t')
    return tsv_store_fpath


class BleuEvaluator:
    def __init__(
            self, 
            dataset_paths: DatasetPaths, 
            system_paths: SystemPaths, 
            valid_max_n: set,
            corpus_name:str,
            sub_corpus_list = None,
            sep_tag:str = '<sep>',
            ):
        self.dataset_paths = dataset_paths
        self.system_paths = system_paths
        self.valid_max_n = valid_max_n

        self.valid_max_n_all = valid_max_n | {'sent', 'doc'}
        self.sub_corpus_list = sub_corpus_list if sub_corpus_list is not None else [ corpus_name ]
        self.recover_doc = RecoverFullDoc(self.dataset_paths, self.system_paths)
        self.sep_tag = sep_tag

    def get_sub_corpus_list(self):
        return self.sub_corpus_list

    def compute_bleu_aligned(self, model, max_n = 'sent', subset = None):    
        
        ref_fpath = self.dataset_paths.testset_ref_path("sent", subset)
        sys_fpath = self.system_paths.realign_path(model,  max_n, subset=subset)
        store_fpath = self.system_paths.get_bleu_path(model, max_n=max_n, subset = subset)

        eval_sacrebleu( ref_fpath, sys_fpath, metric = 'bleu', store_fpath = store_fpath)

    def read_bleu_json(self, bleu_fpath):
        score_df = pd.read_json(bleu_fpath, typ='series').to_frame().rename(index={'score': 'BLEU'}).T
        bp_match = re.match(BP_PATTERN, score_df['verbose_score'].values[0])
        score_df['BP'] = float(bp_match.group(1).strip())
        return score_df

    def compute_bleu(self, model, to_realign = True):
        """compute the vanilla BLEU scores"""
        bleu_dict = {}
        
        for subset in self.sub_corpus_list:
            bleu_dict[subset] = {}

            for max_n in self.valid_max_n_all:

                if max_n == 'sent': 
                    self.compute_bleu_aligned(model, max_n='sent', subset = subset)
                else:
                    if to_realign:
                        store_fpath = self.system_paths.get_bleu_path(model, max_n=max_n, subset = subset)
                        store_fpath.parent.mkdir(parents=True, exist_ok=True)
                        eval_doc2sent( 
                            sent_store_prefix = self.system_paths.realign_store_prefix( model, max_n),
                            sys_fpath = self.system_paths.sys_path(model, max_n, subset = subset),
                            ref_fpath = self.dataset_paths.sep_path(max_n, subset=subset), 
                            store_fpath = store_fpath, 
                            sep_tok = self.sep_tag,
                            )
                    else:
                        self.compute_bleu_aligned(model, max_n=max_n, subset = subset)

                store_fpath = self.system_paths.get_bleu_path(model, max_n=max_n, subset = subset)
                score_df = self.read_bleu_json(store_fpath)
                tmp_bleu = round(score_df['BLEU'].values[0], 1)
                tmp_bp = round(score_df['BP'].values[0], 2)
                bleu_dict[subset][max_n] = f"{tmp_bleu} ({tmp_bp})"
        return bleu_dict

    
    def get_bleu_corpus_dict(self, sys_fname_dict, with_bp = True):
        
        res_table = {}
        for subset in self.sub_corpus_list:
            res_table[subset ] = {}

            for k, model_fname in sys_fname_dict.items():
                res_table[subset ][k] = {}

                for max_n in self.valid_max_n_all:                    
                    store_fpath =  self.system_paths.get_bleu_path(model_fname, max_n=max_n, subset=subset)
                    score_df = self.read_bleu_json(store_fpath)

                    if with_bp:
                        tmp_bleu = round(score_df['BLEU'].values[0], 1)
                        tmp_bp = round(score_df['BP'].values[0], 2)
                        res_table[subset][k][max_n]  = f"{tmp_bleu} ({tmp_bp})"
                    else:
                        res_table[subset][k][max_n] =  score_df['BLEU'].values[0]
                    
        return res_table

    def sub_recover_doc(self, model, subset):
        # sent
        self.recover_doc.recover_from_sep(model, max_n= 'sent', sep_tag=self.sep_tag, subset=subset )
        # pseudo-documents
        for max_n in self.valid_max_n :
            self.recover_doc.recover_from_stats(model, max_n=max_n, subset=subset)        

    def eval_ds_bleu_corpus(self,  sys_fname_dict):
        for model_fname in sys_fname_dict.values():     
            for subset in self.sub_corpus_list:
                # reconstruct the full document for the pseudo-documents and the sentence-level corpus
                self.sub_recover_doc(model_fname, subset)

                ref_fpath = self.dataset_paths.testset_ref_path(max_n="doc", subset=subset)
                for max_n in self.valid_max_n_all:
                    sys_fpath = self.system_paths.full_doc_path(model_fname, max_n=max_n, subset=subset)

                    tsv_store_fpath = self.system_paths.get_ds_bleu_path(model_fname, max_n=max_n, subset=subset, file_type='tsv')
                    eval_ds_bleu(sys_fpath, ref_fpath, tsv_store_fpath )


    def get_ds_bleu_corpus_dict(self, sys_fname_dict, show_bp = True, small_bp_size = False):
        
        res_table = {}
        for subset in self.sub_corpus_list:    
            res_table[subset] = {}
            for k, model_fname in sys_fname_dict.items():
                res_table[subset][k] = {}
                
                for max_n in self.valid_max_n_all:

                    ds_bleu_path = self.system_paths.get_ds_bleu_path(model_fname,  max_n, subset )
                    tmp_df =  pd.read_csv(ds_bleu_path, sep = '\t', index_col = 0)[['BLEU', 'BP']] 
                    tmp = tmp_df.mean().to_dict()
                    if not show_bp:
                        res_table[subset][k][max_n] =  f"{round(tmp['BLEU'], 1)}"
                    else:
                        if small_bp_size:
                            res_table[subset][k][max_n] =  f"{round(tmp['BLEU'], 1)} \\footnotesize{{({round(tmp['BP'], 2)})}}"
                        else:
                            res_table[subset][k][max_n] =  f"{round(tmp['BLEU'], 1)} ({round(tmp['BP'], 2)})"
                    
        return res_table


    def get_ds_bleu_dict(self, sys_fname_dict):
        res_table = {}
        for k, model_fname in sys_fname_dict.items():
            res_table[k] = {}
            for subset in self.sub_corpus_list:    
                res_table[k][subset] = {}
                for max_n in self.valid_max_n_all:
                    ds_bleu_path = self.system_paths.get_ds_bleu_path(model_fname, max_n, subset, file_type='tsv')
                    res_table[k][subset][max_n] = pd.read_csv(ds_bleu_path, sep='\t', index_col=0)[['BLEU', 'BP']]
        
        ds_bleu_dict = {}
        for k in sys_fname_dict.keys():
            ds_bleu_dict[k] = {}
            for max_n in self.valid_max_n_all:
                tmp_ls = []
                for subset in self.sub_corpus_list:
                    tmp_ls += list(res_table[k][subset][max_n]['BLEU'].values) 
                ds_bleu_dict[k][max_n] = tmp_ls
        return ds_bleu_dict 


    def get_test(self,  ds_bleu_dict:dict, to_test_max_n:list,  verbose = False  ):
        assert(set(to_test_max_n).issubset(self.valid_max_n_all))
        to_test_max_n = list(to_test_max_n)

        res_table ={}
        for i in range(len(to_test_max_n)-1):
            l1 = to_test_max_n[i]
            l2 = to_test_max_n[i+1]
            
            ex1= ds_bleu_dict[l1]
            ex2= ds_bleu_dict[l2]
            stat_res = stats.ttest_rel(ex1, ex2)
            mean_diff = np.mean(np.array(ex1)-np.array(ex2))
            # Validation check on output vs statistic
            assert( (mean_diff < 0 and stat_res.statistic < 0) or  (mean_diff > 0 and stat_res.statistic > 0)  )

            # format
            if verbose:
                res_table[f"{l1}-{l2}"] = f"{round(mean_diff, 1)} \\footnotesize{{({round(stat_res.pvalue, 2)})}}"
            else:
                res_table[f"{l1}-{l2}"] =f"{round(mean_diff, 1)}"
                
                if stat_res.pvalue > 0.05:
                    res_table[f"{l1}-{l2}"] ="-" 
            
            if stat_res.pvalue <= 0.05 and stat_res.pvalue > 0.01:            
                res_table[f"{l1}-{l2}"] =f"\\mark{{{res_table[f"{l1}-{l2}"]}}}"

        return res_table

    def get_test_compare_system(self, md1, md2,  ds_bleu_dict, verbose = False, show_p_val = True ):
        res_table ={}
        for max_n in self.valid_max_n:
            
            ex1= ds_bleu_dict[md1][max_n]
            ex2= ds_bleu_dict[md2][max_n]
            
            stat_res = stats.ttest_rel(ex1, ex2)
            mean_diff = np.mean(np.array(ex1)-np.array(ex2))
            assert( (mean_diff < 0 and stat_res.statistic < 0) or  (mean_diff > 0 and stat_res.statistic > 0)  )
            if verbose:
                print(mean_diff, stat_res)

            if show_p_val:
                
                # res_table[max_n] =f"{round(mean_diff, 1)} \\footnotesize{{({round(stat_res.pvalue, 2)})}}"
                res_table[max_n] =f"\\textbf{{{round(mean_diff, 1)} \\footnotesize{{({round(stat_res.pvalue, 2)})}}}}"
                
            else:
                res_table[max_n] =f"{round(mean_diff, 1)}"
            
            if stat_res.pvalue > 0.05:
                if verbose:
                    res_table[max_n] = f"{round(mean_diff, 1)} \\footnotesize{{({round(stat_res.pvalue, 2)})}}"
                else:
                    res_table[max_n] ="-" #f"- ({round(stat_res.pvalue, 4)})"
        return res_table



# ==========================================
# COMET
# ==========================================

class CometEvaluator:
    def __init__(
            self, 
            dataset_paths: DatasetPaths, 
            system_paths: SystemPaths, 
            valid_max_n:list[int],
            corpus_name:str,
            sub_corpus_list: list[str]|None,
            sep_tag:str = '<sep>',
            ):
        self.dataset_paths = dataset_paths
        self.system_paths = system_paths
        self.valid_max_n = valid_max_n
        self.valid_max_n_all = valid_max_n | {'sent', 'doc'}

        self.sub_corpus_list = sub_corpus_list if sub_corpus_list is not None else [ corpus_name ]
        self.sep_tag = sep_tag

    def get_sub_corpus_list(self):
        return self.sub_corpus_list

    def store_comet_score(self, model_fname, subset=None, is_json = False):
        # if the score is stored in json file with sentence pairs
        comet_ls = []
        for max_n in self.valid_max_n_all:
            if is_json:
                tmp_dict = {}
                res_df = pd.read_json(self.system_paths.get_comet_path(model_fname, max_n, subset))
                for i in range(len(res_df)):
                    tmp_dict[i] = res_df.loc[i].values[0]['COMET']
                tmp_df = pd.DataFrame(tmp_dict, index=[max_n])
            else:
                fpath = self.system_paths.get_comet_path( model_fname, max_n, subset)
                tmp_df = pd.read_csv(fpath, sep = '\t', index_col = 0).rename(columns = {str(0): max_n }).T

            comet_ls.append(tmp_df)

        comet_df = pd.concat(comet_ls).T

        store_fpath = self.system_paths.get_comet_tsv_path(model_fname, subset=subset,  level = 'sent')
        comet_df.to_csv(store_fpath, sep='\t')
        return comet_df


    def get_ds_comet_and_store(self, model_fname, subset ):
        # get_store_full_doc_path is a function
        ref_sep_path = self.dataset_paths.sep_path(max_n= 'doc', subset = subset, lang = 'fr')
        with open(ref_sep_path, 'r') as f:
            full_txt = f.read().strip().split('\n')

        info_list = [0]
        for l in full_txt:
            info_list.append(info_list[-1] + len(l.split(self.sep_tag)))

        sent_comet_path = self.system_paths.get_comet_tsv_path(model_fname, level = 'sent', subset=subset)
        comet_df = pd.read_csv(sent_comet_path, sep = '\t', index_col = 0)

        # compute ds-comet
        ds_comet_ls = []
        for i in range(len(info_list)-1):
            id0=info_list[i]
            id1=info_list[i+1]
            ds_comet_ls.append(comet_df[id0:id1].mean(axis = 0).to_frame().rename(columns = {0: i}))

        res_df = pd.concat(ds_comet_ls, axis = 1)

        store_fpath = self.system_paths.get_comet_tsv_path(model_fname, level = 'doc', subset=subset)
        res_df.to_csv(store_fpath, sep = '\t')
        
        return res_df

    def get_comet_sent_dict(self, sys_fname_dict ):
        comet_sent_dict = {}
        for k, model_fname in sys_fname_dict.items():
            tmp_ls = []
            for subset in self.sub_corpus_list:
                sent_comet_path = self.system_paths.get_comet_tsv_path(model_fname, level = 'sent', subset=subset)
                comet_df = pd.read_csv(sent_comet_path, sep = '\t', index_col = 0)
                tmp_ls.append( comet_df.mean().to_frame().rename(columns = {0: subset}) )

            comet_sent_dict[k] = pd.concat(tmp_ls, axis = 1)
        return comet_sent_dict

    def get_ds_comet_dict(self, sys_fname_dict):
        res_table = {}
        for k, model_fname in sys_fname_dict.items():

            res_table[k] = {}
            for subset in self.sub_corpus_list:
                res_table[k][subset] = {}
                
                ds_comet_path =  self.system_paths.get_comet_tsv_path(model_fname, level = 'doc', subset=subset)
                res_table[k][subset] =  pd.read_csv(ds_comet_path, sep = '\t', index_col = 0)
        
        # compute ds-comet
        ds_comet_dict = {}
        for k, model_fname in sys_fname_dict.items():
            ds_comet_dict[k] = {}
            for max_n in self.valid_max_n_all:
        
                tmp_ls = []
                for subset in self.sub_corpus_list:
                    tmp_ls += list(res_table[k][subset].loc[str(max_n)].values) 
                ds_comet_dict[k][max_n] = tmp_ls
        return ds_comet_dict 


    def get_test_comet(self,  ds_comet_dict, to_test_max_n, show_p_val = True ):
        assert(set(to_test_max_n).issubset(self.valid_max_n_all))
        to_test_max_n = list(to_test_max_n)
        
        res_table ={}
        for i in range(len(to_test_max_n )-1):
            l1 = to_test_max_n[i]
            l2 = to_test_max_n[i+1]
            
            ex1= ds_comet_dict[l1]
            ex2= ds_comet_dict[l2]
            stat_res = stats.ttest_rel(ex1, ex2)
            mean_diff = np.mean(np.array(ex1)-np.array(ex2))
            
            assert( (mean_diff < 0 and stat_res.statistic < 0) or  (mean_diff > 0 and stat_res.statistic > 0)  )
            print(f"{l1}-{l2}", round(100*mean_diff, 1), stat_res)

            # format
            if show_p_val :
                # res_table[f"{l1}-{l2}"] =f"{round(100*mean_diff, 1)} ({round(stat_res.pvalue, 2)})"
                res_table[f"{l1}-{l2}"] =f"{round(100*mean_diff, 1)} \\footnotesize{{({round(stat_res.pvalue, 2)})}}"
            else:    
                res_table[f"{l1}-{l2}"] = f"{round(100*mean_diff, 1)}"
                if stat_res.pvalue <= 0.05 and stat_res.pvalue> 0.01:
                    res_table[f"{l1}-{l2}"] = f"\\mark{{{round(100*mean_diff, 1)}}}"

            if stat_res.pvalue > 0.05:
                res_table[f"{l1}-{l2}"] ="-" #f"- ({round(stat_res.pvalue, 4)})"
        return res_table


    def get_test_compare_system_comet(self, md1, md2, ds_comet_dict,  verbose = False, show_p_val = True  ):
        res_table ={}
        for max_n in self.valid_max_n:

            ex1= ds_comet_dict[md1][max_n]
            ex2= ds_comet_dict[md2][max_n]
            
            stat_res = stats.ttest_rel(ex1, ex2)
            mean_diff = np.mean(np.array(ex1)-np.array(ex2))
            assert( (mean_diff < 0 and stat_res.statistic < 0) or  (mean_diff > 0 and stat_res.statistic > 0)  )

            # format
            if show_p_val:
                # res_table[max_n] =f"{round(mean_diff, 1)} ({round(stat_res.pvalue, 2)})"
                res_table[max_n] =f"\\textbf{{{round(100*mean_diff, 1)} \\footnotesize{{({round(stat_res.pvalue, 2)})}}}}"
            else:
                res_table[max_n] =f"{round(100*mean_diff, 1)}"
            
            if stat_res.pvalue > 0.05:
                if verbose:
                    res_table[max_n] =f"{round(100*mean_diff, 1)} \\footnotesize{{({round(stat_res.pvalue, 2)})}}"
                else:
                    
                    res_table[max_n] ="-" 
        return res_table

