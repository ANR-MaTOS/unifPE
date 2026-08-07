# ==========================================
# Compute the position bias
# Table 5 of the paper:
# [Investigating Length Issues in Document-level Machine Translation](https://aclanthology.org/2025.mtsummit-1.3/) (Peng et al., MTSummit 2025)
# see results in eval_TED_window.ipynb
# ==========================================
import pandas as pd
from scipy import stats
import numpy as np

from data_paths import ( 
    DatasetPaths,
    SystemPaths
)

class PositionBiasAnalyzer:
    def __init__(
            self, 
            dataset_paths: DatasetPaths, 
            system_paths: SystemPaths, 
            valid_max_n:set,
            corpus_name:str,
            sub_corpus_list,
            ):
        self.dataset_paths = dataset_paths
        self.system_paths = system_paths
        self.valid_max_n = valid_max_n
        self.valid_max_n_all = valid_max_n | {'sent', 'doc'}
        self.sub_corpus_list = sub_corpus_list if sub_corpus_list is not None else [ corpus_name ]

    def get_sub_corpus_list(self):
        return self.sub_corpus_list
    
    def get_begin_id_stat_testsets(self, tokenizer_name, disable_full_doc = True):
        """get stat (amount of avalaible samples) """
        to_count_max_n = self.valid_max_n
        if not disable_full_doc:
            to_count_max_n = self.valid_max_n | {'doc'}
            assert 'sent' not in to_count_max_n

        res_dict = {}
        for subset in self.sub_corpus_list: 
            res_dict[subset] = {}    
            for max_n in to_count_max_n :
                begin_id_df = pd.read_csv(self.dataset_paths.begin_id_path(tokenizer_name, max_n, subset), sep = '\t', index_col = 0)
                
                tmp_dict =  begin_id_df.to_dict()['begin_id']
                for i in range(len(begin_id_df)):
                    res_dict[subset][i] =res_dict[subset].get(i, []) + [tmp_dict[i]]
        return res_dict

    def count_begin_id_stat_total(self, tokenizer_name, nb_positions = 7, disable_full_doc = True) :
        res_dict = self.get_begin_id_stat_testsets(tokenizer_name, disable_full_doc)
        total=0
        begin_id_stat_dict = {}
        
        for subset in self.sub_corpus_list :
            begin_id_stat_dict[subset] = {}
            for i in range(len(res_dict[subset])):
                begin_id_stat_dict[subset][i] = len( set(res_dict[subset][i]) )
            
            tmp_df = pd.DataFrame(begin_id_stat_dict[subset], index = ['len'])
            tmp_df1 = tmp_df[tmp_df== nb_positions].T.dropna()
            
            total += len(tmp_df1)
            print(f"{subset}, {len(tmp_df1)}, maximum different positions = {np.max(pd.DataFrame(begin_id_stat_dict[subset], index = ['len']).values)}" )
            
        return total

    def extract_begin_id_stat(self, tokenizer_name, nb_positions = 7 , disable_full_doc = True) :
        """extract values"""
        res_dict = self.get_begin_id_stat_testsets(tokenizer_name, disable_full_doc)
        begin_id_dict = {}
        
        for subset in self.sub_corpus_list :
            begin_id_dict[subset] = []
            # for each sentence 
            for i in range(len(res_dict[subset])):
                if len( set(res_dict[subset][i])) == nb_positions: 
                    begin_id_dict[subset].append( i)
        return  begin_id_dict

    
    def extract_begin_id_stat_testsets(self,  sent_id_dict, tokenizer_name, disable_full_doc = True ):
        """get value (the comet scores of each exact (src, hypothesis) sentence pair)"""
        # sent_id_dict contains the sentence idx to extract for each subset of IWSLT (i.e. tst2014, tst2015...)
        # depend on `extract_begin_id_stat`
        to_count_max_n = self.valid_max_n
        if not disable_full_doc:
            to_count_max_n = self.valid_max_n | {'doc'}
            assert 'sent' not in to_count_max_n

        res_dict = {}
        for subset in self.sub_corpus_list :
            res_dict[subset] = {}    
            
            for max_n in to_count_max_n:
                begin_id_df = pd.read_csv(self.dataset_paths.begin_id_path(tokenizer_name, max_n, subset), sep = '\t', index_col = 0)
                
                tmp_dict =  begin_id_df.to_dict()['begin_id']
                for i in sent_id_dict[subset]:
                    res_dict[subset][i] =res_dict[subset].get(i, {})
                    res_dict[subset][i][tmp_dict[i]] = max_n
        return res_dict


    def extract_comet_by_begin_id(self,  md_fname,  begin_id_stat_dict ):
        # store_dir : where we stored the comet-score tsv (sentcen-level)
        # depend on the output of `extract_begin_id_stat_testsets`
        comet_dict = {}

        for subset in self.sub_corpus_list :
            comet_df = pd.read_csv( self.system_paths.get_comet_tsv_path(md_fname, level = 'sent', subset=subset ), sep = '\t', index_col = 0)
            comet_dict[subset] = {}

            for i in begin_id_stat_dict[subset].keys():
                comet_dict[subset][i] = {}

                for begin_id, max_n in begin_id_stat_dict[subset][i].items():
                    comet_dict[subset][i][begin_id] = comet_df.loc[i][str(max_n)]  

            # sort by begin_id
            for i in begin_id_stat_dict[subset].keys():
                comet_dict[subset][i] = {begin_id: comet_dict[subset][i][begin_id] for begin_id in sorted(comet_dict[subset][i].keys())}
                
        return comet_dict 


    def get_comet_by_intervall(self, comet_dict, nb_positions):
        """comet_dict is the output of `extract_comet_by_begin_id`, computed with teh same nb_positions"""
        comet_by_begin_id_dict = {}
        begin_id_avg_dict = {}
        
        for i in range(nb_positions):
            comet_by_begin_id_dict[i] = []
            tmp_ls = []
            
            for subset in self.sub_corpus_list :
                for idx, val in comet_dict[subset].items():
                    # val is a dict of {begin_id : comet-score}            
                    assert(sorted(val.keys()) == list(val.keys()))
                    begin_id = list(val.keys())[i]
                    comet_by_begin_id_dict[i] += [val[begin_id]]
                    tmp_ls.append(begin_id)
                    
            begin_id_avg_dict[i] = np.mean(tmp_ls)    
            # print(f"group {i}, average beginning position: {begin_id_avg_dict[i]}")   
        return comet_by_begin_id_dict, begin_id_avg_dict

    
    def get_ttest_by_begin_id(self, comet_by_begin_id_dict, begin_id_avg_dict, nb_positions, show_p_val = False):
        """ the main function for one MT model """
        res_table ={}
        for i in range(nb_positions -1):
            ex1 = comet_by_begin_id_dict[i]
            ex2 = comet_by_begin_id_dict[i+1]
            stat_res = stats.ttest_rel(ex1, ex2)
            mean_diff = np.mean(np.array(ex1)-np.array(ex2))
            
            assert( (mean_diff < 0 and stat_res.statistic < 0) or  (mean_diff > 0 and stat_res.statistic > 0)  )
            # print(round(mean_diff, 2), stat_res)
            l1 = int(begin_id_avg_dict[i])
            l2 = int(begin_id_avg_dict[i+1])
            
            if show_p_val:
                res_table[f"{l1}-{l2}"] =f"{round(100*mean_diff, 1)} \\footnotesize{{({round(stat_res.pvalue, 2)})}}"
            else:
                res_table[f"{l1}-{l2}"] = f"{round(100*mean_diff, 1)}"
                if stat_res.pvalue <= 0.05 and stat_res.pvalue> 0.01:
                    res_table[f"{l1}-{l2}"] = f"\\mark{{{round(100*mean_diff, 1)}}}"
                    # res_table[f"{l1}-{l2}"] = round(100*mean_diff, 1)         
            
            if stat_res.pvalue > 0.05:
                res_table[f"{l1}-{l2}"] = '-'
                
        return res_table

    
    def get_ttest_testsets(self, sys_fname_dict,  begin_id_stat_dict,  nb_positions, show_p_val = False):
        """ the main function for test sets
        depend on `extract_comet_by_begin_id`, `get_comet_by_intervall` and `get_ttest_by_begin_id`
        """
        print("Comparing positon bias, make sure that models in sys_fname_dict share the same tokenizer")
        res_dict = {}
        for key, md_fname in sys_fname_dict.items():  
            comet_dict = self.extract_comet_by_begin_id( md_fname,  begin_id_stat_dict )
            
            comet_by_begin_id_dict, begin_id_avg_dict = self.get_comet_by_intervall( comet_dict, nb_positions)      
            print([ int(v) for _, v in begin_id_avg_dict.items() ])      
            res_dict[key] = self.get_ttest_by_begin_id(comet_by_begin_id_dict, begin_id_avg_dict, nb_positions, show_p_val)
        
        res_df = pd.DataFrame(res_dict) #.rename(columns = rename_dict)
        return res_df

