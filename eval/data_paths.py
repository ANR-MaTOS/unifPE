from abc import ABC, abstractmethod
from pathlib import Path

import os

class DatasetPaths(ABC):
    """Abstract interface for dataset paths."""

    def __init__(self, root, valid_max_n):
        self.root = Path(root)
        self.valid_max_n = set(valid_max_n)
        self.valid_max_n_all = self.valid_max_n | {'sent', 'doc'}
        print(f"Supported max_n values: {self.valid_max_n_all}")
    
    def get_valid_max_n(self):
        return self.valid_max_n
    
    @abstractmethod
    def stat_path(self, max_n, subset=None, **kwargs):
        pass

    @abstractmethod
    def testset_ref_path(self, max_n, subset=None, **kwargs):
        pass

    def begin_id_path(self, tokenizer_name, max_n, subset=None, **kwargs):
        sep_path = self.sep_path( max_n = max_n, subset = subset, lang="en")
        store_dir = f"{os.path.dirname(sep_path)}/begin_id"
        store_fpath = f"{store_dir}/{os.path.basename(sep_path)}.{tokenizer_name}.begin_id.tsv" 

        return store_fpath
    
    @abstractmethod
    def sep_path(self, max_n = 'doc', subset = None, **kwargs):
        pass



class SystemPaths(ABC):
    """Interface for MT system output paths."""

    def __init__(self, root, valid_max_n):
        self.root = Path(root)
        self.valid_max_n = set(valid_max_n)
        self.valid_max_n_all = self.valid_max_n | {'sent', 'doc'}
        print(f"Supported max_n values: {self.valid_max_n_all}")

    def _check(self, max_n):
        if max_n not in self.valid_max_n_all:
            print(max_n)
            raise ValueError(
                f"max_n must be one of {sorted(self.valid_max_n_all)}."
            )

    def get_valid_max_n(self):
        return self.valid_max_n
    
    @abstractmethod
    def sys_path(self, model_type, max_n, subset=None,  **kwargs):
        pass

    @abstractmethod
    def realign_store_prefix(self, model_type, max_n, subset=None,  **kwargs):
        pass

    def realign_path(self, model_type, max_n, subset=None,  **kwargs):
        res = f"{self.realign_store_prefix(model_type, max_n, subset=subset)}.sent.sys"
        return res

    @abstractmethod
    def full_doc_path(self, model_type, max_n, subset=None,  **kwargs):
        pass


    def get_ds_bleu_path(self, model_type,  max_n, subset = None, file_type = 'tsv', **kwargs):
        # to make sure that the output path is different for different values of max_n, and subset
        store_dir = self.root / "score-bleu" / model_type
        store_dir.mkdir(parents=True, exist_ok=True)
    
        sys_fpath =  self.full_doc_path(model_type, max_n, subset)    
        return store_dir / f"{os.path.basename(sys_fpath)}.{subset}{max_n}.ds-bleu.{file_type}"
    
    def get_bleu_path(self, model_type, max_n, subset=None,  **kwargs):
        store_dir = self.root / "score-bleu" / model_type
        store_dir.mkdir(parents=True, exist_ok=True)
        
        sys_fpath =  self.sys_path(model_type, max_n, subset)

        if sys_fpath.name == "generated.txt":
            subset_name = sys_fpath.parent.name
            bleu_path = store_dir / f"{subset_name}.sent.bleu.json"
        else:
            bleu_path = store_dir / f"{sys_fpath.name}.sent.bleu.json"                

        return bleu_path 

    def get_comet_path(self, model_type, max_n, subset=None,  **kwargs):
        raise NotImplementedError

    def get_comet_tsv_path(self, model_type,  level, subset = None, **kwargs):
        assert(level in ['sent', 'doc'])
        raise NotImplementedError


############################################
class TEDPaths(DatasetPaths):

    # VALID_MAX_N = {256, 512, 768, 1024, 1200, 1600, 2048}

    def __init__(self, root ):
        super().__init__(
            root,
            valid_max_n= {256, 512, 768, 1024, 1200, 1600, 2048},
            )

    def stat_path(self, max_n, subset):
        return self.root / f"pseudo_doc_test{subset}" / f"pseudo_doc_max{max_n}.nb_doc.stat.tsv"


    def testset_ref_path(self, max_n="doc", subset="2014"):
        if max_n not in self.valid_max_n_all:
            raise ValueError(max_n)

        if max_n == "doc":
            return self.root / "doc" / f"TED_test{subset}_doc.fr"

        if max_n == "sent":
            return self.root / "sent" / f"TED_test{subset}_sent.fr"

        return self.root  / f"pseudo_doc_test{subset}" / f"pseudo_doc_max{max_n}.fr"

    def sep_path(self, max_n = 'doc', subset="2014", lang="fr"):
        if max_n not in self.valid_max_n_all or max_n == 'sent':
            raise ValueError(f"No <sep> file for max_n={max_n}")

        if max_n == 'doc':
            return self.root / "doc" / f"TED_test{subset}_doc_sep.{lang}"
        
        return self.root / f"pseudo_doc_test{subset}/pseudo_doc_max{max_n}_sep.en"
        

############################################
class TEDSystemPaths(SystemPaths):

    def __init__(self, root):
        super().__init__(
            root,
            valid_max_n= {256, 512, 768, 1024, 1200, 1600, 2048},
        )
        
    def sys_path(self, model_type, max_n, subset):
        self._check(max_n)
        store_dir = self.root / model_type

        if max_n in {"doc", "sent"}:
            return  store_dir / f"mt_TED_test{subset}_{max_n}" / "generated.txt" 
            # return self.root / model_type / f"mt_TED_test{subset}_sent.sys"

        return store_dir / f"TED_window_pseudo_doc_test{subset}_max{max_n}" / "generated.txt" 


    def realign_store_prefix(self, model_type, max_n, subset='2014'):
        store_dir = self.root  / model_type / "realign-sents"
        store_dir.mkdir(parents=True, exist_ok=True)

        if max_n == "doc":
            return store_dir / f"mt_TED_test{subset}_doc.sent"

        return store_dir / f"TED_window_pseudo_doc_test{subset}_max{max_n}.sent"

    def realign_path(self, model_type,  max_n, subset):
        self._check(max_n)

        if max_n == "sent":
            return self.sys_path(model_type, max_n,  subset)

        return self.realign_store_prefix(model_type, max_n, subset)


    def full_doc_path(self, model_type, max_n, subset):
        self._check(max_n)

        if max_n == "doc":
            return self.sys_path(model_type, max_n, subset)
           
        recover_dir = self.root / "recover-doc"
        recover_dir.mkdir(parents=True, exist_ok=True)

        if max_n == "sent":
            return recover_dir / f"mt_TED_test{subset}_sent.sys.full_doc"

        res = recover_dir / f"TED_window_pseudo_doc_test{subset}_max{max_n}.fulldoc.txt"

        return res
        


    def get_comet_path(self, model_type, max_n, subset = None):
        eval_dir = self.root / "TED-comet-score"
        if max_n == 'sent':
            return f"{eval_dir}/{model_type}/tst{subset}/TED_sent/comet.tsv"
        
        res_dir = f"{eval_dir}/{model_type}/tst{subset}/TED_max{max_n}_sent/comet.tsv"
        return res_dir

    def get_comet_tsv_path(self, model_type, subset, level = 'sent'):
        eval_dir = self.root / "TED-comet-score"
        if level == 'sent':
            return f"{eval_dir}/{os.path.basename(model_type)}/tst{subset}/comet-score-year{subset}.tsv"

        if level == 'doc':
            return f"{eval_dir}/{os.path.basename(model_type)}/tst{subset}/comet-score-year{subset}-fulldoc.tsv"


############################################
class StudentPaths(DatasetPaths):

    # VALID_MAX_N = {512, 1024, 1536, 2048, 3072, 4096}

    def __init__(self, root):
        super().__init__(
            root,
            valid_max_n= {512, 1024, 1536, 2048, 3072, 4096}
        )
        self.pseudo_dir = self.root / "pseudo_doc_STUDENT"

    def stat_path(self, max_n, subset=None):
        return self.pseudo_dir / f"pseudo_doc_max{max_n}.nb_doc.stat.tsv"


    def testset_ref_path(self, max_n, subset=None):
        if max_n not in self.valid_max_n_all:
            raise ValueError(max_n)
        if max_n == "doc":
            return self.root / "STUDENT_article.fr"

        if max_n == "sent":
            return self.root / "STUDENT_sent.fr"

        return self.pseudo_dir / f"pseudo_doc_max{max_n}.fr"

    def sep_path(self, max_n, subset = None, lang="fr"):
        if max_n not in self.valid_max_n_all or max_n == 'sent':
            raise ValueError(f"No <sep> file for max_n={max_n}")
        
        if max_n == "doc":
            return self.root / "doc_auxiliaire" / f"STUDENT_article_sep.{lang}"

        return self.pseudo_dir / f"pseudo_doc_max{max_n}_sep.{lang}"


############################################
class StudentSystemPaths(SystemPaths):
    
    def __init__(self, root):
        super().__init__(
            root,
            valid_max_n =  {512, 1024, 1536, 2048, 3072, 4096},
            )
        

    def sys_path(self, model_type, max_n, subset=None):
        self._check(max_n)

        store_dir = self.root / model_type

        if max_n == "sent":
            return store_dir / "mt_STUDENT_sent" / "generated.txt"

        if max_n == "doc":
            return store_dir / "mt_STUDENT_article" / "generated.txt"

        return   store_dir / f"STUDENT_window_pseudo_doc_STUDENT_max{max_n}" / "generated.txt"


    def realign_store_prefix(self, model_type, max_n):
        self._check(max_n)

        store_dir = self.root / "realign-sents" / model_type
        store_dir.mkdir(parents=True, exist_ok=True)

        return  store_dir / f"STUDENT_max{max_n}"


    def realign_path(self, model_type,  max_n, subset=None):
        if max_n == "sent":
            return self.sys_path(model_type, max_n,  subset=subset)
        return f"{self.realign_store_prefix(model_type, max_n)}.sent.sys"

    
    def full_doc_path(self, model_type, max_n, subset=None):
        self._check(max_n)

        recover_dir = self.root / "recover-doc"
        recover_dir.mkdir(parents=True, exist_ok=True)

        if max_n == "doc":
            return self.sys_path(model_type, "doc")

        if max_n == "sent":
            return  recover_dir / model_type / "mt_STUDENT_sent.fulldoc.txt"

        return  recover_dir / model_type / f"STUDENT_window_pseudo_doc_STUDENT_max{max_n}.fulldoc.txt"

    
    def get_comet_path(self, model_type, max_n, subset = None):
        
        eval_dir = self.root / "STUDENT-comet-score"
        if max_n == 'sent':
            return f"{eval_dir}/{model_type}/STUDENT_sent/comet.tsv"
        
        res_dir = f"{eval_dir}/{model_type}/STUDENT_max{max_n}_sent/comet.tsv"
        return res_dir

    def get_comet_tsv_path(self, model_type, subset=None, level = 'sent'):
        eval_dir = self.root / "STUDENT-comet-score"
        if level == 'sent':
            return f"{eval_dir}/{model_type}/comet-score-STUDENT.tsv"

        if level == 'doc':
            return f"{eval_dir}/{model_type}/comet-score-STUDENT-fulldoc.tsv"


############################################
class MersennePaths(DatasetPaths):

    # VALID_MAX_N = {1024, 1536, 2048, 3072, 4096}

    def __init__(self, root):
        super().__init__(
            root,
            valid_max_n = {512, 1024, 1536, 2048, 3072, 4096},
            )
        self.pseudo_dir = self.root / "pseudo_doc_MERSENNE-GEOS"

    def stat_path(self, max_n, subset=None):
        return self.pseudo_dir / f"pseudo_doc_max{max_n}.nb_doc.stat.tsv"

    def testset_ref_path(self, max_n, subset=None):
        if max_n not in self.valid_max_n_all:
            raise ValueError(max_n)

        if max_n == "doc":
            return self.root / "MERSENNE_article.fr"

        if max_n == "sent":
            return self.root / "MERSENNE_sent.fr"

        return self.pseudo_dir / f"pseudo_doc_max{max_n}.fr"

    def sep_path(self, max_n, subset = None,  lang="fr"):
        if max_n not in self.valid_max_n_all or max_n == 'sent':
            raise ValueError(f"No <sep> file for max_n={max_n}")

        if max_n == "doc":
            return self.root / "doc_auxiliaire" / f"MERSENNE_article_sep.{lang}"

        return self.pseudo_dir / f"pseudo_doc_max{max_n}_sep.{lang}"


############################################
class MersenneSystemPaths(SystemPaths):

    # VALID_MAX_N = {1024, 1536, 2048, 3072, 4096}

    def __init__(self, root):
        super().__init__(
            root,
            valid_max_n = {512, 1024, 1536, 2048, 3072, 4096}
            )


    def sys_path(self, model_type, max_n, subset=None):
        self._check(max_n)

        store_dir = self.root / model_type

        if max_n == "sent":
            return store_dir / "mt_MERSENNE_sent" / "generated.txt"

        if max_n == "doc":
            return store_dir / "mt_MERSENNE_article" / "generated.txt"

        return   store_dir / f"MERSENNE_window_pseudo_doc_MERSENNE-GEOS_max{max_n}" / "generated.txt"


    def realign_store_prefix(self, model_type, max_n):
        self._check(max_n)

        store_dir = self.root / "realign-sents" / model_type
        store_dir.mkdir(parents=True, exist_ok=True)

        return  store_dir / f"MERSENNE-GEOS_max{max_n}"


    def realign_path(self, model_type,  max_n, subset=None):
        if max_n == "sent":
            return self.sys_path(model_type, max_n,  subset=subset)
        
        return f"{self.realign_store_prefix(model_type, max_n)}.sent.sys"

    
    def full_doc_path(self, model_type, max_n, subset=None):
        self._check(max_n)

        recover_dir = self.root / "recover-doc"
        recover_dir.mkdir(parents=True, exist_ok=True)

        if max_n == "doc":
            return self.sys_path(model_type, "doc")

        if max_n == "sent":
            return  recover_dir / model_type / "mt_MERSENNE_sent.fulldoc.txt"

        return  recover_dir / model_type / f"MERSENNE_window_pseudo_doc_MERSENNE-GEOS_max{max_n}.fulldoc.txt"

    def get_comet_path(self, model_type, max_n, subset = None):
        eval_dir = self.root / "MERSENNE-GEOS-comet-score"
        if max_n == 'sent':
            return f"{eval_dir}/{model_type}/MERSENNE-GEOS_sent/comet.tsv"
        
        res_dir = f"{eval_dir}/{model_type}/MERSENNE-GEOS_max{max_n}_sent/comet.tsv"
        return res_dir

    def get_comet_tsv_path(self, model_type, subset=None, level = 'sent'):
        eval_dir = self.root / "MERSENNE-GEOS-comet-score"
        if level == 'sent':
            return f"{eval_dir}/{model_type}/comet-score-MERSENNE-GEOS.tsv"

        if level == 'doc':
            return f"{eval_dir}/{model_type}/comet-score-MERSENNE-GEOS-fulldoc.tsv"

