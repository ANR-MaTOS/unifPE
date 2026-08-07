import pandas as pd

from data_paths import ( 
    DatasetPaths,
    SystemPaths,
)


class RecoverFullDoc:

    def __init__(self, dataset_paths:DatasetPaths, system_paths:SystemPaths):
        self.dataset_paths = dataset_paths
        self.system_paths = system_paths

    @staticmethod
    def _write_documents(info_list, sent_path, output_path):
        """Recover document-level translations from sentence translations."""

        sentences = sent_path.read_text(encoding="utf-8").strip().splitlines()

        docs = [
            " ".join(sentences[info_list[i]:info_list[i + 1]])
            for i in range(len(info_list) - 1)
        ]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(docs), encoding="utf-8")

    def recover_from_stats(self, model, max_n, subset = None, **kwargs):
        """
        Recover document translations from pseudo-document outputs.
        """

        stat_df = pd.read_csv(
            self.dataset_paths.stat_path(max_n=max_n, subset = subset, **kwargs), sep="\t", index_col=0,
        )

        info_list = [0] + stat_df.cumsum().iloc[:, 0].tolist()

        self._write_documents(
            info_list,
            self.system_paths.sys_path(model, max_n=max_n, subset = subset,  **kwargs),
            self.system_paths.full_doc_path(model, max_n=max_n, subset = subset, **kwargs),
        )

    def recover_from_sep(self, model,  sep_tag="<sep>",  subset = None,  **kwargs):
        """
        Recover document translations from sentence translations.
        """
 
        ref_lines = self.dataset_paths.sep_path(max_n= 'doc', subset=subset).read_text(encoding="utf-8").strip().splitlines()

        info_list = [0]
        for l in ref_lines:
            info_list.append(info_list[-1] + len(l.split(sep_tag)))

        self._write_documents(
            info_list,
            self.system_paths.sys_path(model, max_n='sent', subset = subset),
            self.system_paths.full_doc_path(model, max_n='sent', subset = subset),
        )




# if __name__ == "__main__":
#     recoverer = RecoverFullDoc(mersenne_dataset, mersenne_system)

#     recoverer.recover_from_stats(
#         model="Llama-3.1-8B-Instruct",
#         max_n=2048,
#     )

#     recoverer.recover_from_sep(
#         model="Llama-3.1-8B-Instruct",
#     )