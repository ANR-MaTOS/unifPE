# Investigating Length Issues in Document-level Machine Translation

Scripts for the paper 

>Ziqian Peng, Rachel Bawden, and François Yvon. 2025. Investigating Length Issues in Document-level Machine Translation. In Proceedings of Machine Translation Summit XX: Volume 1, pages 4–23, Geneva, Switzerland. European Association for Machine Translation.

The `eval` folder contains scripts for evaluating length issues and position bias in document-level machine translation. The `eval/eval_length_issues.ipynb` notebook demonstrates how to use these scripts.

Complete scripts for unifPE, model fine-tuning, and related tasks will be available soon.

```
@inproceedings{peng-etal-2025-investigating,
    title = "Investigating Length Issues in Document-level Machine Translation",
    author = "Peng, Ziqian  and
      Bawden, Rachel  and
      Yvon, Fran{\c{c}}ois",
    editor = "Bouillon, Pierrette  and
      Gerlach, Johanna  and
      Girletti, Sabrina  and
      Volkart, Lise  and
      Rubino, Raphael  and
      Sennrich, Rico  and
      Farinha, Ana C.  and
      Gaido, Marco  and
      Daems, Joke  and
      Kenny, Dorothy  and
      Moniz, Helena  and
      Szoc, Sara",
    booktitle = "Proceedings of Machine Translation Summit XX: Volume 1",
    month = jun,
    year = "2025",
    address = "Geneva, Switzerland",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2025.mtsummit-1.3/",
    pages = "4--23",
    ISBN = "978-2-9701897-0-1",
    abstract = "Transformer architectures are increasingly effective at processing and generating very long chunks of texts, opening new perspectives for document-level machine translation (MT). In this work, we challenge the ability of MT systems to handle texts comprising up to several thousands of tokens. We design and implement a new approach designed to precisely measure the effect of length increments on MT outputs. Our experiments with two representative architectures unambiguously show that (a) translation performance decreases with the length of the input text; (b) the position of sentences within the document matters and translation quality is higher for sentences occurring earlier in a document. We further show that manipulating the distribution of document lengths and of positional embeddings only marginally mitigates such problems. Our results suggest that even though document-level MT is computationally feasible, it does not yet match the performance of sentence-based MT."
}
```