# NFETC

Neural Fine-grained Entity Type Classification with Hierarchy-Aware Loss

Paper Published in NAACL 2018: [NFETC](https://arxiv.org/abs/1803.03378)

### Prerequisites

- tensorflow >= r1.2
- hyperopt
- gensim
- sklearn
- pandas

### Dataset

Run `./download.sh` to download the corpus and the pre-trained word embeddings

### Preprocessing

Run `python preprocess.py -d <data_name> [ -c ]` to preprocess the data.

Available Dataset Name:

- **wiki**: Wiki/FIGER(GOLD) with original freebase-based hierarchy
- **ontonotes**: ONTONOTES
- **wikim**: Wiki/FIGER(GOLD) with improved hierarchy

Use `-c` to control if filter the data or not

#### Note about wikim

Before preprocessing, you need to:

1. Create a folder `data/wikim` to store data for Wiki with the improved hierarchy
2. Run `python transform.py`

### Hyperparameter Tuning

Run `python task.py -m <model_name> -d <data_name> -e <max_evals> -c <cv_runs>`

See `model_param_space.py` for available model name

The searching procedurce is recorded in one log file stored in folder `log`

### Evaluation

Run `python eval.py -m <model_name> -d <data_name> -r <runs>`

The scores for each run and the average scores are also recorded in one log file stored in folder `log`

### Cite

If you found this codebase or our work useful, please cite:

```
@InProceedings{xu2018neural,
  author = {Xu, Peng and Barbosa, Denilson},
  title = {Neural Fine-Grained Entity Type Classification with Hierarchy-Aware Loss},
  booktitle = {The 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL 2018)},
  month = {June},
  year = {2018},
  publisher = {ACL}
}
```
