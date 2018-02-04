# NFETC: Neural Fine-grained Entity Type Classification with Hierarchy-Aware Loss

### Prerequisites

- TensorFlow >= r1.0
- hyperopt

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
