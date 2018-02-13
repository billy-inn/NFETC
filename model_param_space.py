import numpy as np
from hyperopt import hp

param_space_nfetc = {
    "wpe_dim": hp.quniform("wpe_dim", 5, 100, 5),
    "lr": hp.qloguniform("lr", np.log(1e-4), np.log(1e-2), 1e-4),
    "state_size": hp.quniform("state_size", 100, 500, 10),
    "hidden_layers": 0,
    "hidden_size": 0,
    "dense_keep_prob": hp.quniform("dense_keep_prob", 0.5, 1, 0.1),
    "rnn_keep_prob": hp.quniform("rnn_keep_prob", 0.5, 1, 0.1),
    "l2_reg_lambda": hp.quniform("l2_reg_lambda", 0, 1e-3, 1e-4),
    "batch_size": 512,
    "num_epochs": 20,
    "alpha": 0.3,
}

param_space_best_nfetc_wiki = {
    "wpe_dim": 85,
    "lr": 0.0002,
    "state_size": 180,
    "hidden_layers": 0,
    "hidden_size": 0,
    "dense_keep_prob": 0.7,
    "rnn_keep_prob": 0.9,
    "l2_reg_lambda": 0.0000,
    "batch_size": 512,
    "num_epochs": 20,
    "alpha": 0.0,
}

param_space_best_nfetc_wiki_hier = {
    "wpe_dim": 85,
    "lr": 0.0002,
    "state_size": 180,
    "hidden_layers": 0,
    "hidden_size": 0,
    "dense_keep_prob": 0.7,
    "rnn_keep_prob": 0.9,
    "l2_reg_lambda": 0.0000,
    "batch_size": 512,
    "num_epochs": 20,
    "alpha": 0.4,
}

param_space_best_nfetc_ontonotes = {
    "wpe_dim": 20,
    "lr": 0.0002,
    "state_size": 440,
    "hidden_layers": 0,
    "hidden_size": 0,
    "dense_keep_prob": 0.5,
    "rnn_keep_prob": 0.5,
    "l2_reg_lambda": 0.0001,
    "batch_size": 512,
    "num_epochs": 20,
    "alpha": 0.0,
}

param_space_best_nfetc_ontonotes_hier = {
    "wpe_dim": 20,
    "lr": 0.0002,
    "state_size": 440,
    "hidden_layers": 0,
    "hidden_size": 0,
    "dense_keep_prob": 0.5,
    "rnn_keep_prob": 0.5,
    "l2_reg_lambda": 0.0001,
    "batch_size": 512,
    "num_epochs": 20,
    "alpha": 0.3,
}

param_space_dict = {
    "nfetc": param_space_nfetc,
    "best_nfetc_wiki": param_space_best_nfetc_wiki,
    "best_nfetc_wiki_hier": param_space_best_nfetc_wiki_hier,
    "best_nfetc_ontonotes": param_space_best_nfetc_ontonotes,
    "best_nfetc_ontonotes_hier": param_space_best_nfetc_ontonotes_hier,
}

int_params = [
    "wpe_dim", "state_size", "batch_size", "num_epochs", "hidden_size", "hidden_layers",
]

class ModelParamSpace:
    def __init__(self, learner_name):
        s = "Wrong learner name!"
        assert learner_name in param_space_dict, s
        self.learner_name = learner_name

    def _build_space(self):
        return param_space_dict[self.learner_name]

    def _convert_into_param(self, param_dict):
        if isinstance(param_dict, dict):
            for k, v in param_dict.items():
                if k in int_params:
                    param_dict[k] = int(v)
                elif isinstance(v, list) or isinstance(v, tuple):
                    for i in range(len(v)):
                        self._convert_into_param(v[i])
                elif isinstance(v, dict):
                    self._convert_into_param(v)
        return param_dict
