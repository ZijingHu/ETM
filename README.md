# Embedded Topic Model

ETM was originally published by Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei on a article titled ["Topic Modeling in Embedding Spaces"](https://arxiv.org/abs/1907.04907) in 2019. This code is an adaptation of the [original](https://github.com/adjidieng/ETM) provided with the article. Most of the original code was kept here, with some changes here and there, mostly for ease of usage.

# Usage

```
from TopicModel import *
from Word2Vec import *

w2v_path = './path/GoogleNews-vectors-negative300.bin'
w2v = Word2Vec(w2v_path, limit=15000, stop_words=True)

seed = 1

model_com_args = {
    'V': w2v.V,
    'H': 512,
    'T': 8,
    'E': w2v.E,
    'activation': 'relu',
    'dropout_rate': 0.2
}

optimizer_com_args = {
    'optimizer_name': 'adam',
    'lr': 0.001,
    'wdecay': 1.2e-6
}


training_com_args = {
    'batch_size': 512,
    'epoch_nums': 20,
    'normalized': True,
    'check': False,
    'model_path': 'etm_model',
    'load_model': False,
    'save_model': 10000
}


train_set = 
test_set = 

device = torch.device('cuda:0')

set_random_seed(seed)
etm = ETM(**{**model_com_args, **model_sp_args})
optimizer = get_optimizer(etm.parameters(), **{**optimizer_com_args, **optimizer_sp_args})
etm_train(etm, w2v, train_set, test_set, optimizer, device, 
          **{**training_com_args, **training_sp_args})
```