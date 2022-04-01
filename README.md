# Embedded Topic Model

ETM was originally published by Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei on a article titled ["Topic Modeling in Embedding Spaces"](https://arxiv.org/abs/1907.04907) in 2019. This code is an adaptation of the [original](https://github.com/adjidieng/ETM) provided with the article. Most of the original code was kept here, with some changes here and there, mostly for ease of usage.

# Usage

```
from TopicModel import *
from Word2Vec import *
from Parameters import *

w2v_path = './path/GoogleNews-vectors-negative300.bin'
w2v = Word2Vec(w2v_path, limit=15000, stop_words=True)

data = {
    'train': ,
    'test': 
}

default = {
    'seed': 1,
    'model': {
        'V': w2v.V,
        'H': 512,
        'T': 8,
        'E': w2v.E,
        'activation': 'relu',
        'dropout_rate': 0.2,
        'pretrained_weight': False, 
        'rho_grad': True
    },
    'potimizer': {
        'optimizer_name': 'adam',
        'lr': 0.002,
        'wdecay': 1.2e-6
    },
    'training': {
        'batch_size': 256,
        'epoch_nums': 20,
        'normalized': True,
        'check': False,
        'model_path': 'etm_model',
        'load_model': False,
        'save_model': 10000
    }
}

params = Parameters(default)

# ETM
T = params['model']['T']
params['training'].update({'model_path': f'MODEL_ETM_T{T}_W2V_FIXED'})
params['model'].update({'pretrained_weight': True, 'rho_grad': False})

etm = etm_train_new(params, data, w2v)

# LDA
T = params['model']['T']
params['training'].update({'model_path': f'MODEL_LDA_T{T}'})

lda = lda_train_new(params, data, w2v)
```