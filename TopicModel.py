import re
import os
import copy
import time
import torch
import random
import pickle
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from gensim.models import LdaModel
from sklearn.metrics.pairwise import cosine_similarity

from .Parameters import Parameters

class ETM(nn.Module):
    '''Topic Modeling in Embedding Spaces

    INTERFACE
    ---------
    self.__init__
      V [int] # of words
      H [int] # of hidden layer dimensions
      T [int] # of topics
      E [int] # of embedding layer dimensions
      args [dict] activation function, dropout rate

    self.encode
      bows [torch.tensor] bag of words: # of docs * V

    self.forward
      bows [torch.tensor] bag of words: # of docs * V
      normalized [bool] normalize bows if true

    self.get_topics => top k words for each topic [numpy.array]
      w2v [script.etm.Word2Vec] Word2Vec model
      k [int] top k words chosen from each topic to represent the topic
      output [bool] return the top k words if true

    self.alphas => topic embeddings
    self.beta => topic embeddings * word embedding

    self.train_model => ETM model
      w2v [script.etm.Word2Vec] Word2Vec model
      train_set [list] list of docs
      test_set [list] list of docs
      batch_size [int]
      optimizer [torch.optim.{optimizer}]
      device [torch.device]
      epoch_nums [int] 
      model_path [str] 
      load_model [bool] 
      save_model [int] save the model after {save_model} batches

    REFERENCE
    ---------
    Topic Modeling in Embedding Spaces [Dieng et al. 2019]
    Auto-Encoding Variational Bayes [Kingma and Welling 2013] (KW2013)
    An Introduction to Variational Autoencoders  [Kingma and Welling 2019] (KW2019)
    '''
    def __init__(self, V, H, T, E, activation, dropout_rate, 
                 pretrained_weight=False, rho_grad=True, word2vec_model=None):
        super(ETM, self).__init__()
        # input => hidden state
        word2vec_model = copy.deepcopy(word2vec_model)
        self._q_theta = nn.Sequential(
            nn.Linear(V, H),
            nn.Dropout(dropout_rate),
            get_activation(activation),
            nn.Linear(H, H),
            nn.Dropout(dropout_rate),
            get_activation(activation)
        )
        for l in self._q_theta:
            if type(l) == nn.Linear:
                nn.init.kaiming_normal_(l.weight, nonlinearity=activation)
                l.bias.data.fill_(0)

        # hidden state => mu, log(sigma^2)
        self._mu_q_theta = nn.Linear(H, T, bias=True)
        self._mu_q_theta.bias.data.fill_(0)
        nn.init.kaiming_normal_(self._mu_q_theta.weight, nonlinearity=activation)

        self._log_sigma2_q_theta = nn.Linear(H, T, bias=True)
        self._log_sigma2_q_theta.bias.data.fill_(0)
        nn.init.kaiming_normal_(self._log_sigma2_q_theta.weight, nonlinearity=activation)

        # rho & alpha(s)
        self._rho = nn.Linear(E, V, bias=False) # shape of weight matrix: V * E
        if pretrained_weight and word2vec_model is None:
            raise TypeError('word2vec_model is None.')
        elif pretrained_weight and word2vec_model is not None:
            v, e = word2vec_model.weight.shape
            assert (v, e) == (V, E), f'pre-trained weight: expect ({V}, {E}) but get ({v}, {e}).'
            self._rho.weight = nn.Parameter(word2vec_model.weight)
            self._rho.weight.requires_grad = rho_grad
        else:
            nn.init.xavier_normal_(self._rho.weight)
            if not rho_grad:
                print("No pre-trained weights, rho_grad is set to True")
        self._alphas = nn.Linear(E, T, bias=False)
        nn.init.orthogonal_(self._alphas.weight)

    def _reparameterize(self, mu, log_sigma2):
        # KW2013: Section 2.4
        if self.training:
            sigma = torch.exp(0.5 * log_sigma2) 
            epsilon = torch.randn_like(sigma)
            return mu + sigma * epsilon
        else:
            return mu

    def encode(self, bows):
        # KW2019
        # input => hidden state
        q_theta = self._q_theta(bows)
        # hidden state => mu, log(sigma^2)
        mu_theta = self._mu_q_theta(q_theta)
        log_sigma2_theta = self._log_sigma2_q_theta(q_theta)
        # KW2013: Section 3 & Appendix B
        kld_theta = -0.5 * torch.sum(1+log_sigma2_theta-mu_theta.pow(2)-log_sigma2_theta.exp(), dim=-1).mean()
        # KW2013: Section 2.4
        z = self._reparameterize(mu_theta, log_sigma2_theta)
        theta = nn.functional.softmax(z, dim=-1)
        return theta, kld_theta

    def decode(self, theta):
        # KW2019
        beta = nn.functional.softmax(self._alphas(self._rho.weight), dim=0).transpose(1, 0)
        res = torch.mm(theta, beta)
        pred = torch.log(res+torch.full_like(res, 1e-6))
        return pred

    def forward(self, bows, normalized=True):
        # encoding
        if normalized: 
            normalized_bows = bows / bows.sum(1, keepdim=True)
            normalized_bows[normalized_bows!=normalized_bows] = 0
            theta, kld_theta = self.encode(normalized_bows)
        else: 
            theta, kld_theta = self.encode(bows)
        # decoding
        pred = self.decode(theta)
        # compute loss
        recon_loss = -(pred*bows).sum(1).mean()
        return recon_loss, kld_theta

    def get_topics(self, w2v, k=10, output=False):
        alphas = self.alphas.tolist()
        wx = self._rho.weight.to('cpu').tolist()
        sim = np.argsort(cosine_similarity(alphas, wx))[:, -k:]
        rep_words = list(map(lambda x: [w2v.inv_vocab[i] for i in x], sim))
        print('\n'.join([f'  Topic {i+1}: '+', '.join(words) for i, words in enumerate(rep_words)]))
        if output:
            return rep_words

    @property
    def alphas(self):
        return self._alphas.weight
        
    @property
    def beta(self):
        beta = nn.functional.softmax(self._alphas(self._rho.weight), dim=0).transpose(1, 0)
        return beta

    def train_model(self, w2v, train_set, test_set, optimizer, device, batch_size,
              epoch_nums, model_path, load_model, save_model, 
              normalized=True, check=True):
        # check
        if check:
            flag = input('Start[y/n]?')
            assert flag in ('y', 'n')
            if flag == 'n': return

        # [SL] model path
        print('[Init]')
        if os.path.exists(model_path):
            print(f'  Model path "{model_path}" exists.')
        else:
            os.mkdir(model_path)
            print(f'  Create {model_path}.')
            if load_model == True:
                print(f"  Set load_model to False as the model path doesn't exist")
                load_model = False

        # [SL] load model
        if os.path.exists(os.path.join(model_path, 'training_state.pkl')) and load_model:
            print('  LOAD MODEL: {load_model}')
            with open(os.path.join(model_path, 'training_state.pkl'), 'rb') as file:
                training_state = pickle.load(file)
                epoch_n = training_state['epoch_num']
            with open(os.path.join(model_path, 'train_dataloader.pkl'), 'rb') as file:
                train_dataloader = pickle.load(file)
            checkpoint = torch.load(os.path.join(model_path, f'EP{epoch_n}.torch'))
            self.load_state_dict(checkpoint['model_state'])
        # [SL] new model
        else:
            training_state = {'epoch_num': 0, 'step_num': 0}
            train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            with open(os.path.join(model_path, 'train_dataloader.pkl'), 'wb') as file:
                pickle.dump(train_dataloader, file)

        length = len(train_dataloader)
        test_dataloader = DataLoader(test_set, batch_size=batch_size)
        self.to(device)
        t = time.time()
        for epoch in range(epoch_nums):
            start = time.time()
            t0 = start
            loss_sum = []
            i = 0

            # [SL] check breakpoint (epoch)
            if training_state['step_num'] > 0: 
                training_state['epoch_num'] -= 1
            epoch_n = training_state['epoch_num'] + epoch + 1

            # [TRAIN & EVAL]
            self.train()
            for data in train_dataloader:
                # [SL] check breakpoint (batch)
                if i <= training_state['step_num']:
                    i += 1
                    continue
                else:
                    training_state['step_num'] = 0

                # [TRAIN & EVAL]
                bows = w2v.corpus2bows(data).to(device)
                optimizer.zero_grad()
                recon_loss, kld_theta = self(bows, normalized=normalized)
                # orthogonal regularization
                loss_orth = (self.alphas @ self.alphas.T / (self.alphas ** 2).sum(1)).mean()
                loss = recon_loss + kld_theta + loss_orth
                loss.backward()
                loss_sum.append(loss.item())
                optimizer.step()

                # [SL] save the model regularly (batch)
                if i % save_model == 0:
                    if i == save_model:
                        print('\n  checkpoint: ', end='')
                    t1 = time.time() - t0
                    print(f'{i} {int(t1)}s; ', end='' if length-i>=save_model else '\n')
                    t0 = time.time()
                    torch.save({'model_state': self.state_dict()},
                               os.path.join(model_path, f'EP{epoch_n}.torch'))
                    with open(os.path.join(model_path, 'training_state.pkl'), 'wb') as file:
                        pickle.dump({'epoch_num': epoch_n, 'step_num': i}, file)  
                i += 1
            t1 = time.time() - start
            print(f'[Epoch {epoch_n}] {int(t1)}s; Completed...', end='\r') 
            
            # [SL] save the model regularly (epoch)
            torch.save({'model_state': self.state_dict()},
                       os.path.join(model_path, f'model.torch' if epoch_n == epoch_nums else f'EP{epoch_n}.torch'))
            last_train = os.path.join(model_path, f'EP{epoch_n-1}.torch')
            if os.path.exists(last_train):
                os.remove(last_train)
            train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            with open(os.path.join(model_path, 'train_dataloader.pkl'), 'wb') as file:
                pickle.dump(train_dataloader, file)
            with open(os.path.join(model_path, 'training_state.pkl'), 'wb') as file:
                pickle.dump({'epoch_num': epoch_n, 'step_num': 0}, file)

        print()
        t1 = time.time() - t
        print(f'[Summary]\n  Total time spent: {int(t1)}s;')
        print('[Evaluation]')
        beta = self.beta.cpu().detach().numpy()
        td, tc = evaluation(beta, train_set, w2v.inv_vocab)
        print(f'  [Train] TD: {td:.4f}; TC: {tc:.4f};')
        td, tc = evaluation(beta, test_set, w2v.inv_vocab)
        print(f'  [Test] TD: {td:.4f}; TC: {tc:.4f};')
        print('[Top 10 Words]')
        self.get_topics(w2v, 10)
    

class LDA(object):
    '''Topic Modeling in Embedding Spaces

    INTERFACE
    ---------
    self.train
      w2v [script.etm.Word2Vec] Word2Vec model
      train [list] list of docs
      num_topics [int]
    
    self.get_topics => top k words for each topic [numpy.array]
      k [int] top k words chosen from each topic to represent the topic
      output [bool] return the top k words if true
    
    self.train_model => LDA model
      w2v [script.etm.Word2Vec] Word2Vec model
      train [list] list of docs
      num_topics [int]
    '''
    def __init__(self):
        self.model = None
        
    def get_topics(self, k=10, output=False):
        rep_words = [re.findall('"([a-z]+)"', i[1]) for i in self.model.print_topics(k)]
        print('\n'.join([f'  Topic {i+1}: '+', '.join(words) for i, words in enumerate(rep_words)]))
        if output:
            return rep_words

    @staticmethod
    def _doc2bow(doc, w2v):
        corpus_whole = zip(range(w2v.V), w2v.corpus2bows([doc]).int().tolist()[0])
        return list(filter(lambda x: x[1]!=0, corpus_whole))


    def train_model(self, w2v, train_set, num_topics, model_path):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        t = time.time()
        corpus = [self._doc2bow(doc, w2v) for doc in train_set]
        self.model = LdaModel(corpus=corpus, id2word=w2v.inv_vocab, num_topics=num_topics)
        with open(os.path.join(model_path, 'model.pkl'), 'wb') as file:
            pickle.dump(self.model, file)
        t1 = time.time() - t
        print(f'[Summary]\n  Total time spent: {int(t1)}s;')
        print('[Top 10 Words]')
        self.get_topics(10)        


def etm_train_new(params, data, w2v, device='GPU'):
    if not os.path.exists(params['training']['model_path']):
        os.mkdir(params['training']['model_path'])
    params.save(params['training']['model_path'])
    set_random_seed(params['seed'])
    word2vec_model = w2v if params['model']['pretrained_weight'] else None
    etm = ETM(word2vec_model=word2vec_model, **params['model'])
    optimizer = get_optimizer(etm.parameters(), **params['potimizer'])
    device_ = torch.device('cuda:0') if device=='GPU' else torch.device('cpu')
    etm.train_model(w2v, data['train'], data['test'], optimizer, device_, **params['training'])
    return etm


def etm_load(model_path, w2v=None, model_name='model'):
    params = Parameters(load=model_path)
    word2vec_model = w2v if params['model']['pretrained_weight'] else None
    etm = ETM(word2vec_model=word2vec_model, **params['model'])
    if os.path.exists(os.path.join(model_path, 'training_state.pkl')):
        checkpoint = torch.load(os.path.join(model_path, f'{model_name}.torch'))
        etm.load_state_dict(checkpoint['model_state'])
        print('Successfully load the model.')
    else:
        print('Invalid model path.')
    return etm


def lda_train_new(params, data, w2v):
    if not os.path.exists(params['training']['model_path']):
        os.mkdir(params['training']['model_path'])
    set_random_seed(params['seed'])
    lda = LDA()
    lda.train_model(w2v, data['train'], params['model']['T'], params['training']['model_path'])
    return lda


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    

def get_activation(act):
    if act == 'tanh': 
        return nn.Tanh()
    if act == 'relu': 
        return nn.ReLU()
    if act == 'softplus': 
        return nn.Softplus()
    if act == 'rrelu': 
        return nn.RReLU()
    if act == 'leakyrelu':  
        return nn.LeakyReLU()
    if act == 'elu': 
        return nn.ELU()
    if act == 'selu': 
        return nn.SELU()
    if act == 'glu': 
        return nn.GLU()
    raise ValueError('Invalid activation function.')


def get_optimizer(param, optimizer_name, lr, wdecay=None):
    if optimizer_name == 'sgd':
        return optim.SGD(param, lr=lr)
    assert wdecay is not None
    if optimizer_name == 'adam':
        return optim.Adam(param, lr=lr, weight_decay=wdecay)
    if optimizer_name == 'adagrad':
        return optim.Adagrad(param, lr=lr, weight_decay=wdecay)
    if optimizer_name == 'adadelta':
        return optim.Adadelta(param, lr=lr, weight_decay=wdecay)
    if optimizer_name == 'rmsprop':
        return optim.RMSprop(param, lr=lr, weight_decay=wdecay)
    if optimizer_name == 'asgd':
        return optim.ASGD(param, lr=lr, weight_decay=wdecay, t0=0, lambd=0.)
    raise ValueError('Invalid optimizer.')


def doc_freq(data, wi, wj=None):
    if wj:
        n_wj, n_wi_wj = 0, 0
        for doc in data:
            if wj in doc:
                n_wj += 1
                if wi in doc:
                    n_wi_wj += 1
        return n_wj, n_wi_wj
    else:
        n_wi = 0
        for doc in data:
            if wi in doc:
                n_wi += 1
        return n_wi    


def topic_coherence(beta, data, vocab, topk=10):
    # Dieng et al. 2019
    n = len(data)
    tc = []
    num_topics = len(beta)
    for k in range(num_topics):
        top_k = list(beta[k].argsort()[-topk:][::-1])
        tc_k = 0
        counter = 0
        for i, word in enumerate(top_k):
            n_wi = doc_freq(data, vocab[word])
            j = i + 1
            tmp = 0
            while j < len(top_k) and j > i:
                n_wj, n_wi_wj = doc_freq(data, vocab[word], vocab[top_k[j]])
                mut_info = (np.log(n_wi)+np.log(n_wj)-2.0*np.log(n)) / (np.log(n_wi_wj)-np.log(n)) if n_wi_wj else 0
                f_wi_wj = -1 + mut_info
                tmp += f_wi_wj
                j += 1
                counter += 1
            tc_k += tmp 
        tc.append(tc_k)
    tc = np.mean(tc) / counter
    return tc


def topic_diversity(beta, topk=10):
    # Dieng et al. 2019
    num_topics = beta.shape[0]
    list_w = np.zeros((num_topics, topk))
    for k in range(num_topics):
        idx = beta[k,:].argsort()[-topk:][::-1]
        list_w[k,:] = idx
    n_unique = len(np.unique(list_w))
    td = n_unique / (topk * num_topics)
    return td


def evaluation(beta, data, vocab, topk=10):
    td = topic_diversity(beta, topk)
    tc = topic_coherence(beta, data, vocab, topk)
    return td, tc