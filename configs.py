import os

data_folder = '../NER/data/for_ncrf'

datasets = {
    'gold_morpheme': {
        '_unit': 'morpheme',
        '_scheme': 'bioes',
        'train_dir': 'morph_gold_train.bmes',
        'dev_dir': 'morph_gold_dev.bmes',
        'test_dir': 'morph_gold_test.bmes', 
    },
    'gold_token_bioes': {
        '_unit': 'token',
        '_scheme': 'bioes',
        'train_dir': 'token_gold_train_fix.bmes',
        'dev_dir': 'token_gold_dev_fix.bmes',
        'test_dir': 'token_gold_test_fix.bmes',
    },
    'gold_token_raw': {
        '_unit': 'token',
        '_scheme': 'concat_bioes',
        'seg': False,
        'train_dir': 'token_gold_train_concat.bmes',
        'dev_dir': 'token_gold_dev_concat.bmes',
        'test_dir': 'token_gold_test_concat.bmes',
    },
    'yap_morpheme': {
        '_unit': 'morpheme',
        '_scheme': 'bioes',
        'dev': 'morph_yap_dev.bmes',
        'test': 'morph_yap_test.bmes',
    },
    'yap_morpheme_pruned_all': {
        '_unit': 'morpheme',
        '_scheme': 'bioes',
        'dev': 'morph_yap_dev_pruned_all.bmes',
    },
    'yap_morpheme_pruned_non_o': {
        '_unit': 'morpheme',
        '_scheme': 'bioes',
        'dev': 'morph_yap_dev_pruned_non_o.bmes',
    }
}


word_embedding_files = {
    'alt_tok_yap_ft_sg': '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.yap_form.fasttext_skipgram.model.vec.nofirstline',
    'alt_tok_tokenized_ft_sg': '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.tokenized.fasttext_skipgram.model.vec.nofirstline',
    'htb_all_alt_tok_yap_ft_sg': 'data/htb_all_words.wikipedia.alt_tok.yap_form.fasttext_skipgram.txt',
    'htb_all_alt_tok_tokenized_ft_sg': 'data/htb_all_words.wikipedia.alt_tok.tokenized.fasttext_skipgram.txt',
    'alt_tok_yap_w2v_sg': '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.yap_form.word2vec_skipgram.txt.nofirstline',
    'alt_tok_tokenized_w2v_sg': '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.tokenized.word2vec_skipgram.txt.nofirstline',
    'alt_tok_yap_w2v_sg7': '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.yap_form.word2vec_skipgram.7.txt.nofirstline',
    'alt_tok_tokenized_w2v_sg7': '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.tokenized.word2vec_skipgram.7.txt.nofirstline',
    'alt_tok_yap_w2v_sg5': '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.yap_form.word2vec_skipgram.5.txt.nofirstline',
    'alt_tok_tokenized_w2v_sg5': '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.tokenized.word2vec_skipgram.5.txt.nofirstline',
    'alt_tok_yap_glove': '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.yap_form.glove.txt',
    'alt_tok_tokenized_glove': '../wordembedding-hebrew/vectors_alt_tok/wikipedia.alt_tok.tokenized.glove.txt',
    'orig_tok_yap_w2v_sg': '../wordembedding-hebrew/vectors_orig_tok/wikipedia.yap_form.word2vec_skipgram.txt.nofirstline',
    'orig_tok_tokenized_w2v_sg': '../wordembedding-hebrew/vectors_orig_tok/wikipedia.tokenized.word2vec_skipgram.txt.nofirstline',
    
}


'''
### I/O ###
```Python
train_dir=xx    #string (necessary in training). Set training file directory.
dev_dir=xx    #string (necessary in training). Set dev file directory.
test_dir=xx    #string . Set test file directory.
model_dir=xx    #string (optional). Set saved model file directory.
word_emb_dir=xx    #string (optional). Set pretrained word embedding file directory.

raw_dir=xx    #string (optional). Set input raw file directory.
decode_dir=xx    #string (necessary in decoding). Set decoded file directory.
dset_dir=xx    #string (necessary). Set saved model file directory.
load_model_dir=xx    #string (necessary in decoding). Set loaded model file directory. (when decoding)
char_emb_dir=xx    #string (optional). Set pretrained character embedding file directory.

norm_word_emb=False    #boolen. If normalize the pretrained word embedding.
norm_char_emb=False    #boolen. If normalize the pretrained character embedding.
number_normalized=True    #boolen. If normalize the digit into `0` for input files.
seg=True    #boolen. If task is segmentation like, tasks with token accuracy evaluation (e.g. POS, CCG) is False; tasks with F-value evaluation(e.g. Word Segmentation, NER, Chunking) is True .
word_emb_dim=50    #int. Word embedding dimension, if model use pretrained word embedding, word_emb_dim will be reset as the same dimension as pretrained embedidng.
char_emb_dim=30    #int. Character embedding dimension, if model use pretrained character embedding, char_emb_dim will be reset as the same dimension as pretrained embedidng.
```

### NetworkConfiguration ###
```Python
use_crf=True    #boolen (necessary in training). Flag of if using CRF layer. If it is set as False, then Softmax is used in inference layer.
use_char=True    #boolen (necessary in training). Flag of if using character sequence layer. 
word_seq_feature=XX    #boolen (necessary in training): CNN/LSTM/GRU. Neural structure selection for word sequence. 
char_seq_feature=CNN    #boolen (necessary in training): CNN/LSTM/GRU. Neural structure selection for character sequence, it only be used when use_char=True.
feature=[POS] emb_size=20 emb_dir=xx   #feature configuration. It includes the feature prefix [POS], pretrained feature embedding file and the embedding size. 
feature=[Cap] emb_size=20 emb_dir=xx    #feature configuration. Another feature [Cap].
nbest=1    #int (necessary in decoding). Set the nbest size during decoding.
```

### TrainingSetting ###
```Python
status=train    #string: train or decode. Set the program running in training or decoding mode.
optimizer=SGD    #string: SGD/Adagrad/AdaDelta/RMSprop/Adam. optimizer selection.
iteration=1    #int. Set the iteration number of training.
batch_size=10    #int. Set the batch size of training or decoding.
ave_batch_loss=False    #boolen. Set average the batched loss during training.
```

### Hyperparameters ###
```Python
cnn_layer=4    #int. CNN layer number for word sequence layer.
char_hidden_dim=50    #int. Character hidden vector dimension for character sequence layer.
hidden_dim=200    #int. Word hidden vector dimension for word sequence layer.
dropout=0.5    #float. Dropout probability.
lstm_layer=1    #int. LSTM layer number for word sequence layer.
bilstm=True    #boolen. If use bidirection lstm for word seuquence layer.
learning_rate=0.015    #float. Learning rate.
lr_decay=0.05    #float. Learning rate decay rate, only works when optimizer=SGD.
momentum=0    #float. Momentum 
l2=1e-8    #float. L2-regulization.
#gpu=True  #boolen. If use GPU, generally it depends on the hardward environment.
#clip=     #float. Clip the gradient which is larger than the setted number.
```
'''
BOOL = [True, False]
OPTIMIZERS = ["SGD", "RMSProp", "Adam"] # "AdaGrad", "AdaDelta",

'''
# FIRST GRIDS
default_grid = { 
        # FIXED
        'word_seq_feature': 'LSTM',
        'word_emb_dim': 300,
        'char_emb_dim': 30,
        'iteration': 100,
        'bilstm': True,
        'norm_word_emb': False,
        'norm_char_emb': False,
        'ave_batch_loss': False,
        'l2': 1e-8,
        'lstm_layer': [1, 2],
        'batch_size': [10, 20, 30],
        'number_normalized': False,
        'optimizer': OPTIMIZERS,
        'nbest': 1,
    }
    
arch_grids = {
    'CharLSTM': {
        'char_seq_feature': 'LSTM',
        'use_char': True,
        'use_crf': True,
        'char_hidden_dim': [20, 50, 70], 
        'hidden_dim': [50, 100, 200],
        'dropout': [0.1, 0.3, 0.5],
    },
    'CharCNN': {
        'char_seq_feature': 'CNN',
        'use_char': True,
        'use_crf': True,
        'cnn_layer': [2,4,8],
        'hidden_dim': [50, 100, 200],
        'dropout': [0.1, 0.3, 0.5],
    }
}

optimizer_grids = {
    
    'sgd': {
        'learning_rate': [0.01, 0.015, 0.03],
        'lr_decay': [0.01, 0.05, 0.1],
        'momentum': [0, 0.3, 0.9],
        #clip: ,
    },
    'adam': {
        'learning_rate': [5e-4, 1e-3, 5e-3],
        #clip: ,
    },
    'rmsprop': {
        'learning_rate': [5e-3, 1e-2, 5e-2],
        #clip: ,
    },
}
'''
'''
### SECOND GRIDS
default_grid = { 
        # FIXED
        'word_seq_feature': 'LSTM',
        'word_emb_dim': 300,
        'char_emb_dim': 30,
        'iteration': 100,
        'bilstm': True,
        'norm_word_emb': False,
        'norm_char_emb': False,
        'ave_batch_loss': False,
        'l2': 1e-8,
        'lstm_layer': [1, 2],
        'batch_size': [1, 8],
        'number_normalized': False,
        'optimizer': 'SGD',
        'nbest': 1,
        'hidden_dim': [50, 100, 200],
        'dropout': [0.4, 0.5, 0.6],
    }
    
arch_grids = {
    'CharLSTM': {
        'char_seq_feature': 'LSTM',
        'use_char': True,
        'use_crf': True,
        'char_hidden_dim': [50, 70, 90], 
    },
    'CharCNN': {
        'char_seq_feature': 'CNN',
        'use_char': True,
        'use_crf': True,
        'char_hidden_dim': [50, 70, 90],
        'char_kernel_size': [3, 5, 7]
    },
    'NoChar': {
        'use_char': False,
        'use_crf': True,
     },
}

optimizer_grids = {
    
    'sgd': {
        'learning_rate': [0.005, 0.01, 0.015],
        'lr_decay': 0.05,
        'momentum': 0,
    },
    'adam': {
        'learning_rate': [0.0005, 0.001],
    },
    'rmsprop': {
        'learning_rate': [0.001, 0.005],
    },
}
'''

default_grid = { 
        # FIXED
        'word_seq_feature': 'LSTM',
        'word_emb_dim': 300,
        'char_emb_dim': 30,
        'iteration': 200,
        'bilstm': True,
        'norm_word_emb': False,
        'norm_char_emb': False,
        'ave_batch_loss': False,
        'l2': 1e-8,
        'lstm_layer': 2,
        'batch_size': 8,
        'number_normalized': False,
        'optimizer': 'SGD',
        'nbest': 1,
        'hidden_dim': 200,
        'dropout': [0.4, 0.5],
    }
    
arch_grids = {
    'CharLSTM': {
        'char_seq_feature': 'LSTM',
        'use_char': True,
        'use_crf': True,
        'char_hidden_dim': [50, 70], 
    },
    'CharCNN': {
        'char_seq_feature': 'CNN',
        'use_char': True,
        'use_crf': True,
        'char_hidden_dim': [50, 70],
        'char_kernel_size': [3, 5, 7]
    },
    'NoChar': {
        'use_char': False,
        'use_crf': True,
     },
}

optimizer_grids = {
    
    'sgd': {
        'learning_rate': [0.005, 0.01],
        'lr_decay': 0.05,
        'momentum': 0,
    },
    'adam': {
        'learning_rate': [0.0005, 0.001],
    },
    'rmsprop': {
        'learning_rate': [0.001, 0.005],
    },
}

import random


def get_value(values):
    if type(values) is list or type(values) is tuple:
        return random.choice(values)
    else:
        return values

        
def get_random_grid_config(arch):
    base_grid = default_grid
    base_grid.update(arch_grids[arch])
    
    config = {}
    for param, values in base_grid.items():
        config[param] = get_value(values)
    for param, values in optimizer_grids[config['optimizer'].lower()].items():
        config[param] = get_value(values)

    return config
        
        
def get_unique_grid_config(arch, existing):
    conf = get_random_grid_config(arch)
    found = True
    while found:
        found=False
        for ex in existing:
            if len(conf.items() & ex.items()) == len(conf):
                found=True
                conf = get_random_grid_config(arch)
                break
    return conf

def config_generator(arch):
    existing = []
    while True:
        conf = get_unique_grid_config(arch, existing)
        existing.append(conf)
        yield conf

from itertools import islice

def get_x_configs(arch, x):
    return [c for c in islice(config_generator(arch), x)]


def create_conf_file(out_path, model_dir, dataset, conf, word_emb_dir):
    full_conf_dict = {}
    
    full_conf_dict['model_dir'] = model_dir
    
    for k, v in datasets[dataset].items():
        if not k.startswith('_'):
            if k in ['train_dir', 'dev_dir', 'test_dir']:
                full_conf_dict[k] = os.path.join(data_folder, v)
            else:
                full_conf_dict[k] = v
    
    full_conf_dict['word_emb_dir'] = word_embedding_files[word_emb_dir]
    
    full_conf_dict['status'] = 'train'
    
    full_conf_dict.update(conf)
    
    with open(out_path, 'w', encoding='utf8') as of:
        for k, v in full_conf_dict.items():
            of.write(k+'='+str(v)+'\n')
            
    return full_conf_dict
    