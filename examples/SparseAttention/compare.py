from scipy.sparse import coo_matrix
import numpy as np
import torch
from ctypes import cdll
cdll.LoadLibrary('/home/ljq/mine/graphiler/src/build/sputnik/libsputnik.so')
import mygraph
from graphiler.utils import init_log, load_data, setup, empty_cache, bench

device = setup()

USE_DGL_DATASET = True

if USE_DGL_DATASET:
    from graphiler.utils import homo_dataset
else:
    from dataset import *
    homo_dataset = {
                    'citeseer':3703, 'cora':1433, 'pubmed':500, 
                    'ppi':50, 'PROTEINS_full':29, 'OVCAR-8H':66,
                    'Yeast':74, 'DD':89, 'YeastH':75, 'amazon0505':96,
                    'artist':100, 'com-amazon':96, 'soc-BlogCatalog':128,
                    'amazon0601':96, 'web-BerkStan':100, 'web-Google':100,
                    'web-NotreDame':100, 'web-Stanford':100}
    
def profile(data_name, feat_dim, repeat=1000):
    log = init_log(["1-FlashInfer", "2-FlashGAT_16x16", "3-FlashGAT_16x8", \
                    "4-FlashGAT_8x16", "5-FlashGAT_8x8", "6-FlashGAT_4x8", \
                    "6-FlashGAT_2x16", "6-FlashGAT_csr"],
                   ["time", "mem"])
    print("benchmarking on: " + data_name)

    features, dataset = None, None
    if USE_DGL_DATASET:
        dataset, features = load_data(data_name, feat_dim, prepare=False)
        features = features.to(device)
    else:
        dataset = TCGNN_dataset(data_name, feat_dim, load_from_txt=False)    
        features = dataset.x.to(device)
    
    @empty_cache
    def run_flashinfer(dataset, features):
        return
