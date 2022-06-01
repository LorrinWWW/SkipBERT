import copy
import json
import time
import torch.nn.functional as F
import torch
import numpy as np
import sys,os
import numba

def _set_madvise(large_data, advise=1):
    '''
    0: MADV_NORMAL
    1: MADV_RANDOM
    2: MADV_SEQUENTIAL
    3: MADV_WILLNEED
    4: MADV_DONTNEED
    '''
    import ctypes
    madvise = ctypes.CDLL("libc.so.6").madvise
    madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    madvise.restype = ctypes.c_int
    assert madvise(large_data.ctypes.data, large_data.size * large_data.dtype.itemsize, advise) == 0, "MADVISE FAILED" # 1 means MADV_RANDOM

def _read_or_create_memmap(path, return_tensor=True, *args, **kargs):
    if os.path.exists(path):
        a = np.memmap(path, mode='r+', *args, **kargs)
    else:
        a = np.memmap(path, mode='w+', *args, **kargs)
        # first row is reserved for oovs
        a[0] = 0
    _set_madvise(a, advise=1)
    if return_tensor:
        a = torch.from_numpy(a) # zero-copy
    return a

def _to_key(k):
    return tuple(k.tolist())

@numba.njit()
def _input_ids_to_tri_grams(x: np.array):
    bs, seq_len = x.shape
    ret = np.zeros((bs*seq_len, 3), dtype=np.int64)
    i_ret = 0
    for i_bs in range(bs):
        for i_token in range(seq_len):
            if x[i_bs, i_token] == 0:
                i_ret += 1
                break
            if i_token == 0:
                ret[i_ret][1] = x[i_bs, i_token]
                ret[i_ret][2] = x[i_bs, i_token+1]
            elif i_token == seq_len - 1:
                ret[i_ret][0] = x[i_bs, i_token-1]
                ret[i_ret][1] = x[i_bs, i_token]
            else:
                ret[i_ret] = x[i_bs, i_token-1:i_token+2]
            i_ret += 1
    return ret[:i_ret]


@numba.njit()
def _input_ids_to_ngram_ids(d: dict, x: np.array):
    '''
    input_ids tp ngram_ids.
    try match (x0, x1, x2) -> id;
    if not possible, match (0, x1, x2) -> id;
    if also not possible, match (0, x1, 0) -> id.
    '''
    bs, seq_len = x.shape
    ret = np.zeros(bs*seq_len, dtype=np.int64)
    i_ret = 0
    for i_bs in range(bs):
        for i_token in range(seq_len):
            if x[i_bs, i_token] == 0:
                i_ret += 1
                break
            if i_token == 0:
                k = (0, x[i_bs, i_token], x[i_bs, i_token+1])
            elif i_token == seq_len - 1:
                k = (x[i_bs, i_token-1], x[i_bs, i_token], 0)
            else:
                k = (x[i_bs, i_token-1], x[i_bs, i_token], x[i_bs, i_token+1])
            if k in d: # tri-gram
                ret[i_ret] = d[k]
            else:
                k = (0, k[1], k[2])
                if k in d: # bi-gram
                    ret[i_ret] = d[k]
                else:
                    k = (0, k[1], 0)
                    if k in d: # uni-gram
                        ret[i_ret] = d[k]
            i_ret += 1
    return ret[:i_ret]

@numba.njit()
def _has_oov(d: dict, x: np.array):
    bs, seq_len = x.shape
    for i_bs in range(bs):
        for i_token in range(seq_len):
            if x[i_bs, i_token] == 0:
                break
            if i_token == 0:
                k = (0, x[i_bs, i_token], x[i_bs, i_token+1])
            elif i_token == seq_len - 1:
                k = (x[i_bs, i_token-1], x[i_bs, i_token], 0)
            else:
                k = (x[i_bs, i_token-1], x[i_bs, i_token], x[i_bs, i_token+1])
            if k not in d:
                return True
    return False


class Plot:
    def __init__(self, max_num_entries=100000, hidden_size=768):
        
        self.max_num_entries = max_num_entries
        self.hidden_size = hidden_size
        
        self.trigram_to_id, self.id_to_trigram = self.build_hash_table('input_ids_tri_gram.memmap', max_num_entries)
        self.orig_trigram_hidden_states =  _read_or_create_memmap("plot_hidden_states_tri_gram.memmap", dtype='float16', shape=(max_num_entries, 3, hidden_size))

    def build_hash_table(self, path, max_num_entries):
        n_gram = 3
        hash_table1 = numba.typed.Dict()
        hash_table1[tuple([0]*n_gram)] = 0 # dummy entry
        orig_ngram_ids_mmap = _read_or_create_memmap(
            path, return_tensor=False, dtype='int32', shape=(max_num_entries, n_gram))

        for i in range(1, self.max_num_entries):
            _tmp = orig_ngram_ids_mmap[i]
            # break when meet all 0 ngram
            if (_tmp==0).all():
                break
            tmp_hash = _to_key(_tmp)
            if tmp_hash not in hash_table1:
                hash_table1[tmp_hash] = i
                
        return hash_table1, orig_ngram_ids_mmap
    
    def input_ids_to_tri_grams(self, input_ids):
        return _input_ids_to_tri_grams(input_ids)
    
    def update_data(self, ngram_input_ids, ngram_hidden_states):
        ngram_input_ids = ngram_input_ids.cpu().numpy()
        ngram_hidden_states = ngram_hidden_states.detach().half().cpu() # FP16
        bs, ngram = ngram_input_ids.shape
        ngram_to_id, id_to_ngram, id_to_hidden_state = \
            self.trigram_to_id, self.id_to_trigram, self.orig_trigram_hidden_states
        # TODO: optimize the for-loop later
        id_to_save = []
        for i in range(bs):
            ngram = _to_key(ngram_input_ids[i])
            # TODO: handle ngram_id > max_size
            ngram_id = ngram_to_id.get(ngram, len(ngram_to_id))
            if ngram_id >= self.max_num_entries:
                print('Exceed max number of entries...')
                print('Skip current entry...')
                continue
            ngram_to_id[ngram] = ngram_id
            id_to_ngram[ngram_id] = ngram
            id_to_save.append(ngram_id)
        id_to_hidden_state[id_to_save] = ngram_hidden_states
    
    def retrieve_data(self, input_ids):
        input_ids = input_ids.numpy()
        id_to_get = _input_ids_to_ngram_ids(self.trigram_to_id, input_ids)
        hidden_states = self.orig_trigram_hidden_states[id_to_get]
        return hidden_states
    
    def has_oov(self, input_ids):
        return _has_oov(self.trigram_to_id, input_ids.numpy())
    