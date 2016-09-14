import numpy as np
import copy

class Homogeneous_Data():
    
    
    def __init__(self, captions, batch_size):
        self.captions = captions
        self.batch_size = batch_size
        self.prepare()
        self.reset()

    def prepare(self):
        # compute length for each caption
        cap_len = []
        for cap in self.captions:
            length = 0
            for word in cap:
                if word != 0:
                    length += 1
            cap_len.append(length) 
        self.cap_len = np.array(cap_len)
        self.len_unique = np.unique(cap_len)

        # generate dictionary for mapping length to the relevant caption indices
        self.len_indices = {}
        self.len_count = {}
        for ll in self.len_unique:
            self.len_indices[ll] = np.where(self.cap_len == ll)[0]
            self.len_count[ll] = self.len_indices[ll].shape[0]

    def reset(self):
        # reset data for new train epoch
        self.len_curr_count = copy.copy(self.len_count)
        self.len_unique = np.random.permutation(self.len_unique)
        self.len_indices_pos = {}
        for ll in self.len_unique:
            self.len_indices_pos[ll] = 0
            self.len_indices[ll] = np.random.permutation(self.len_indices[ll])
        self.len_idx = -1

    def get_next(self):
        count = 0 
        while True:
            self.len_idx = np.mod(self.len_idx+1, len(self.len_unique))
            if self.len_curr_count[self.len_unique[self.len_idx]] > 0:
                break
            count += 1
            if count >= len(self.len_unique):
                break
        if count >= len(self.len_unique):
            self.reset()

        # get current batch size
        curr_len = self.len_unique[self.len_idx]
        curr_batch_size = np.minimum(self.batch_size, self.len_curr_count[curr_len])
        # get indices of same length captions for current batch
        curr_pos = self.len_indices_pos[curr_len]
        curr_indices = self.len_indices[curr_len][curr_pos:curr_pos+curr_batch_size]
        self.len_indices_pos[curr_len] += curr_batch_size
        self.len_curr_count[curr_len] -= curr_batch_size

        return curr_indices