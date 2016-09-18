import nltk.translate.bleu_score as BLEU
import numpy as np
import cPickle as pickle
import os

def sentence_bleu(references, candidate, weights=[0.25, 0.25, 0.25, 0.25]):
    # modified precision for n-gram
    p_ns = [BLEU.modified_precision(references, candidate, i) for i, _ in enumerate(weights, start=1)]
    
    # all of n-gram modified precision should be positive otherwise return 0 (geometric average)
    if all([x.numerator > 0 for x in p_ns]):
        s = np.sum(w * np.log(float(p_n.numerator)/p_n.denominator) for w, p_n in zip(weights, p_ns))
        
        # brevity penalty 
        c = len(candidate)
        r = BLEU.closest_ref_length(references, c)
        if c > r:
            bp = 1
        else:
            bp = np.exp(1. - float(r)/c)
        return bp * np.exp(s)
    else: 
        return 0

def bleu(targets, preds, n_gram):
    '''
    Inputs: 
    - targets: reference sentences of shape (5000, 5)
    - preds: candidate sentences of shape (5000, 1)
    
    Returns: 
    - bleu score
    '''
    weights = []
    w = 1. / n_gram
    for _ in range(n_gram):
        weights.append(w)
    
    bleu_score = 0
    for i, pred in enumerate(preds):
        pred = pred.split()
        target = map(lambda x: x.split(), targets[i])
        bleu_score += sentence_bleu(target, pred, weights)
        
    return bleu_score / len(preds)

def compute_bleu_1to4(references, candidates):
    # compute bleu1 ~ bleu4 scores
    for i in range(4):
        bleu_score = bleu(references, candidates, n_gram=i+1)
        print "BLEU%d: %f" %(i+1, bleu_score)

def evaluate(data_path='./data', split='val'):
    # load candidates and references
    data_path = os.path.join(data_path, split)
    ref_cap_path = os.path.join(data_path, '%s.reference.captions.pkl' %split)
    cand_cap_path = os.path.join(data_path, '%s.candidate.captions.pkl' %split)

    with open(ref_cap_path, 'rb') as f:
        references = pickle.load(f)
    with open(cand_cap_path, 'rb') as f:
        candidates = pickle.load(f)

    compute_bleu_1to4(references, candidates)
