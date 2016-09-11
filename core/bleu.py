import nltk.translate.bleu_score as BLEU
import numpy as np

def sentence_bleu(references, candidate, weights=[0.25, 0.25, 0.25, 0.25]):
    p_ns = [BLEU.modified_precision(references, candidate, i) for i, _ in enumerate(weights, start=1)]
    if all([x.numerator > 0 for x in p_ns]):
        s = np.sum(w * np.log(float(p_n.numerator)/p_n.denominator) for w, p_n in zip(weights, p_ns))
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
    - targets: 5000 x (4~5)
    - preds: 5000 x 1
    
    Returns: bleu score
    
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
    for i in range(4):
        bleu_score = bleu(references, candidates, n_gram=i+1)
        print "BLEU%d: %f" %(i+1, bleu_score)
    