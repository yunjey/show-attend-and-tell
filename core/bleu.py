import cPickle as pickle
import os
import sys
sys.path.append('../coco-caption')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

def score(ref, hypo):
    scorers = [
        (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(),"ROUGE_L"),
        (Cider(),"CIDEr")
    ]
    final_scores = {}
    for scorer,method in scorers:
        score,scores = scorer.compute_score(ref,hypo)
        if type(score)==list:
            for m,s in zip(method,score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores
    

def evaluate(data_path='./data', split='val', get_scores=False):
    reference_path = os.path.join(data_path, "%s/%s.references.pkl" %(split, split))
    candidate_path = os.path.join(data_path, "%s/%s.candidate.captions.pkl" %(split, split))
    
    # load caption data
    with open(reference_path, 'rb') as f:
        ref = pickle.load(f)
    with open(candidate_path, 'rb') as f:
        cand = pickle.load(f)
    
    # make dictionary
    hypo = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption]
    
    # compute bleu score
    final_scores = score(ref, hypo)

    # print out scores
    print 'Bleu_1:\t',final_scores['Bleu_1']  
    print 'Bleu_2:\t',final_scores['Bleu_2']  
    print 'Bleu_3:\t',final_scores['Bleu_3']  
    print 'Bleu_4:\t',final_scores['Bleu_4']  
    print 'METEOR:\t',final_scores['METEOR']  
    print 'ROUGE_L:',final_scores['ROUGE_L']  
    print 'CIDEr:\t',final_scores['CIDEr']
    
    if get_scores:
        return final_scores
    
   
    
    
    
    
    
    
    
    
    
    


