def build_word_to_idx(sentences, threshold=5): 
    word_counts = {}
    n_sents = 0
    max_len = 0
    for i, sentence in enumerate(sentences):
        n_sents += 1
        
        if len(sentence.split(" ")) > max_len:
            max_len = len(sentence.split(" "))
            
        for word in sentence.lower().split(' '):
            word_counts[word] = word_counts.get(word, 0) + 1
    vocab = [word for word in word_counts if word_counts[word] >= threshold]
    print 'Filtered words from %d to %d' % (len(word_counts), len(vocab))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx_to_word = {0: u'<NULL>' , 1: u'<START>', 2: u'<END>'}
    
    idx = 4
    for word in vocab:
        word_to_idx[word] = idx
        idx_to_word[idx] = word
        idx += 1

    return word_to_idx, idx_to_word, max_len