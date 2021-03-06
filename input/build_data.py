import pycrfsuite
import sklearn
from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import re
import json
import nltk
from nltk import sent_tokenize, pos_tag, word_tokenize
import numpy as np

annotypes = ['Participants', 'Intervention', 'Outcome']
path = '/nlp/data/romap/set/'
#path = '/Users/romapatel/Desktop/set/'

set_path = 'cross-val/joint/1/'

upper = "[A-Z]"
lower = "a-z"
punc = "[,.;:?!()]"
quote = "[\'\"]"
digit = "[0-9]"
multidot = "[.][.]"
hyphen = '[/]'
dollar = '[$]'
at = '[@]'

all_cap = upper + "+$"
all_digit = digit + "+$"

contain_cap = ".*" + upper + ".*"
contain_punc = ".*" + punc + ".*"
contain_quote = ".*" + quote + ".*"
contain_digit = ".*" + digit + ".*"
contain_multidot = ".*" + multidot + ".*"
contain_hyphen = ".*" + hyphen + ".*"
contain_dollar = ".*" + dollar + ".*"
contain_at = ".*" + at + ".*"

cap_period = upper + "\.$"


list_reg = [contain_cap, contain_punc, contain_quote,
            contain_digit, cap_period, punc, quote, digit,
            contain_multidot, contain_hyphen, contain_dollar, contain_at]

def reg_features(word, start=''):
    res = []
    for p in list_reg:
        if re.compile(p).match(word): 
            res.append(start + p + '=TRUE')
        else:
            res.append(start + p + '=FALSE')
    return res

def build_data(set_path
def run(data_dict, train_docids, test_docids, fold):
    train_dict, test_dict = {}, {}
    for docid in train_docids:
        train_dict[docid] = data_dict[docid]
    for docid in test_docids:
        test_dict[docid] = data_dict[docid]

    print 'Inside run'
    aslog_patterns = get_aslog_patterns()
    pos_patterns = get_pos_patterns()
    indwords = get_ind_words()

    #list of annotated sentences over all docids
    test_sents, train_sents = [], []
    for docid in test_dict: test_sents.extend(test_dict[docid])
    for docid in train_dict: train_sents.extend(train_dict[docid])

    X_train = [sent_features(sentence, indwords, aslog_patterns, pos_patterns) for sentence in train_sents]
    y_train = [sent_labels(sentence) for sentence in train_sents]

    #test_sents = test_sents[:10]
    #X_test = [sent_features(sentence, indwords, aslog_patterns, pos_patterns) for sentence in test_sents]
    #y_test = [sent_labels(sentence) for sentence in test_sents]

    print 'Training'
    trainer = pycrfsuite.Trainer(verbose=False)
    
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
        
    trainer.set_params({'c1': 1.0,
                        'c2': 1e-3,
                        'max_iterations': 50,
                        'feature.possible_transitions': True})
    
    trainer.train(path + set_path + 'PICO.crfsuite')
    
    tagger = pycrfsuite.Tagger()
    tagger.open(path + set_path + 'PICO.crfsuite')

    fin_dict = {}
    for docid in test_docids:
        fin_dict[docid] = {}
        for annotype in annotypes:
            fin_dict[docid][annotype] = {}
            fin_dict[docid][annotype]['CRFPattern'] = []
            
        text = ''
        f = open('/nlp/data/romap/docs/' + str(docid) + '.txt', 'r')
        #f = open('/Users/romapatel/Desktop/PICO-data/docs/' + str(docid) + '.txt', 'r')
        for line in f:
            text += line
        tokens = tokenize(text)
        sentences, sentence = [], []; all_indices, indices = [], [];
        for i in range(len(tokens)):
            if len(tokens[i][0]) == 0: continue

            if tokens[i][0] == '.':
                sentences.append(sentence); sentence = [];
                all_indices.append(indices); indices = [];

            else:
                sentence.append(tokens[i][0])
                indices.append([tokens[i][1], tokens[i][2]])

        part_spans, int_spans, out_spans = [], [], []
        for i in range(len(sentences)):
            sent_tuples, sent_indices = [], all_indices[i]
            pos_tags = pos_tag(sentences[i])
            
            for j in range(len(pos_tags)):
                item = pos_tags[j]
                sent_tuples.append([item[0], 0, item[1], 0, 0, 0])

            X_test = [sent_features(sent_tuples, indwords, aslog_patterns, pos_patterns)]
            pred = [tagger.tag(xseq) for xseq in X_test][0]

            p, i, o = [], [], []
            for item in pred:
                if item == 'N':
                    p.append(0); i.append(0); o.append(0);
                if item == 'P':
                    p.append(1); i.append(0); o.append(0);
                if item == 'I':
                    p.append(0); i.append(1); o.append(0);
                if item == 'O':
                    p.append(0); i.append(0); o.append(1);
            
            p_spans = mask2spans(p); i_spans = mask2spans(i); o_spans = mask2spans(o);
            for span in p_spans:
                begin_w, end_w = span[0], span[1]-1
                begin_i, end_i = sent_indices[begin_w][0], sent_indices[end_w][1]
                part_spans.append((begin_i, end_i))
            for span in i_spans:
                begin_w, end_w = span[0], span[1]-1
                begin_i, end_i = sent_indices[begin_w][0], sent_indices[end_w][1]
                int_spans.append((begin_i, end_i))

            for span in o_spans:
                begin_w, end_w = span[0], span[1]-1
                begin_i, end_i = sent_indices[begin_w][0], sent_indices[end_w][1]
                out_spans.append((begin_i, end_i))
    
        fin_dict[docid]['Participants']['CRFPattern'] = part_spans
        fin_dict[docid]['Intervention']['CRFPattern'] = int_spans
        fin_dict[docid]['Outcome']['CRFPattern'] = out_spans

    hmm_dict = {}
    f = open(path + set_path + 'hmm_input.json', 'r')
    for line in f:
        hmm_dict = json.loads(line)
    for docid in fin_dict:
        hmm_dict[docid] = fin_dict[docid]

    f = open(path + set_path + 'hmm_input.json', 'w+')
    f.write(json.dumps(hmm_dict))


def all_metrics(y_true, y_pred):
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
        )

def mask2spans(mask):
    spans = []
    if mask[0] == 1:
        sidx = 0
    for idx, v in enumerate(mask[1:], 1):
        if v==1 and mask[idx-1] == 0: # start of span
            sidx = idx
        elif v==0 and mask[idx-1] == 1 : # end of span
            eidx = idx
            spans.append( (sidx, eidx) )
    return spans

def precision(spans, ref_mask):
    if len(spans) == 0: return 0
    precision_arr = []
    for span in spans:
        length = span[1]-span[0]
        poss = sum(ref_mask[span[0]:span[1]])
        precision_arr.append(1.0*poss / length)
    precision = np.mean(precision_arr)

    return precision

def recall(gold_spans, anno_mask):
    recall_arr = []
    for span in gold_spans:
        length = span[1]-span[0]
        poss = sum(anno_mask[span[0]:span[1]])
        recall_arr.append(1.0*poss / length)
    recall = np.mean(recall_arr)

    return recall

def accuracy(gold_mask, pred_mask):
    true_pos = 0
    for i in range(len(gold_mask)):
        if gold_mask[i] == pred_mask[i]: true_pos += 1
    return 1.0*true_pos/len(gold_mask)

def get_results(y_true, y_pred, test_sents, fold):
    print 'Getting results'
    gold_mask, pred_mask, tuples = [], [], []
    for i in range(len(y_true)):
        gold_mask.extend(y_true[i])
        pred_mask.extend(y_pred[i])
        tuples.extend(test_sents[i])
        
    p_true, i_true, o_true, p_pred, i_pred, o_pred = np.zeros(len(gold_mask)), np.zeros(len(gold_mask)), np.zeros(len(gold_mask)), np.zeros(len(gold_mask)), np.zeros(len(gold_mask)), np.zeros(len(gold_mask))
    for i in range(len(gold_mask)-1):
        if gold_mask[i] == 'P': p_true[i] = 1
        if gold_mask[i] == 'I': i_true[i] = 1
        if gold_mask[i] == 'O': o_true[i] = 1
        if pred_mask[i] == 'P': p_pred[i] = 1
        if pred_mask[i] == 'I': i_pred[i] = 1
        if pred_mask[i] == 'O': o_pred[i] = 1

    print 'Part'
    annotype = 'Participants'
    gold_spans = mask2spans(p_true)
    pred_spans = mask2spans(p_pred)
    prec = precision(pred_spans, p_true)
    recl = recall(gold_spans, p_pred)
    acc = accuracy(p_true, p_pred)

    f = open(path + set_path + 'results/' + 'fold_' + str(fold) + '.txt', 'w+')
    f.write(annotype + ':\n')
    f.write('precision: ' + str(round(100*prec, 3)) + '\n')
    f.write('recall: ' + str(round(100*recl, 3)) + '\n')
    f.write('accuracy: ' + str(round(100*acc, 3)) + '\n\n\n')

    annotype = 'Intervention'
    gold_mask, pred_mask = i_true, i_pred

    gold_spans = mask2spans(i_true)
    pred_spans = mask2spans(i_pred)

    prec = precision(pred_spans, i_true)
    recl = recall(gold_spans, i_pred)
    acc = accuracy(i_true, i_pred)

    f.write(annotype + ':\n')
    f.write('precision: ' + str(round(100*prec, 3)) + '\n')
    f.write('recall: ' + str(round(100*recl, 3)) + '\n')
    f.write('accuracy: ' + str(round(100*acc, 3)) + '\n\n\n')

    annotype = 'Outcome'
    gold_mask, pred_mask = o_true, o_pred

    gold_spans = mask2spans(o_true)
    pred_spans = mask2spans(o_pred)

    prec = precision(pred_spans, o_true)
    recl = recall(gold_spans, o_pred)
    acc = accuracy(o_true, o_pred)

    f.write(annotype + ':\n')
    f.write('precision: ' + str(round(100*prec, 3)) + '\n')
    f.write('recall: ' + str(round(100*recl, 3)) + '\n')
    f.write('accuracy: ' + str(round(100*acc, 3)) + '\n')
        
def sent_features(sent, indwords, aslog_patterns, pos_patterns):
    return [word_features(sent, i, indwords, aslog_patterns, pos_patterns) for i in range(len(sent))]

def sent_labels(sent):
    #print sent
    p_labels = [token_tuple[3] for token_tuple in sent]
    i_labels = [token_tuple[4] for token_tuple in sent]
    o_labels = [token_tuple[5] for token_tuple in sent]
    all_labels = []
    for i in range(len(p_labels)):
        if p_labels[i] == 1: all_labels.append('P')
        elif i_labels[i] == 1: all_labels.append('I')
        elif o_labels[i] == 1: all_labels.append('O')
        else: all_labels.append('N')

    return all_labels

def is_disease(word, indwords):
    diseases = indwords[0]
    if word in diseases or word[:-1] in diseases:
        return True
    else:
        return False

def is_outcome(word, indwords):
    outcomes = indwords[2]
    if word in outcomes or word[:-1] in outcomes:
        return True
    else:
        return False

def is_drug(word, indwords):
    drugs = indwords[1]
    for i in range(len(drugs)):
        if drugs[i] in word:
            return True
    return False
    
def word_features(sent, i, indwords, aslog_patterns, pos_patterns):
    aslog_patterns = aslog_patterns[0]
    pos_patterns = pos_patterns[0]
    sent_str = ' '.join(item[0].lower() for item in sent)
    aslog = is_aslog_pattern(sent_str, aslog_patterns)
    word, postag = sent[i][0], sent[i][2]
    word_lower = word.lower()
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
                ]
    
    if i == 0: features.append('BOS')
    if i == len(sent)-1: features.append('EOS')
    
    features.extend(reg_features(word))

    L = 3
    for l in range(1, L+1, 1):
        if i-l >= 0:
            word1 = sent[i-l][0]
            postag1 = sent[i-l][2]
            features.extend([
                '-%d:word.lower=' % l + word1.lower(),
                '-%d:word.istitle=%s' % (l, word1.istitle()),
                '-%d:word.isupper=%s' % (l, word1.isupper()),
                '-%d:postag=' % l + postag1,
                '-%d:postag[:2]=' % l + postag1[:2],
            ])
            features.extend(reg_features(word1, '-%d' % l))

        if i + l < len(sent):
            word1 = sent[i+l][0]
            postag1 = sent[i+l][2]
            features.extend([
                '+%d:word.lower='  % l + word1.lower(),
                '+%d:word.istitle=%s' % (l, word1.istitle()),
                '+%d:word.isupper=%s' % (l, word1.isupper()),
                '+%d:postag=' % l + postag1,
                '+%d:postag[:2]=' % l + postag1[:2],
            ])
            features.extend(reg_features(word1, '+%d' % l))
            
    if i > 0:
        bigram_prev = sent[i-1][0].lower() + ' ' + word_lower
        features.append('bigram-1=' + bigram_prev)
    if i > 1:
        trigram_prev = sent[i-2][0].lower() + ' ' + sent[i-1][0].lower() + ' ' + word_lower
        features.append('trigram-1=' + trigram_prev)


        
    if i < len(sent) - 1:
        bigram_next =  word_lower + ' ' + sent[i+1][0].lower() 
        features.append('bigram+1=' + bigram_next)
    if i < len(sent) - 2:
        trigram_next = word_lower + ' ' + sent[i+1][0].lower() + ' ' + sent[i+2][0].lower() 
        features.append('trigram+1=' + trigram_next)


    return features

def map(words, pos):
    word_list = words.strip().split(' ')
    pos_list = pos.strip().split(' ')
    fin_str, open_classes = '', ['NN', 'NNS', 'JJ', 'CD', 'JJR', 'NNP', 'NNPS', 'PRP', 'RB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'SP', 'POS', 'SYM', 'XX', 'X']
    open_list = ['with', 'in', 'after', 'over', 'at', 'from', 'into', 'during', 'including', 'until',
                 'against', 'among', 'throughout', 'despite', 'towards', 'upon', 'concerning',
                 'of', 'to', 'for', 'on', 'by', 'about', 'like', 'through', 'before', 'between',
                 'since', 'without', 'under', 'within', 'along', 'following', 'across', 'behind',
                 'beyond', 'plus', 'except', 'but', 'up', 'out', 'around', 'down', 'off', 'above',
                 'near']
    for i in range(len(word_list)):
        if word_list[i] in open_list:
            fin_str += word_list[i] + ' '
        #if pos_list[i] in open_classes:
            #fin_str += pos_list[i] + ' '
        elif word_list[i] == '(':
            fin_str += '-LLB-' + ' '
        elif word_list[i] == ')':
            fin_str += '-RRB-' + ' '
        else:
            fin_str += pos_list[i] + ' '
    return fin_str.strip()
    
def get_aslog_patterns():
    fin_list, overlap = [], []
    for annotype in annotypes:
        anno_list = []
        f = open(path + 'data/features/' + annotype.lower() + '_frames.tsv', 'r')
        for line in f:
            pattern = line.strip().split(' ')
            if len(pattern) == 1: continue
            anno_list.append(line.strip())
        fin_list.append(sorted(anno_list))
    return (fin_list)

def get_pos_patterns():
    fin_list, stop_symbols = [], ['.', ',', '\''] 
    for annotype in annotypes:
        anno_list = []
        f = open(path + 'data/features/' + annotype.lower() + '_patterns.tsv', 'r')
        for line in f:
            items = line.strip().split('\t')
            prec, pattern = float(items[0]), items[1]
            if prec < 0.8: continue
            if re.search('[\'\".,:]', pattern) is not None: continue
            anno_list.append(pattern)
        fin_list.append(anno_list)
    return (fin_list)

def get_ind_words():
    fin_list = []    
    anno_list = []
    f = open(path + 'data/features/' + 'disease_names.txt', 'r')
    for line in f:
        anno_list.append(line.strip())
    f = open(path + 'data/features/' + 'numbers.txt', 'r')
    for line in f:
        anno_list.append(line.strip())
    fin_list.append(anno_list)
    anno_list = []
    f = open(path + 'data/features/' + 'drug_names.txt', 'r')
    for line in f:
        drug = line.strip()
        if len(drug) <= 3: continue
        anno_list.append(drug)
    fin_list.append(anno_list)
    anno_list = []
    f = open(path + 'data/features/' + 'outcome_names.txt', 'r')
    for line in f:
        outcome = line.strip()
        if len(outcome) <= 3: continue
        anno_list.append(outcome)
    fin_list.append(anno_list)
    return (fin_list)

def is_aslog_pattern(word_window, pattern_list):
    for pattern in pattern_list:
        pattern = ' ' + pattern + ' '
        if pattern in word_window:
            return True
    return False

def is_pos_pattern(word_window, pattern_list):
    for pattern in pattern_list:
        if pattern in word_window:
            return True
    return False

def is_indword(word, word_list):
    if word in word_list:
        return True
    else:
        return False

def get_train_test_sets():
    test_docids, train_docids, dev_docids, gold_docids = [], [], [], []
    f = open(path + 'data/docids/train.txt', 'r')
    for line in f:
        train_docids.append(line.strip())
    f = open(path + 'data/docids/test.txt', 'r')
    for line in f:
        test_docids.append(line.strip())
    f = open(path + 'data/docids/dev.txt', 'r')
    for line in f:
        dev_docids.append(line.strip())
    f = open(path + 'data/docids/gold.txt', 'r')
    for line in f:
        gold_docids.append(line.strip())
    print 'Finished loading docids'
    test_dict, train_dict, dev_dict, gold_dict = {}, {}, {}, {}
    f = open(path + 'data/annotations/HMMCrowd/training_all.json', 'r')
    for line in f:
        dict = json.loads(line)

    print 'Finished first dict'
    for docid in dict:
        train_dict[docid] = dict[docid]
        if docid in gold_docids:
            gold_dict[docid] = dict[docid]

    print 'Finished loading data'

    print ('Train set: ' + str(len(train_dict.keys())))
    print ('Test set: ' + str(len(gold_dict.keys())))
    return train_dict, gold_dict

def tokenize(s):
    """
    :param s: string of the abstract
    :return: list of word with original positions
    """
    def white_char(c):
        return c.isspace() or c in [',', '?']
    res = []
    i = 0
    while i < len(s):
        while i < len(s) and white_char(s[i]): i += 1
        l = i
        while i < len(s) and (not white_char(s[i])): i += 1
        r = i
        if s[r-1] == '.':       # consider . a token
            res.append( (s[l:r-1], l, r-1) )
            res.append( (s[r-1:r], r-1, r) )
        else:
            res.append((s[l:r], l, r))
    return res

if __name__ == '__main__':
    f = open(path + set_path + 'hmm_input.json', 'w+')
    train, gold = get_train_test_sets()
    new_train = {}
    for i in range(len(train.keys())):
        new_train[train.keys()[i]] = train[train.keys()[i]]

    train = new_train
    cv, split = len(train.keys())/10, []
    for i in range(10):
        split.append([])

    split = [train.keys()[i:i+cv] for i in xrange(0, len(train.keys()), cv)]
    fin_split = split[:10]
    fin_split[9].extend(split[10])

    for fold in range(0, 10):
        train_split, test_split = [], []
        test_split = split[fold]
        for i in range(0, 10):
            if i == fold: continue
            train_split.extend(split[i])
        run(train, train_split, test_split, fold)


