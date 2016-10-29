__author__ = 'quynhdo'
# define getFeatureValue functions that can be used to specify different feature types
#
#
from liir.nlp.representation.Word import Word


def getWordFeatureFunc(ins, kwargs):
    pos=0
    target_word = 'word'
    word_data = 'form'
    if 'pos' in kwargs.keys():
        pos = kwargs['pos']
    if 'target_word' in kwargs.keys():
        target_word = kwargs['target_word']
    if 'word_data' in kwargs.keys():
        word_data = kwargs['word_data']

    if isinstance(ins, Word):
        w = ins
    else:
        w = ins [pos]

    wt = getattr(w, target_word)()

    if wt is None:
            return None

    rs = None

    if isinstance(wt, Word):
            rs =  getattr(wt, word_data)()

    if isinstance(wt, list):
            rs = [getattr(wt, word_data)() for wt in wt]
    return rs

def getWordPairFeatureFunc(ins, kwargs):
     wpd = 'position'
     pos_1 = 0
     pos_2 = 1
     if 'wpd' in kwargs.keys():
        wpd = kwargs['wpd']

     if 'pos_1' in kwargs.keys():
        pos_1 = kwargs['pos_1']

     if 'pos_2' in kwargs.keys():
        pos_2 = kwargs['pos_2']


     w1 = ins[pos_1]
     w2 = ins[pos_2]

     if wpd == 'position':
            posInSentence1 = w1.id
            posInSentence2 = w2.id
            if posInSentence1 == posInSentence2:
                v = 0
            else:
                if posInSentence1 > posInSentence2:
                    v = 1
                else:
                    v = 2
            return v

     if wpd == 'pos_path':
            mp=w1.getPath(w2)
            v= ""
            for i in mp:
                if isinstance(i,str):
                    v+=i
                else:
                    v+=i.pos
            return v
     if wpd == 'deprel_path':
            mp=w1.getPath(w2)
            v= ""
            for i in mp:
                if isinstance(i,str):
                    v+=i
                else:
                    v+=i.deprel

            return v

     if wpd == 'in_context':
            windows = 3
            if 'windows' in kwargs.keys():
                windows = kwargs['windows']

            if abs(w1.id-w2.id) <= (windows -1)//2:
                    return 1
            else:
                return 0

     if wpd == 'path_of_word':
            mp=w1.getWordPath(w2)
            v= []
            for w in mp:
                v.append(w.word)
            return v

     if wpd == 'interaction':
            v = [w1.word, w2.word]
            return v

def getNeighbourFeatureFunc(ins, kwargs):
    pos=0
    target_word = 'word'
    word_data = 'form'
    nei=0
    if 'pos' in kwargs.keys():
        pos = kwargs['pos']
    if 'target_word' in kwargs.keys():
        target_word = kwargs['target_word']
    if 'word_data' in kwargs.keys():
        word_data = kwargs['word_data']
    if 'nei' in kwargs.keys():
        nei = kwargs['nei']

    wIns = None
    if isinstance(ins, Word):
        wIns=ins
    else:
        wIns = ins[pos]

    wTarget = getattr(wIns, target_word)()
    if wTarget is None:
            return None

    if wTarget.id + nei >= 0 and wTarget.id + nei < len(wTarget.sentence):
            wTarget = wTarget.sentence[wTarget.id + nei]
    else:
            return None

    return  getattr(wTarget, word_data)()