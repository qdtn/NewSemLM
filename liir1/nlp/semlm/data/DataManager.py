import pickle
from liir.nlp.io.Reader import Conll2009BatchReader
from liir.nlp.representation.Text import Text
from liir.nlp.representation.Word import Predicate
from utils.Data import prepare_data_seq
import  numpy as np
__author__ = 'quynhdo'
# this class is used to manage the data
# fist, extract the feature


def extract_sequence(txt, pos="V", data="form"):
    '''

    :param txt: Text
    :return: list of sequence to build SemLM
    '''
    all_data = []
    for sen in txt:
        for w in sen:

                if isinstance(w, Predicate):
                    if data=="form":
                        lst=[(w.form.lower(),"PRED")]
                    elif data== "lemma":
                        lst=[(w.lemma,"PRED")]


                    for arg in sen:
                        if arg in w.arguments.keys():
                            hn = getHeadNoun(arg, data)
                            if hn is not None:
                                lst.append((hn, w.arguments[arg]))

                    if pos == w.pos[0]:
                        all_data.append(lst)


    return all_data


def make_simple_seq(data, eos_at_begining=True):
    '''
    make a simple data sequence
    1-1 data

    '''
    seqX = []
    seqY = []
    for i in range(len(data)):

        X = []
        Y = []
        if eos_at_begining:
            X.append("EOS")
            Y.append(data[i][0][0] +  "_" + data[i][0][1])

        for j in range(len(data[i])-1):
            X.append(data[i][j][0] +  "_" + data[i][j][1])
            Y.append(data[i][j+1][0] +  "_" + data[i][j+1][1])

        X.append(data[i][-1][0] +  "_" + data[i][-1][1])
        Y.append("EOS")
        seqX.append(X)
        seqY.append(Y)
    return seqX,seqY

def make_pair_seq(data, eos_at_begining=True):
    '''
    Creating 2-2 data
    '''
    seqX1 = []
    seqY1 = []
    seqX2 = []
    seqY2 = []
    for i in range(len(data)):

        X1 = []
        Y1 = []
        X2 = []
        Y2 = []
        if eos_at_begining:
            X1.append("EOS")
            X2.append("EOS")
            Y1.append(data[i][0][0])
            Y2.append(data[i][0][1])

        for j in range(len(data[i])-1):
            X1.append(data[i][j][0])
            X2.append(data[i][j][1])
            Y1.append(data[i][j+1][0])
            Y2.append(data[i][j+1][1])

        X1.append(data[i][-1][0] )
        X2.append( data[i][-1][1])
        Y1.append("EOS")
        Y2.append("EOS")
        seqX1.append(X1)
        seqY1.append(Y1)
        seqX2.append(X2)
        seqY2.append(Y2)
    return seqX1,seqX2,seqY1,seqY2


def make_pair_seq_simple(data,  eos_at_begining=True):
    '''
    creating 2-1 data
    '''
    seqX1 = []
    seqY = []
    seqX2 = []
    for i in range(len(data)):

        X1 = []
        Y = []
        X2 = []

        if eos_at_begining:
            X1.append("EOS")
            X2.append("EOS")
            Y.append(data[i][0][0] + "_" + data[i][0][1])

        for j in range(len(data[i])-1):
            X1.append(data[i][j][0])
            X2.append(data[i][j][1])
            Y.append(data[i][j+1][0] + "_" + data[i][j+1][1])

        X1.append(data[i][-1][0] )
        X2.append( data[i][-1][1])
        Y.append("EOS")
        seqX1.append(X1)
        seqY.append(Y)
        seqX2.append(X2)
    return seqX1,seqX2,seqY


def getHeadNoun(w, data="form"):
    '''
    data = form or lemma
    '''
    if w.pos[0] == "N" or w.pos[0] == "P":
        if data == "form":
            return w.form.lower()
        else:
            return w.lemma
    if w.pos == "IN" or w.pos == "TO":
        children = w.getChildren()
        if len(children)>0:
            for i in range(len(children)-1,-1,-1):
                if children[i].pos[0] == "N" or children[i].pos[0] == "P":
                    if data == "form":
                        return children[i].form.lower()
                    else:
                        return children[i].lemma

    return None



def generate_sequential_data(lst, pos="V", data="form", eos_at_begining=True, type="1-1"):

    txt= Text()

    for l in lst:
        txt.readConll2009Sentences(l)

    all_data = extract_sequence(txt, pos, data)
    if type == "1-1":
        return make_simple_seq(all_data, eos_at_begining)
    if type=="2-1":
        return make_pair_seq_simple(all_data, eos_at_begining)
    if type == "2-2":
        return make_pair_seq(all_data, eos_at_begining)

    return None


def preprare_seq_seq_data(X, Y = None):
    x, x_mask = prepare_data_seq(X)
    #xaa= np.asarray(x)
    #print (xaa.shape)
    x = np.asarray(x, dtype='int32')

    x_mask = np.asarray(x_mask, dtype='int32')
    if Y is not None:
        y, y_mask = prepare_data_seq(Y)
    #xaa= np.asarray(y)
    #print (xaa.shape)
        y = np.asarray(y, dtype='int32')
        y_mask = np.asarray(y_mask, dtype='int32')


    #y = y.flatten()
    #y_mask = y_mask.flatten()

        return x, x_mask, y, y_mask

    return x, x_mask