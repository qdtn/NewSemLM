from liir.ml.core.options.Option import Option
from liir.ml.core.layers.Dense import Dense
from liir.ml.core.layers.Dropout import Dropout
from liir.ml.core.layers.Embedding import Embedding
from liir.ml.core.layers.LSTM import LSTM
from liir.ml.core.layers.Model import Sequential
from liir.ml.core.layers.TimeDistributed import TimeDitributed
from liir.nlp.we.WEDict import WEDict
from theano.tensor.shared_randomstreams import RandomStreams
from liir1.nlp.semlm.data.DataManager import generate_sequential_data, preprare_seq_seq_data
from liir1.nlp.semlm.exp.ConfigReader import read_config
from liir1.nlp.semlm.features.FeatureManager import FeatureManager
import theano as th
import numpy as np
import theano.tensor as T

__author__ = 'quynhdo'

#### in the simplest model, we just have to use standard RNN model
####

class Model11(Sequential):
    def __init__(self, input_dim, output_dim, hidden_dim, dep=2, loss="nl", optimizer="ada", we_dict=None, map=None, use_noise=True):
        '''

        :param input_dim:
        :param output_dim:
        :param hidden_dim:
        :param dep:
        :param loss:
        :param optimizer:
        :param we_dict: word embedding dictionary
        :param map: mapping from word to index
        :return:
        '''
        Sequential.__init__(self, use_mask=True, input_value_type="int32", prediction_type="vector",
                            prediction_value_type='int32', use_noise=use_noise)

        self.option[Option.LOSS] = loss
        self.option[Option.OPTIMIZER] = optimizer
        l1 = Embedding(input_dim, hidden_dim, we_dict=we_dict, map=map)  # first layer is an embedding layer
        self.add_layer(l1)

        l2 = Dropout(hidden_dim,theano_rng= RandomStreams(128))
        self.add_layer(l2)
        for i in range(dep):
            l3 = LSTM(hidden_dim, hidden_dim, return_sequences=True)
            self.add_layer(l3)

        l4 = Dropout(hidden_dim,theano_rng= RandomStreams(128))
        self.add_layer(l4)
        l5 = TimeDitributed(core_layer=Dense(hidden_dim, output_dim, activation="softmax"))

        self.add_layer(l5)
        self.option[Option.IS_SEQUENCE_WORK] = True




def trainSemLM11(train_texts, valid_texts, we_dict_path=None, dep=1,hidden_size=32, batch_size=200, save_folder=".", model_name="simple", max_epochs=120,
                 load_dt=True):

    fm = FeatureManager()
    all_texts = train_texts + valid_texts
    X,Y = generate_sequential_data(all_texts, loaddata=load_dt)

    for i in range(len(X)):
        fm.extract_features(X[i],Y[i])



    X,Y= generate_sequential_data(train_texts, loaddata=load_dt)

    X = [[fm.f.map[fm.f.getFeatureValue(x)] +1 for x in XX] for XX in X ]
    Y = [[fm.fY.map[fm.fY.getFeatureValue(x)] + 1 for x in XX] for XX in Y]



    Xv,Yv=generate_sequential_data(valid_texts, loaddata=load_dt)



    Xv = [[fm.f.map[fm.f.getFeatureValue(x)] +1 for x in XX] for XX in Xv ]
    Yv = [[fm.fY.map[fm.fY.getFeatureValue(x)] + 1 for x in XX] for XX in Yv]



    we_dict = None
    if we_dict_path is not None:
        we_dict = WEDict(we_dict_path)

    mdl =Model11(fm.f.current_index + 1,
                  fm.fY.current_index + 1,

                  hidden_size, dep=dep, we_dict=we_dict,map=fm.f.map)

    mdl.option[Option.SAVE_TO] = save_folder + "/" + model_name + ".pkl"
    mdl.option[Option.SAVE_FREQ] = 20
    mdl.option[Option.VALID_FREQ] = 20
    mdl.option[Option.BATCH_SIZE] = batch_size
    mdl.option[Option.SAVE_BEST_VALID_TO] = save_folder + "/" + "best_" + model_name + ".pkl"
    mdl.option[Option.MAX_EPOCHS]=max_epochs
    mdl.compile()
    func = preprare_seq_seq_data
    mdl.fit_shuffer(X, Y, Xv,Yv, process_data_func=func)


def loadSemLM11(load_path, train_texts, valid_texts, we_dict_path=None, dep=1,hidden_size=32, batch_size=200, save_folder=".", model_name="simple",
                          max_epochs=120, continue_train=False, load_dt=True):

    fm = FeatureManager()
    all_texts = train_texts + valid_texts
    X,Y = generate_sequential_data(all_texts, loaddata=load_dt)

    for i in range(len(X)):
        fm.extract_features(X[i],Y[i])



    X,Y= generate_sequential_data(train_texts, loaddata=load_dt)

    X = [[fm.f.map[fm.f.getFeatureValue(x)] +1 for x in XX] for XX in X ]
    Y = [[fm.fY.map[fm.fY.getFeatureValue(x)] + 1 for x in XX] for XX in Y]



    Xv,Yv=generate_sequential_data(valid_texts, loaddata=load_dt)



    Xv = [[fm.f.map[fm.f.getFeatureValue(x)] +1 for x in XX] for XX in Xv ]
    Yv = [[fm.fY.map[fm.fY.getFeatureValue(x)] + 1 for x in XX] for XX in Yv]



    we_dict = None
    if we_dict_path is not None:
        we_dict = WEDict(we_dict_path)

    mdl =Model11(fm.f.current_index + 1,
                  fm.fY.current_index + 1,

                  hidden_size, dep=dep, we_dict=we_dict,map=fm.f.map)

    mdl.option[Option.SAVE_TO] = save_folder + "/" + model_name + ".pkl"
    mdl.option[Option.SAVE_FREQ] = 20
    mdl.option[Option.VALID_FREQ] = 20
    mdl.option[Option.BATCH_SIZE] = batch_size
    mdl.option[Option.SAVE_BEST_VALID_TO] = save_folder + "/" + "best_" + model_name + ".pkl"
    mdl.option[Option.MAX_EPOCHS]=max_epochs

    mdl.compile()
    func = preprare_seq_seq_data

    mdl.load_params(load_path)
    if continue_train:
        mdl.fit_shuffer(X, Y, Xv,Yv, process_data_func=func)
    else:
        return mdl, fm


def get_verb_embeddings(mdl, fm, ofn, embedding_layer=2, fn = None, vob = None):
    '''
    extract the verb embeddings
    :param mdl : the model
    :fm : feature manager
    :param fn: file containing the verbs, verbs are separated by a space
    :ofn : output file
    :return: the new file containing  the embeddings of the verbs
    '''

    if fn is not None:
        vobs = set()
        f = open(fn, "r")
        for l in f.readlines():
            tmps = l.split(" ")
            for tmp in tmps:
                if tmp != "":
                    vobs.add(tmp)

        vobs = list(vobs)
    else:
        if vob is not None:
            vobs = vob

    vv = []

    for v in vobs:
        if v + "_" + "PRED" in fm.f.map.keys():
            vv.append(v)

    vobs = vv

    X = [ [v + "_" + "PRED"] for v in vobs]

    Y = [["EOS"] for i in range(len(X))]

    X = [[fm.f.map[fm.f.getFeatureValue(x)] +1 for x in XX] for XX in X ]
    Y = [[fm.fY.map[fm.fY.getFeatureValue(x)] + 1 for x in XX] for XX in Y]

    x,x_mask,y,y_mask = preprare_seq_seq_data(X,Y)
    x, y,  mask_x,mask_y, _, _,_, _ = mdl.standardize_data(x, y, x_mask, y_mask, None,None, None,None)
    rs = mdl.get_output_layer(embedding_layer, x, mask_x)
    f = open (ofn, "w")
    for i in range(len(vobs)):
        w = vobs[i]
        em = rs[0][i]
        f.write(w + " ")
        for e in em:
            f.write(str(e))
            f.write(" ")
        f.write("\n")
    f.close()


def get_embeddings(mdl, fm, ofn, embedding_layer=2, fn = None, vob = None):
    '''
    extract embeddings
    :param mdl : the model
    :fm : feature manager
    :param fn: file containing words that need to be represented as embeddings/ Word,Role separated by a space
    :return: the new file containing  the embeddings of the verbs
    '''

    if fn is not None:
        vobs = set()

        f = open(fn, "r")
        for l in f.readlines():
            tmps = l.split(" ")
            for tmp in tmps:
                if tmp != "":
                    tmpps = tmp.split (",")
                    if len(tmpps) ==2:

                        vobs.add((tmpps[0],tmpps[1]))

        vobs = list(vobs)
    else:
        if vob is not None:
            vobs = vob

    vv = []

    for v in vobs:  # vobs must be a pair of word and label
        if v[0] + "_" + v[1] in fm.f.map.keys():
            vv.append(v[0] + "_" + v[1])

    vobs = vv
    print (vobs)

    X = [ [v] for v in vobs]

    Y = [["EOS"] for i in range(len(X))]

    X = [[fm.f.map[fm.f.getFeatureValue(x)] +1 for x in XX] for XX in X ]
    Y = [[fm.fY.map[fm.fY.getFeatureValue(x)] + 1 for x in XX] for XX in Y]

    x,x_mask,y,y_mask = preprare_seq_seq_data(X,Y)
    x, y,  mask_x,mask_y, _, _,_, _ = mdl.standardize_data(x, y, x_mask, y_mask, None,None, None,None)
    rs = mdl.get_output_layer(embedding_layer, x, mask_x)
    f = open (ofn, "w")
    for i in range(len(vobs)):
        w = vobs[i]
        em = rs[0][i]
        f.write(w + " ")
        for e in em:
            f.write(str(e))
            f.write(" ")
        f.write("\n")


    f.close()


########### functions to calculate preference selectional scores  ###########

def calculate_selectional_preferences(config_path, model_path, list_preds, vob, output ):
    cfg = read_config(config_path)

    mdl, fm = loadSemLM11 (model_path, cfg['train'], cfg['valid'], cfg['we_dict'], cfg['dep'],
                 cfg['hidden_size'], cfg['batch_size'], cfg['save_folder'])

    for p in list_preds:
        scores  = get_probability_is_argument(mdl, fm, p)

        process_probability(fm, scores,  p, output + "/" + p + ".out.txt", vobs=vob[p])


def process_probability(fm, scores, pred, output, vobs):
    f = open(output, "w")
    '''
    scores =[]
    for l in range(levels):
        print ('reading ...', l)
        scores_l = read_probability_file(dir, pred, l)
        scores.append(scores_l)
    '''


    for arg in fm.fY.map.keys():
        if arg != "EOS":
            tmps = arg.split ("_")
            k = ""
            for t in range(len(tmps)-1):
                k += tmps[t]

            if not k in vobs:
                continue
            pos = fm.fY.map[arg]+1

            final_score = 0.0
            exact_prob0 =  get_score( scores[0][0],  pos)[0] # probability at the first position
            final_score+=exact_prob0
            continue_probs0  = np.asarray( scores[0][1])


            start_probs1 = continue_probs0

            exact_prob1 =  get_score( scores[1][0],  pos)[0] # probability at the first position
            final_score+=np.sum(exact_prob1 * start_probs1)
            continue_probs1  = np.asarray( scores[1][1])

            start_probs1 = start_probs1.flatten().repeat(1)

            start_probs2 = start_probs1 * continue_probs1.flatten()



            exact_prob2 =  get_score( scores[2][0],  pos)[0] # probability at the first position
            final_score+=np.sum(exact_prob2 * start_probs2)
            continue_probs2  = np.asarray( scores[2][1])

            start_probs2 = start_probs2.repeat(1)

            start_probs2 = start_probs2 * continue_probs2.flatten()



            prob = final_score
            print (arg + ": "+ str(prob))
            f.write(arg)
            f.write(":")
            f.write(str(prob))
            f.write(" ")

    f.close()




def get_score(sl, pos):
        rs =[]
        print('len ', len(sl))
        for allval in sl:
            for val in allval[1]:

                if val[0]==pos:
                    rs.append(val[1])
        return np.asarray(rs)






def get_probability_is_argument(mdl, fm, predicate_words,pos=5):
    '''
    calculate the probability that a word is an argument of a predicate
    :param mdl:
    :param fm:

    :return:
    '''
    # calculate the score at pos 0
    #
    X = []

    X_new = []


    X.append([] )
    X_new.append([[predicate_words , "PRED" ]])

    score = []
    for p in range(pos):
        n=1

        X_new, rs_scores, X, myscores = get_scores_all(mdl, fm, X, X_new,  num_select=n)
        score.append((myscores, rs_scores))

    return score



def to_string(lst):
    s = ""
    for l in lst:
        s += l + ","
    s = s[0:len(s)-1]
    return s


def get_scores_all(mdl, fm, X, X_new,   num_select = 10):
    #f = open(output, "w")
    X1 = []

        # x = X[i], we add new values to the end of x

    for j in range(len(X_new)):

            for k in range(len(X_new[j])):
                xx =[ xxx for xxx in X[j]]

                xx.append(X_new[j][k][0] + "_" + X_new[j][k][1] )
                X1.append(xx)

    X = [[fm.f.map[fm.f.getFeatureValue(x)] +1 for x in XX] for XX in X1 ]

    x,x_mask= preprare_seq_seq_data(X)

    x, _,  mask_x,_, _, _,_, _ = mdl.standardize_data(x, None, x_mask, None, None,None, None,None)

    score_pos=mdl.get_output_layer(-1, x, mask_x)
    score_pos=score_pos.swapaxes(0,1)
    score_pos = score_pos[:,-1]
    print (score_pos)
    x = T.matrix("score")

    sort_f = th.function([x], T.argsort(x))

    sorted_values = sort_f(score_pos)
    sorted_values = sorted_values
    print (sorted_values)
    rs = []
    rs_scores = []
    my_scores = []
    for i in range(sorted_values.shape[0]):
        #f.write(to_string(X1[i]) + " ")
        ss=[]
        for j in range(1,sorted_values.shape[1]+1):
            val = sorted_values[i][sorted_values.shape[1]-j]

            #val_map = fm.fY.map_inversed[val-1]
            score = score_pos[i][val]
            #f.write(str(val) + ":" + str(score) + " ")
            ss.append((val,score))
        #f.write("\n")
        my_scores.append((to_string(X1[i]), ss))



        vals = []
        c = 0
        for t in range(sorted_values.shape[1]-1, -1, -1):
            if c == num_select:
                break
            v = sorted_values[i][t]

            if fm.fY.map_inversed[v-1]!="EOS":
                vals.append(v)
                c+=1
        #vals = sorted_values[i][sorted_values.shape[1]-num_select:sorted_values.shape[1]]

        val_maps = [fm.fY.map_inversed[v-1].split("_") for v in list(vals) ]#if  fm.fY.map_inversed[v-1]!="EOS" ]
        scores = [score_pos[i][v] for v in list(vals)]# if fm.fY.map_inversed[v-1]!="EOS"]
        rs.append(val_maps)
        rs_scores.append(scores)

    return rs, rs_scores,   X1, my_scores


    # calculate scores for pos 1


def get_scores(mdl, fm, X, X_new, w_lbl,  num_select = 10):
    X1 = []

        # x = X[i], we add new values to the end of x

    for j in range(len(X_new)):

            for k in range(len(X_new[j])):
                xx =[ xxx for xxx in X[j]]
                xx.append(X_new[j][k][0] + "_" + X_new[j][k][1] )
                X1.append(xx)

    X = [[fm.f.map[fm.f.getFeatureValue(x)] +1 for x in XX] for XX in X1 ]

    x,x_mask= preprare_seq_seq_data(X)

    x, _,  mask_x,_, _, _,_, _ = mdl.standardize_data(x, None, x_mask, None, None,None, None,None)

    score_pos=mdl.get_output_layer(-1, x, mask_x)
    score_pos=score_pos.swapaxes(0,1)
    score_pos = score_pos[:,-1]
    #print (score_pos)
    x = T.matrix("score")

    sort_f = th.function([x], T.argsort(x))

    sorted_values = sort_f(score_pos)
    sorted_values = sorted_values
    #print (sorted_values)
    rs = []
    rs_scores = []
    for i in range(sorted_values.shape[0]):
        vals = sorted_values[i][sorted_values.shape[1]-num_select:sorted_values.shape[1]-1]

        val_maps = [fm.fY.map_inversed[v-1].split("_") for v in list(vals) if fm.fY.map_inversed[v-1]!=w_lbl and fm.fY.map_inversed[v-1]!="EOS" ]
        scores = [score_pos[i][v] for v in list(vals) if fm.fY.map_inversed[v-1]!=w_lbl and fm.fY.map_inversed[v-1]!="EOS"]
        rs.append(val_maps)
        rs_scores.append(scores)

    return rs,  rs_scores,  score_pos[:,fm.fY.map[fm.fY.getFeatureValue(w_lbl)] + 1 ], X1


