from liir.nlp.we.WEDict import WEDict
from theano.tensor.shared_randomstreams import RandomStreams
from liir1.nlp.semlm.data.DataManager import generate_sequential_data, preprare_seq_seq_data
from liir1.nlp.semlm.features.FeatureManager import FeatureManager
from liir1.nlp.semlm.models.twotwo.SoftmaxHybrid import SoftmaxHybrid
from liir1.nlp.semlm.models.twotwo.TimeDistributedHybrid import TimeDitributedHybrid

__author__ = 'quynhdo'
import sys
from theano import tensor
import time
from liir.ml.core.options.Option import Option
from liir.ml.core.layers.Dense import Dense
from liir.ml.core.layers.Dropout import Dropout
from liir.ml.core.layers.Embedding import Embedding
from liir.ml.core.layers.LSTM import LSTM
from liir.ml.core.layers.Model import Sequential
from liir.ml.core.layers.TimeDistributed import TimeDitributed

from optimizer.Optimizer2 import getOptimizer
from utils.Data import get_minibatches_idx
from utils.Functions import getFunction
import theano as th
import numpy as np
__author__ = 'quynhdo'

class Model22(Sequential):

    def __init__ (self, input_dim1, input_dim2, output_dim1, output_dim2, hidden_dim1, hidden_dim2, semantic_label_map, dep=2, loss="nl", optimizer="ada", we_dict1=None, we_dict2=None, map1=None, map2=None, use_noise=True):
        Sequential.__init__(self, use_mask=True, input_value_type="int32", prediction_type="vector",
                            prediction_value_type='int32', use_noise=use_noise)

        self.extra_input = tensor.matrix('x_extra', dtype="int32")
        self.extra_output = None
        self.extra_prediction = tensor.ivector('ye')
        self.extra_gold = tensor.ivector('ge')


        self.option[Option.LOSS] = loss
        self.option[Option.OPTIMIZER] = optimizer
        l1 = Embedding(input_dim1, hidden_dim1, we_dict=we_dict1, map=map1)

        l2 = Embedding(input_dim2, hidden_dim2, we_dict=we_dict2, map=map2)

        self.add_layer([l1,l2])

        l11 = Dropout(hidden_dim1,theano_rng= RandomStreams(128))
        self.add_layer(l11)

        l21 = Dropout(hidden_dim2,theano_rng= RandomStreams(128))
        self.add_layer(l21)


        l4 = LSTM (hidden_dim1, hidden_dim1, return_sequences=True)

        l5 = LSTM (hidden_dim2, hidden_dim2, return_sequences=True)

        self.add_layer([l4,l5])
        l22 = Dropout(hidden_dim2,theano_rng= RandomStreams(128))
        self.add_layer(l22)
        l6 = TimeDitributed(core_layer=Dense(hidden_dim2, output_dim2, activation="softmax"))



        l12 = Dropout(hidden_dim1,theano_rng= RandomStreams(128))
        self.add_layer(l12)

        l7 = SoftmaxHybrid(hidden_dim1, hidden_dim2, output_dim1, semantic_label_map)




        l8 = TimeDitributedHybrid(core_layer=l7)
        self.add_layer([l6,l8])

        self.option[Option.IS_SEQUENCE_WORK] = True

    def compile_layers(self):
        self.layers[0].compile()      # for word
        self.layers[1].compile() # for label

        self.layers[2].input =  self.layers[0].output  # dropout for word
        self.layers[2].compile()

        self.layers[3].input =  self.layers[1].output  # dropout for label
        self.layers[3].compile()



        self.layers[4].input =  self.layers[2].output  # hidden word
        self.layers[4].compile()
        self.layers[5].input =  self.layers[3].output # hidden label
        self.layers[5].compile()


        self.layers[6].input =  self.layers[4].output  # dropout for lstm word
        self.layers[6].compile()

        self.layers[7].input =  self.layers[5].output  # dropout for lstm label
        self.layers[7].compile()




        self.layers[8].input =  self.layers[7].output  # output label
        self.layers[8].compile()

        self.layers[9].input = self.layers[6].output
        self.layers[9].extra_input = self.layers[7].output
        self.layers[9].semantic_prediction = self.layers[8].output

        self.layers[9].compile()





    def compile(self,  compile_layer=True):
        # set sequential ids to all layers in the model
        for i in range(len(self.layers)):
            self.layers[i].id = str(i)

        #if self.option[Option.SAVE_TOPOLOGY] != "":

        #    with open(self.option[Option.SAVE_TOPOLOGY], 'wb') as f:
        #                    pickle.dump(self, f)

        if self.use_mask:
            self.input_mask = tensor.imatrix('i_mask')
            if self.prediction_type == "vector":
                self.output_mask = tensor.ivector('o_mask')
            else:
                self.output_mask = tensor.imatrix('o_mask')

        # start passing the data
        self.layers[0].input = self.input
        self.layers[1].input = self.extra_input

        if self.use_mask:
            for l in self.layers:
                l.mask = self.input_mask

        if self.use_noise is not None:
            for l in self.layers:
                if isinstance(l, Dropout):
                    l.use_noise = self.use_noise

        if compile_layer:
            self.compile_layers()

        for l in self.layers:
            for kk, pp in l.params.items():
                    self.params[kk] = pp

        self.output = self.layers[9].output
        self.extra_output = self.layers[8].output


        if self.option[Option.IS_SEQUENCE_WORK]:
            self.output = tensor.reshape(self.output,(-1, self.output.shape[-1]))
            self.extra_output = tensor.reshape(self.extra_output,(-1, self.extra_output.shape[-1]))

        if self.prediction_type == "vector":
            self.prediction = tensor.argmax(self.output, axis=-1)
            self.extra_prediction = tensor.argmax(self.extra_output, axis=-1)


        if self.use_mask:
            self.cost = getFunction(self.option[Option.LOSS])(self.output, self.gold, mask=self.output_mask)
            self.extra_cost = getFunction(self.option[Option.LOSS])(self.extra_output, self.extra_gold, mask=self.output_mask)
            self.total_cost = self.cost + self.extra_cost
            self.f_cost = th.function([self.input, self.extra_input, self.gold, self.extra_gold, self.input_mask, self.output_mask], self.total_cost, name='f_cost')
            #self.f_cost = th.function([self.input, self.gold, self.input_mask], self.cost, name='f_cost')
            grads = tensor.grad(self.total_cost, wrt=list(self.params.values()))
            self.f_grad = th.function([self.input, self.extra_input, self.gold, self.extra_gold, self.input_mask, self.output_mask], grads, name='f_grad')
            self.f_grad_shared, self.f_update = getOptimizer(self.option[Option.OPTIMIZER])(self.lr, self.params, grads,



                                              [self.input, self.extra_input, self.gold, self.extra_gold, self.input_mask, self.output_mask], self.total_cost)


    def evaluation(self,x1,x2, y1,y2,  input_mask = None, output_mask=None):
        #x,y,input_mask,output_mask,_,_,_,_ = self.standardize_data(x,y,input_mask,output_mask, None, None,None,None)

        if self.use_noise is not None:
            self.use_noise.set_value(0.)

        error_mask =  (tensor.neq(self.prediction, self.gold) * self.output_mask).sum() / self.output_mask.sum()

        error_mask_extra =  (tensor.neq(self.extra_prediction, self.extra_gold) * self.output_mask).sum() / self.output_mask.sum()



        test_model = th.function(
                    inputs=[self.input,self.extra_input, self.gold, self.extra_gold, self.input_mask, self.output_mask],
                    outputs=error_mask + error_mask_extra,
                )
        return test_model(x1,x2, y1,y2, input_mask, output_mask)

    def fit_shuffer(self, X1,X2, Y1, Y2,  X_valid1=None,  X_valid2=None,Y_valid1=None, Y_valid2=None,process_data_func=None):
        # in this fit function, data will be shuffered
        history_errs = []
        best_p = None
        bad_count = 0
        uidx = 0  # the number of update done
        estop = False  # early stop

        start_time = time.time()

        try:
            for eidx in range(self.option[Option.MAX_EPOCHS]):
                n_samples = 0

                # Get new shuffled index for the training set.
                kf = get_minibatches_idx(len(X1), self.option[Option.BATCH_SIZE], shuffle=True)
                for _, train_index in kf:
                    if self.use_noise is not None:
                        self.use_noise.set_value(1.)
                    uidx += 1

                    # Select the random examples for this minibatch
                    y1 = [Y1[t] for t in train_index]
                    x1 = [X1[t]for t in train_index]
                    y2 = [Y2[t] for t in train_index]
                    x2 = [X2[t]for t in train_index]


                    x1, mask_x, y1, mask_y = process_data_func(x1, y1)

                    x2, mask_x, y2, mask_y = process_data_func(x2, y2)

                    x1, y1,  mask_x,mask_y, _, _,_, _ = self.standardize_data(x1, y1, mask_x, mask_y, None,None, None,None)

                    x2, y2,  _,_, _, _,_, _ = self.standardize_data(x2, y2, None, None, None,None, None,None)

                    '''
                    print(x.shape)
                    m  = self.get_output_layer(1,x, mask_x)
                    print (m.shape)
                    m  = self.get_output_layer(2,x, mask_x)
                    print (m.shape)
                    '''
                    cost = self.f_grad_shared(x1, x2,  y1, y2, mask_x, mask_y)


                    self.f_update(self.option[Option.LRATE])
                    n_samples += x1.shape[1]
                    if np.isnan(cost) or np.isinf(cost):
                        print('bad cost detected: ', cost)
                        self.load_params(self.option[Option.SAVE_TO])
                        return 1., 1., 1.


                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                    if uidx % self.option[Option.SAVE_FREQ]== 0:
                        self.save_params(self.option[Option.SAVE_TO])
                        train_err = self.evaluation(x1,x2,y1,y2,mask_x,mask_y)
                        print ("train error: ", train_err)


                    if X_valid1 is not None:

                        if uidx % self.option[Option.VALID_FREQ]== 0:
                            if self.use_noise is not None:
                                self.use_noise.set_value(0.)

                            nb_valid_batchs =  len(X_valid1) //  self.option[Option.BATCH_SIZE]

                            start = 0
                            end = 0
                            valid_rs = []




                            for g in range(nb_valid_batchs+1):
                                start = g * self.option[Option.BATCH_SIZE]
                                end =  start + self.option[Option.BATCH_SIZE]
                                if end > len(X_valid1):
                                    end = len(X_valid1)

                                X_valid1t = X_valid1[start:end]
                                Y_valid1t= Y_valid1[start:end]
                                Y_valid2t= Y_valid2[start:end]
                                X_valid2t = X_valid2[start:end]


                                x_v1, mask_vx, y_v2, mask_vy = process_data_func(X_valid1t, Y_valid1t)

                                x_v2, mask_vx, y_v2, mask_vy = process_data_func(X_valid2t, Y_valid2t)

                                x_v1, y_v1,  mask_vx,mask_vy, _, _,_, _ = self.standardize_data(x_v1, y_v1, mask_vx, mask_vy, None,None, None,None)

                                x_v2, y_v2,  _,_, _, _,_, _ = self.standardize_data(x_v2, y_v2, None, None, None,None, None,None)
                                valid_err = self.evaluation(x_v1,x_v2,y_v1, y_v2,mask_vx,mask_vy)
                                valid_rs.append(valid_err)
                                print (valid_rs)

                            valid_err = np.mean(np.asarray(valid_rs))


                            history_errs.append([valid_err])

                            if (best_p is None or
                                valid_err <= np.array(history_errs)[:,
                                                                       0].min()):

                                #with open(self.option[Option.SAVE_TO], 'wb') as f:
                                #        pickle.dump(self, f)
                                self.save_params(self.option[Option.SAVE_BEST_VALID_TO])
                                bad_counter = 0

                            print( ('Train ', train_err, 'Valid ', valid_err,
                                   ) )

                            if (len(history_errs) > self.option[Option.PATIENCE] and
                                valid_err >= np.array(history_errs)[:-self.option[Option.PATIENCE],
                                                                       0].min()):
                                bad_counter += 1
                                if bad_counter > self.option[Option.PATIENCE]:
                                    print('Early Stop!')
                                    estop = True
                                    break


                print('Seen %d samples' % n_samples)

                if estop:
                    break

        except KeyboardInterrupt:
            print("Training interupted")

        end_time = time.time()



        #self.use_noise.set_value(0.)
        print('The code run for %d epochs, with %f sec/epochs' % (
            (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
        print( ('Training took %.1fs' %
                (end_time - start_time)), file=sys.stderr)

    def get_output_layer(self, layer_id, x1, x2, input_mask=None):
        if input_mask is None:
            l = self.layers[layer_id]
            output_func =  th.function([], l.output, givens={
                self.input: x1,
                self.extra_input:x2,


            })
            return output_func()
        else:
            l = self.layers[layer_id]
            print (self.input)
            print (self.extra_input)
            print (self.input_mask)
            output_func =  th.function([], l.output, givens={
                self.input: x1,
                self.extra_input:x2,
                self.input_mask:input_mask,




            },on_unused_input='warn')
            return output_func()

    def get_hybrid_scores(self, x1,x2, input_mask=None):

            l = self.layers[-1]




            output_func =  th.function([], l.output_hybrid, givens={
                self.input: x1,
                self.extra_input:x2,
                self.input_mask:input_mask,




            },on_unused_input='warn')
            return output_func()


def trainSemLM22(train_texts, valid_texts, we_dict_path1=None, we_dict_path2=None, dep=1,hidden_size1=32, hidden_size2=32, batch_size=200, save_folder=".", model_name="hybrid", max_epochs=120, load_dt=True):

    fm1 = FeatureManager()
    fm2 = FeatureManager()

    all_texts = train_texts + valid_texts
    X1,X2,Y1,Y2 = generate_sequential_data(all_texts, type="2-2", loaddata=load_dt)

    for i in range(len(X1)):
        fm1.extract_features(X1[i],Y1[i])

    for i in range(len(X2)):
        fm2.extract_features(X2[i],Y2[i])


    X1,X2, Y1, Y2= generate_sequential_data(train_texts,  type="2-2", loaddata=load_dt)

    X1 = [[fm1.f.map[fm1.f.getFeatureValue(x)] +1 for x in XX] for XX in X1 ]
    Y1= [[fm1.fY.map[fm1.fY.getFeatureValue(x)] + 1 for x in XX] for XX in Y1]

    X2 = [[fm2.f.map[fm2.f.getFeatureValue(x)] +1 for x in XX] for XX in X2 ]
    Y2= [[fm2.fY.map[fm1.fY.getFeatureValue(x)] + 1 for x in XX] for XX in Y2]




    X1v,X2v, Y1v, Y2v=generate_sequential_data(valid_texts, type="2-2", loaddata=load_dt)




    X1v = [[fm1.f.map[fm1.f.getFeatureValue(x)] +1 for x in XX] for XX in X1v ]
    Y1v = [[fm1.fY.map[fm1.fY.getFeatureValue(x)] + 1 for x in XX] for XX in Y1v]

    X2v = [[fm2.f.map[fm2.f.getFeatureValue(x)] +1 for x in XX] for XX in X2v ]
    Y2v = [[fm2.fY.map[fm2.fY.getFeatureValue(x)] + 1 for x in XX] for XX in Y2v]


    we_dict1=None
    we_dict2=None
    if we_dict_path1 is not None:
        we_dict1 = WEDict(we_dict_path1)

    if we_dict_path2 is not None:
        we_dict2 = WEDict(we_dict_path2)

    mdl =Model22(fm1.f.current_index + 1, fm2.f.current_index + 1,
                  fm1.fY.current_index + 1, fm2.fY.current_index + 1,
                  hidden_size1, hidden_size2, dep=dep, we_dict1=we_dict1,we_dict2=we_dict2,map1=fm1.f.map, map2=fm2.f.map,
                  semantic_label_map=fm2.fY.map)

    mdl.option[Option.SAVE_TO] = save_folder + "/" + model_name + ".pkl"
    mdl.option[Option.SAVE_FREQ] = 20
    mdl.option[Option.VALID_FREQ] = 20
    mdl.option[Option.BATCH_SIZE] = batch_size
    mdl.option[Option.SAVE_BEST_VALID_TO] = save_folder + "/" + "best_" + model_name + ".pkl"
    mdl.option[Option.MAX_EPOCHS]=max_epochs
    mdl.compile()
    func = preprare_seq_seq_data
    mdl.fit_shuffer(X1,X2, Y1, Y2, X1v,X2v, Y1v , Y2v, process_data_func=func)
