import theano as th
import numpy as np

from liir.ml.core.Option import Option
from liir.ml.core.layers.Layer import Layer
from utils.Functions import WeightInit, getFunction
from utils.Variant import _p
from theano import tensor

__author__ = 'quynhdo'


class SoftmaxHybrid (Layer):
    ####
    # A SoftmaxHybrid Layer,
    # It will receive as input the hidden layers of LSTM layer A, and the hidden layers of LSTM layer B, and the output of a TimeDistributedLayer which runs on top of B
    ####
    def __init__(self, input_dim1, input_dim2, output_dim,  semantic_label_map,
                 w_lamda=1.0, rng=None,  idx="0", activation="sigmoid",
                 transfer="dot_transfer"):
        Layer.__init__(self, idx=idx, output_dim=output_dim)

        self.activation = activation
        self.transfer = transfer

        self.W = None
        self.Ws = None
        self.b = None

        # params to initialize weights
        self.w_lamda = w_lamda
        self.rng = rng  # numpy random
        self.semantic_label_map = semantic_label_map
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2

        self.extra_input = None
        self.semantic_prediction = None

        self.output_func= None

        #self.pos_A0 = 0
        #self.pos_A1 = 0
        #self.pos_A2 = 0
        #self.pos_A3 = 0
        #self.pos_A4 = 0
        #self.pos_A5 = 0


        self.output_hybrids = []

    def init_params(self, initial_w=None, initial_b=None):
        # init params
        self.W = th.shared(value=np.asarray(WeightInit(self.input_dim1,
                                                           self.option[Option.OUTPUT_DIM],
                                                           self.w_lamda, self.rng), dtype=th.config.floatX),
                                                           name=_p(self.id,"W"), borrow=True)



        self.params[_p(self.id, "W")] = self.W


        self.Ws = np.asarray([np.asarray(WeightInit(self.input_dim1,
                                                           self.input_dim1,
                                                           self.w_lamda, self.rng), dtype=th.config.floatX)
                           for i in range(len(self.semantic_label_map.keys())+1) ])

        self.Ws = th.shared(self.Ws, _p(self.id, 'Ws'))
        self.params[_p(self.id, "Ws")] = self.Ws



        self.V = th.shared(value=np.asarray(WeightInit(self.input_dim2,
                                                           self.option[Option.OUTPUT_DIM],
                                                           self.w_lamda, self.rng), dtype=th.config.floatX),
                                                           name=_p(self.id,"V"), borrow=True)



        self.params[_p(self.id, "V")] = self.V


        self.Vs = np.asarray([np.asarray(WeightInit(self.input_dim2,
                                                           self.input_dim2,
                                                           self.w_lamda, self.rng), dtype=th.config.floatX)
                           for i in range(len(self.semantic_label_map.keys()) +1 ) ])

        self.Vs = th.shared(self.Vs, _p(self.id, 'Vs'))

        self.params[_p(self.id, "Vs")] = self.Vs



        if initial_b is not None:
            self.b = initial_b
        else:
            self.b = th.shared(value=np.zeros(self.option[Option.OUTPUT_DIM], dtype=th.config.floatX), name=_p(self.id,"b"), borrow=True)

            self.params[ _p(self.id, "b")]=self.b

    def compute_output2(self):

        label_results =  self.process_label_results(self.semantic_prediction)#tensor.round(self.semantic_prediction)



        label_specific_Ws = tensor.tensordot(label_results, self.Ws, axes=[1,0])

        label_specific_Vs = tensor.tensordot(label_results, self.Vs, axes=[1,0])


        label_specific_W = th.dot (label_specific_Ws, self.W)

        label_specific_V = th.dot (label_specific_Vs, self.V)


        # compute output
        self.output = getFunction('softmax')(tensor.batched_dot(self.input, label_specific_W)   +  tensor.batched_dot(self.extra_input, label_specific_V)   +
                       self.b)

     #   self.get_output()


    def process_label_results(self, x):
        axis = 1

        # Identify the largest value in each row
        x_argmax = tensor.argmax(x, axis=axis, keepdims=True)

        # Construct a row of indexes to the length of axis
        indexes = tensor.arange(x.shape[axis]).dimshuffle(
            *(['x' for dim1 in range(axis)] + [0] + ['x' for dim2 in range(x.ndim - axis - 1)]))


        # Create a binary mask indicating where the maximum values appear
        mask = tensor.eq(indexes, x_argmax)
        return mask

    def process_label_results_idx(self, x, idx):
        axis = 1

        # Identify the largest value in each row

        # Construct a row of indexes to the length of axis
        indexes = tensor.arange(x.shape[axis]).dimshuffle(
            *(['x' for dim1 in range(axis)] + [0] + ['x' for dim2 in range(x.ndim - axis - 1)]))

        indexes = tensor.repeat(indexes, x.shape[0], axis=0)


        # Create a binary mask indicating where the maximum values appear
        mask = tensor.eq(indexes, idx)
        return mask

    def compute_output(self):

        label_results =  self.process_label_results(self.semantic_prediction)#tensor.round(self.semantic_prediction)
        print (label_results)
        print (tensor.round(self.semantic_prediction))

        label_specific_Ws = tensor.tensordot(label_results, self.Ws, axes=[1,0])

        label_specific_Vs = tensor.tensordot(label_results, self.Vs, axes=[1,0])


        label_specific_W = th.dot (label_specific_Ws, self.W)

        label_specific_V = th.dot (label_specific_Vs, self.V)


        # compute output
        self.output = getFunction('softmax')(tensor.batched_dot(self.input, label_specific_W)   +  tensor.batched_dot(self.extra_input, label_specific_V)   +
                       self.b)

        for i in range(len(self.semantic_label_map.keys())+1):
            ho = self.get_output(i)
            self.output_hybrids.append(ho)



    def get_output(self, idx):

        label_results =  self.process_label_results_idx(self.semantic_prediction, idx)#tensor.round(self.semantic_prediction)

        label_specific_Ws = tensor.tensordot(label_results, self.Ws, axes=[1,0])

        label_specific_Vs = tensor.tensordot(label_results, self.Vs, axes=[1,0])


        label_specific_W = th.dot (label_specific_Ws, self.W)

        label_specific_V = th.dot (label_specific_Vs, self.V)


        # compute output
        self.output_func = getFunction('softmax')(tensor.batched_dot(self.input, label_specific_W)   +  tensor.batched_dot(self.extra_input, label_specific_V)   +
                       self.b)

        return self.output_func

if __name__ == "__main__":

    x = tensor.tensor3('x')
    y = tensor.tensor3('y')
    f = th.function([x,y], tensor.batched_dot(y,x))


    print( "f...")
    print ("...")

    k =tensor.tensor4( 'f')


    w = tensor.matrix('w')
    ff = th.function([w,x,y], tensor.dot(tensor.tensordot(y,x, axes=[2,1]),w))
    x = np.asarray([[[1,2,4],[2,3,3]],[[5,6,6],[7,8,9]]])


    print (x)
    print(x.shape)

    y = np.asarray([[[0,1],[1,0]],[[0,1],[1,0]]])

    print (y)
    print(y.shape)

    print ("my test")
    print (f(x,y))
    print ("my test")

    xx = np.asarray([[1,2],[3,4]])
    print (xx)
    y = np.asarray([[[0,1],[1,0]], [[0,1],[1,0]]])

    print (f(x,y))

    print (ff(xx,x,y))



