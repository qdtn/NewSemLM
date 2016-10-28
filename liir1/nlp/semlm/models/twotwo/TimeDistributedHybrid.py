__author__ = 'quynhdo'
import theano as th

from liir.ml.core.layers.Layer import Layer
from utils.Data import numpy_floatX

__author__ = 'quynhdo'
# an embedding layer


class TimeDitributedHybrid(Layer):
    def __init__(self, idx="0", core_layer=None):
        Layer.__init__(self, id=idx)
        self.core_layer = core_layer
        self.extra_input = None
        self.semantic_prediction = None
        self.hybrid_results = None
        self.lbl_idx = None

    def compile(self):
        #####
        if self.mask is not None:
            self.core_layer.mask = self.mask
        self.core_layer.id = self.id

        self.core_layer.init_params()
        #####
        if isinstance(self.input, list):
            time_steps = len(self.input)
        else:
            time_steps =  self.input.shape[0]


        def _step(_m, _em, _s):
            self.core_layer.input = _m
            self.core_layer.extra_input = _em
            self.core_layer.semantic_prediction = _s
            self.core_layer.compile(init_params=False)
            o =  [self.core_layer.output]


            for oo in self.core_layer.output_hybrids:
                o.append(oo)

            return o



        result_hybrid = []
        for i in range(len(self.core_layer.semantic_label_map.keys())+1):
            xx= None
            result_hybrid.append(xx)

        (result, updates) = th.scan(fn=_step,

                              sequences=[self.input, self.extra_input, self.semantic_prediction]
                              )



        #self.output = result.swapaxes(0,1)
        self.output = result[0]
        self.params = self.core_layer.params
        self.output_hybrid = result[1:]







