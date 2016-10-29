__author__ = 'quynhdo'

from liir.nlp.features.Feature import Feature
import numpy as np




class WEFeature(Feature):
    def __init__(self, fn, f, we_dict, ope="mean"):
        Feature.__init__(self, fn, None)
        self.wedict = we_dict
        self.ope = ope
        self.f = f
    def getFeatureValue(self, ins):
        return self.f.getFeatureValue(ins)

    def get_dim(self):
        return self.wedict.we_size

    def getRepresentation(self, s):
        if s is None:
            return np.zeros(self.wedict.we_size)
        if isinstance(s, list) or isinstance(s, set):
            if len(s) == 0:
                if self.ope == "mean" or self.ope == "sum":
                    return np.zeros(self.wedict.we_size)
            arr = []
            for vv in s:
                av = self.wedict.getWE(vv)
                arr.append(av)

            arr = np.asarray(arr)

            if self.ope == "mean":
                arr = np.mean(arr, axis=0)

            if self.ope == "sum":
                arr = np.sum(arr, axis=0)

        else:

            arr = self.wedict.getWE(s)

        return arr





