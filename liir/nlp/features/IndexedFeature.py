__author__ = 'quynhdo'

from liir.nlp.features.Feature import Feature
import numpy as np

class IndexedFeature(Feature):
    def __init__(self, fn, get_feature_value_func, is1Hot=True,  **kwargs):
        Feature.__init__(self, fn, get_feature_value_func)
        self.map = {}  # mapping from feature string - index
        self.map_inversed = {}
        self.current_index = 0
        self.is1Hot = is1Hot
        self.func_kwargs = kwargs




    def addFeatureValueToMap(self, s):
        '''
        add a feature string to the map
        :param s:
        :return:
        '''
        if isinstance(s, list) or isinstance(s, set):
            for v in s:
                self.addFeatureValueToMap(v)
        else:
            if s not in self.map.keys():
                self.map[s] = self.current_index
                self.map_inversed[self.current_index] = s
                self.current_index += 1

    def get_dim(self):
        return self.current_index

    def getRepresentation(self, s):
        if isinstance(s, list) or isinstance(s, set):
            if not self.is1Hot:
                return [self.map[v] for v in s]
            else:
                arr = np.zeros((self.get_dim()), dtype='int')
                for v in s:
                    arr[self.map[v]]=1
                return arr
        else:
            if not self.is1Hot:
                return [self.map[s]]
            else:
                arr = np.zeros((self.get_dim()), dtype='int')
                arr[self.map[s]]=1
                return arr


    def getValueAndRepresentation(self, ins, indices=None, offset=None):
        if indices is None:
            s = self.getFeatureValue(ins)
            return self.getRepresentation(s)
        else:
            s = self.getFeatureValue(ins)

            if isinstance(s, list) or isinstance(s, set):
                    for v in s:
                        if v in self.map.keys():
                            indices.add(offset + self.map[v])


            else:
                if s in self.map.keys():

                    indices.add(offset + self.map[s])
