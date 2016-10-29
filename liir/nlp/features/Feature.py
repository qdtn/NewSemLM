__author__ = 'quynhdo'
class Feature(object):

    def __init__(self,fn, get_feature_value_func, **kwargs):
        '''
        :param fn: Feature Name
        :return:
        '''
        self.feature_name = fn
        self.dim = 0
        self.getFeatureValueFunc=get_feature_value_func
        self.func_kwargs = kwargs

    def getFeatureValue(self, ins):
        return self.getFeatureValueFunc(ins, self.func_kwargs)

    def getRepresentation(self, s):
        raise NotImplementedError("Subclasses should implement this!")

    def getValueAndRepresentation(self, ins):
        s = self.getFeatureValue(ins)
        return self.getRepresentation(s)

    def get_dim(self):
        return self.dim