__author__ = 'quynhdo'

from liir.nlp.features.Feature import Feature

class CoFeature(Feature):
    def __init__(self, fn, f1, f2, join_symbol = "_"):
        Feature.__init__(self, fn, None)
        self.f1=f1
        self.f2=f2
        self.join_symbol = join_symbol


    def getFeatureValue(self, ins):
        val1 = self.f1.getFeatureValue(ins)
        val2 = self.f2.getFeatureValue(ins)


        if val1 is None:
            val1 =  'null'
        if val2 is None:
            val2 = 'null'

        if isinstance(val1, str) and  isinstance(val2, str):
            return val1 + self.join_symbol + val2


        if ((isinstance(val1, list) or isinstance(val1, set) ) and (isinstance(val2, list) or isinstance(val2, set) ) ):
            rs =[]

            if len(val1)== len(val2):
                for i in range(len(val1)):
                    rs.append(val1[i] +  self.join_symbol + val2[i])
            return rs

        strval = ""
        setval = []
        if (isinstance(val1, str) and isinstance(val2, set) ) or (isinstance(val1, str) and isinstance(val2, list) ):
            strval = val1
            setval = val2

        if (isinstance(val2, str) and isinstance(val1, set) ) or (isinstance(val2, str) and isinstance(val1, list) ):
            strval = val2
            setval = val1
        rs = []
        if len(setval) == 0:
            rs.append("")

        for sv in setval:
            rs.append(sv + self.join_symbol + strval)

        return rs

