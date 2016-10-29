from liir.nlp.features.FeatureName import FeatureName
from liir.nlp.features.IndexedCoFeature import IndexedCoFeature
from liir.nlp.features.IndexedFeature import IndexedFeature
from liir.nlp.features.WEFeature import WEFeature
from liir.nlp.features.WordFeatureFuncs import getWordFeatureFunc, getWordPairFeatureFunc, getNeighbourFeatureFunc

__author__ = 'quynhdo'
class FeatureGeneratorBase(list):
    def __init__(self, feature_file=None, we_dicts = None):
        list.__init__(self)

        self.we_dicts = we_dicts
        if feature_file is not None:
            self.ParseFeatureFile(feature_file)


    def ParseFeatureFile(self, feature_file):
        f = open(feature_file,  "r")
        for line in f.readlines():
            if line.startswith("#"):
                continue
            line=line.strip()
            tmps = line.split(" ")

            if tmps[0]== "WE":
                    attr = {}
                    for j in range(2, len(tmps)):
                        at = tmps[j].split(":")
                        if len(at)==2:
                            attr[at[0]]= at[1]
                    f = self.getFeature(FeatureName(tmps[1]),attr)
                    wf = WEFeature("WE" + str(f.feature_name),  f, self.we_dicts[tmps[2]])
                    self.append(wf)
            else:

                if tmps[0]== "WESum":
                        attr = {}
                        for j in range(2, len(tmps)):
                            at = tmps[j].split(":")
                            if len(at)==2:
                                attr[at[0]]= at[1]
                        f = self.getFeature(FeatureName(tmps[1]),attr)
                        wf = WEFeature("WE" + str(f.feature_name),  f, self.we_dicts[tmps[2]], ope="sum")
                        self.append(wf)

                else:
                        if tmps[0]== "CO":
                            f1 = self.getFeature(FeatureName(tmps[2]))
                            f2= self.getFeature(FeatureName(tmps[3]))

                            f = IndexedCoFeature("CO" + "_" + str(f1.feature_name) + str(f2.feature_name), f1, f2)
                            self.append(f)


                        else:

                            attr = {}
                            for j in range(1, len(tmps)):
                                    at = tmps[j].split(":")
                                    if len(at)==2:
                                        attr[at[0]]= at[1]
                            f = self.getFeature(FeatureName(tmps[0]), attr)
                            self.append(f)


    def getFeature(self, fn, attr = None):
        raise NotImplementedError("Subclasses should implement this!")

    def buildFeature(self, ins):
        for f in self:
            if type(f) is not WEFeature:
                f.addFeatureValueToMap(f.getFeatureValue(ins))

