from liir.nlp.features.FeatureGeneratorBase import FeatureGeneratorBase
from liir.nlp.features.FeatureName import FeatureName
from liir.nlp.features.IndexedCoFeature import IndexedCoFeature
from liir.nlp.features.IndexedFeature import IndexedFeature
from liir.nlp.features.WEFeature import WEFeature
from liir.nlp.features.WordFeatureFuncs import getWordFeatureFunc, getWordPairFeatureFunc, getNeighbourFeatureFunc

__author__ = 'quynhdo'
class SRLFeatureGenerator(FeatureGeneratorBase):
    def __init__(self, feature_file, we_dicts = None):
        FeatureGeneratorBase.__init__(self, feature_file, we_dicts)

    def getFeature(self, fn, attr = None):
        if fn == FeatureName.Word:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getWord', word_data = 'getWordValue')

        if fn == FeatureName.POS:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getWord', word_data = 'getPosValue')

        if fn == FeatureName.Deprel:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getWord', word_data = 'getDeprelValue')

        if fn == FeatureName.IsCapital:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getWord', word_data = 'getCapitalValue')

        if fn == FeatureName.PredWord:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getWord', word_data = 'getWordValue', pos=1)

        if fn == FeatureName.LeftChildWord:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getLeftDep', word_data = 'getWordValue')

        if fn == FeatureName.RightChildWord:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getRightDep', word_data = 'getWordValue')


        if fn == FeatureName.LeftChildPOS:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getLeftDep', word_data = 'getPosValue')

        if fn == FeatureName.RightChildPOS:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getRightDep', word_data = 'getPosValue')

        if fn == FeatureName.InContext:
            return  IndexedFeature(fn,getWordPairFeatureFunc,is1Hot=False,  )
            return WordPairContextFeature(fn, windows=attr['w'])




        if fn == FeatureName.NeighbourWord:
            return IndexedFeature(fn, getNeighbourFeatureFunc, is1Hot=False, target_word = 'getWord', word_data = 'getWordValue', nei= int(attr['p']))

        if fn == FeatureName.NeighbourPOS:
            return IndexedFeature(fn, getNeighbourFeatureFunc, is1Hot=False, target_word = 'getWord', word_data = 'getPosValue', nei= int(attr['p']))


        if fn == FeatureName.PredNeighbourWord:
            return IndexedFeature(fn, getNeighbourFeatureFunc, is1Hot=False, target_word = 'getWord', word_data = 'getWordValue', nei= int(attr['p']), pos=1)



        if fn == FeatureName.PredNeighbourPOS:
            return IndexedFeature(fn, getNeighbourFeatureFunc, is1Hot=False, target_word = 'getWord', word_data = 'getPosValue', nei= int(attr['p']), pos=1)



        if fn == FeatureName.POSPath:
            return IndexedFeature(fn, getWordPairFeatureFunc, is1Hot=False, wpd = 'pos_path')


        if fn == FeatureName.DeprelPath:
            return IndexedFeature(fn, getWordPairFeatureFunc, is1Hot=False, wpd = 'deprel_path')

        if fn == FeatureName.ChildWordSet:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getChildren', word_data = 'getWordValue')

        if fn == FeatureName.ChildDeprelSet:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getChildren', word_data = 'getDeprelValue')

        if fn == FeatureName.ChildPOSSet:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getChildren', word_data = 'getPosValue')


        if fn == FeatureName.DepSubCat:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getWord', word_data = 'getDeprelSubCatValue')


        if fn == FeatureName.PredParentPOS:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getParent', word_data = 'getPosValue', pos=1)

        if fn == FeatureName.PredParentWord:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getParent', word_data = 'getWordValue', pos=1)


        if fn == FeatureName.PredPOS:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getWord', word_data = 'getPosValue', pos=1)


        if fn == FeatureName.PredLemma:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getWord', word_data = 'getLemmaValue', pos=1)


        if fn == FeatureName.PredDeprel:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getWord', word_data = 'getDeprelValue', pos=1)



        if fn == FeatureName.PredLemmaSense:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getWord', word_data = 'getLemmaSenseValue', pos=1)


        if fn == FeatureName.PredSense:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getWord', word_data = 'getSenseValue', pos=1)



        if fn == FeatureName.RightSiblingWord:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getRightSibling', word_data = 'getWordValue')



        if fn == FeatureName.RightSiblingPOS:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getRightSibling', word_data = 'getPosValue')



        if fn == FeatureName.RightSiblingDeprel:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getRightSibling', word_data = 'getDeprelValue')




        if fn == FeatureName.LeftSiblingWord:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getLeftSibling', word_data = 'getWordValue')




        if fn == FeatureName.LeftSiblingPOS:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getLeftSibling', word_data = 'getPosValue')


        if fn == FeatureName.LeftSiblingDeprel:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getLeftSibling', word_data = 'getDeprelValue')



        if fn == FeatureName.Position:
            return IndexedFeature(fn, getWordPairFeatureFunc, is1Hot=False, wpd = 'position')

        if fn == FeatureName.SpanWordSet:
            return IndexedFeature(fn, getWordFeatureFunc, is1Hot=False, target_word = 'getSpan', word_data = 'getWordValue')


        if fn == FeatureName.PathWordSet:
            return IndexedFeature(fn, getWordPairFeatureFunc, is1Hot=False, wpd = 'path_of_word')

        if fn == FeatureName.Interaction:
            return IndexedFeature(fn, getWordPairFeatureFunc, is1Hot=False, wpd = 'interaction')

    