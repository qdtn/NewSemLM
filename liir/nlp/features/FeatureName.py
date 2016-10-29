__author__ = 'quynhdo'
from enum import Enum
class FeatureName(Enum):
    Word = 'Word'
    POS = 'POS'
    Deprel = 'Deprel'
    InContext = 'InContext'
    LeftChildWord = 'LeftChildWord'
    WELeftChildWord = 'WELeftChildWord'
    RightChildWord = 'RightChildWord'
    RightChildPOS = 'RightChildPOS'
    LeftChildPOS = 'LeftChildPOS'
    WERightChildWord = 'WERightChildWord'

    PredWord = 'PredWord'
    WEWord = 'WEWord'
    PredicateContext = 'PredicateContext'

    IsCapital = 'IsCapital'
    NeighbourWord = 'NeighbourWord'
    WENeighbourWord = 'WENeighbourWord'

    NeighbourPOS = 'NeighbourPOS'

    PredNeighbourWord= "PredNeighbourWord"

    PredNeighbourPOS = "PredNeighbourPOS"

    WEPredNeighbourWord= "WEPredNeighbourWord"

    WEPredNeighbourPOS = "WEPredNeighbourPOS"

    Position = 'Position'
    POSPath= "POSPath"
    DeprelPath = "DeprelPath"

    ChildWordSet = "ChildWordSet"
    ChildDeprelSet = "ChildDeprelSet"
    ChildPOSSet = "ChildPOSSet"

    SpanWordSet = "SpanWordSet"
    DepSubCat = 'DepSubCat'

    PredParentPOS= 'PredParentPOS'
    PredParentWord= 'PredParentWord'

    PredPOS = 'PredPOS'

    PredLemma = 'PredLemma'
    PredDeprel = 'PredDeprel'
    PredSense = 'PredSense'
    PredLemmaSense = 'PredLemmaSense'

    RightSiblingWord = 'RightSiblingWord'

    RightSiblingPOS= 'RightSiblingPOS'

    RightSiblingDeprel = 'RightSiblingDeprel'

    LeftSiblingWord = 'LeftSiblingWord'

    LeftSiblingPOS= 'LeftSiblingPOS'

    LeftSiblingDeprel = 'LeftSiblingDeprel'

    ChildWordDeprelSet = 'ChildWordDeprelSet'

    ChildPOSDeprelSet = 'ChildPOSDeprelSet'

    PathWordSet = 'PathWordSet'
    Interaction = 'Interaction'