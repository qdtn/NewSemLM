

class Word(object):

    def __init__(self,idx=None, form=None, useDependency=True):
            self.id = idx
            self.form = form
            self.lemma = None
            self.pos = None
            self.useDependency = useDependency # use dependency style or not
            if useDependency:
                self.head = None
                self.deprel = None
            else:
                self.parsebit = None
            self.sentence=None


    def makePredicate(self):
        '''
        convert a word to a predicate
        '''

        self.__class__ = Predicate
        self.sense = None
        if self.useDependency:
            self.arguments = {}
        else:
            self.arguments = []


    def getWord(self):
        return self

    def getParent(self):
        if self.head >= 0:
            return self.sentence[self.head]
        return None

    def getChildren(self):
        children = []
        for w in self.sentence:
            if w.head == self.id:
                children.append(w)
        return children

    def getLeftDep(self):
        for w in self.sentence:
            if w.head == self.id:
                return w
        return None

    def getRightDep(self):
        for i in range(len(self.sentence)-1,-1,-1):
            if self.sentence[i].head == self.id:
                return self.sentence[i]
        return None

    def getRightSibling(self):
        for i in range(self.id+1, len(self.sentence)):
            if self.sentence[i].head  ==  self.head:
                return self.sentence[i]

    def getLeftSibling(self):
        for i in range(self.id -1, -1):
            if self.sentence[i].head  ==  self.head:
                return self.sentence[i]

    def getSpan(self):
        s = []
        for w in self.sentence:
            if w == self or w in self.children:
                s.append(w)
        return s


    def getWordValue(self):
        return self.form.lower()

    def getPosValue(self):
        return self.pos

    def getDeprelValue(self):
        return self.deprel

    def getLemmaValue(self):
        return self.lemma

    def getCapitalValue(self):
        return self.form[0].isupper()

    def getDeprelSubCatValue(self):
        if len(self.children) == 0:
            return " "
        else:
            return " ".join([wc.deprel for wc in self.children])

    def getPathToRoot(self): #get dependency path from w to the root
        lst=[]
        h=self
        while h.head != -1:
            lst.append(h)
            h=self.sentence[h.head]

        lst.append(h)
        return lst

    def findRoot(self):
        lst = []
        for w in self.sentence:
            if w.head == -1:
                lst.append(w)
        return lst

    def getPath(self, w):
        path2=w.getPathToRoot()
        path1=self.getPathToRoot()

        common1=-1
        common2=-1
        for i in range(len(path1)-1,-1,-1):
                    for j in range (len(path2)-1,-1,-1):
                        if path1[i] == path2[j]:
                            common1=i
                            common2=j
                            break

        lst=[]

        for i in range (0, common2,1):
                    lst.append(path2[i])



        for i in range (common1, -1,-1):
                    lst.append(path1[i])

        if len(lst)>0:

                up=True


                s=[]
                for i in range(len(lst)):
                        node=lst[i]
                        if node != self:
                            s.append(node)

                        if i < len(lst)-1:
                            if up:
                                if node.sentence[node.head] == lst[i+1]:
                                    s.append("0")
                                else:
                                    s.append("1")
                                    up=False
                            else:
                                s.append("1")
                return s
        return [' ']


    def getWordPath(self, w):
        '''
        get all the words in the path from w to the current word
        '''
        path2=w.getPathToRoot()
        path1=self.getPathToRoot()

        common1=-1
        common2=-1
        for i in range(len(path1)-1,-1,-1):
                    for j in range (len(path2)-1,-1,-1):
                        if path1[i] == path2[j]:
                            common1=i
                            common2=j
                            break

        lst=[]

        for i in range (0, common2,1):
                    lst.append(path2[i])

        for i in range (common1, -1,-1):
                    lst.append(path1[i])
        return lst


class Predicate(Word):
    def __init__(self, idx=None, form=None, useDependency=True):
        Word.__init__(self,idx, form, useDependency)
        self.sense = None
        if self.useDependency:
            self.arguments = {}
        else:
            self.arguments = []

    def clear(self):
        self.sense = None
        if self.useDependency:
            self.arguments = {}
        else:
            self.arguments = []

    def getLemmaSenseValue(self):
        return self.lemma + "." + self.sense

    def getSenseValue(self):
        return self.sense




