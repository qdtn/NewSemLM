import re

from liir.nlp.representation.Sentence import SentenceConll2005, SentenceConll2009, SentenceConll2009POS


__author__ = 'quynhdo'

# this class is used to define  a text
class Text(list):
    def __init__(self):   # value should be  a list of Sentence
        list.__init__(self)

    def readConll2005Sentences(self, path):
        f = open(path, 'r')
        sens=[]
        words=[]

        for l in f:
            match = re.match("\\s+", l)
            if match:
                if len(words) != 0:
                    sens.append(words)
                    words = []
            else:
                words.append(l.strip())
        if len(words) != 0:
            sens.append(words)

        for sen in sens:
            conll2005sen = SentenceConll2005(sen)
            self.append(conll2005sen)

    def readConll2009Sentences(self, path, read_label=True, use_gold=False):
        f = open(path, 'r')
        sens=[]
        words=[]

        for l in f:
            match = re.match("\\s+", l)
            if match:
                if len(words) != 0:
                    sens.append(words)
                    words = []
            else:
                words.append(l.strip())
        if len(words) != 0:
            sens.append(words)

        for sen in sens:
            conll2009sen = SentenceConll2009(sen, read_label, use_gold=use_gold)
            self.append(conll2009sen)

    def readConll2009SentencesRange(self, path, start, end=None, read_label=True, use_gold=False):
        f = open(path, 'r')
        sens=[]
        words=[]
        idx = 0
        for l in f:
            match = re.match("\\s+", l)
            if match:
                if len(words) != 0:
                    if end is None:
                        if idx >= start:
                            sens.append(words)

                    else:

                        if idx >= start:
                            if idx < end:
                                sens.append(words)
                            else:
                                break
                    idx +=1
                    words = []
            else:
                words.append(l.strip())
        if len(words) != 0:
            if end is None:
                if idx >= start:
                    sens.append(words)

            else:

                if idx>= start:
                    if idx < end:
                        sens.append(words)


        for sen in sens:
            conll2009sen = SentenceConll2009(sen, read_label, use_gold=use_gold)
            self.append(conll2009sen)

    def readConll2009SentencesPOS(self, path):
        f = open(path, 'r')
        sens=[]
        words=[]

        for l in f:
            match = re.match("\\s+", l)
            if match:
                if len(words) != 0:
                    sens.append(words)
                    words = []
            else:
                words.append(l.strip())
        if len(words) != 0:
            sens.append(words)

        for sen in sens:
            possen = SentenceConll2009POS(sen)
            self.append(possen)




    def getVob(self, type="all"):
        vob=set()
        if type == "all":
            for sen in self:
                for w in sen:
                    vob.add(w.word)

        if type == "pred":
            for sen in self:
                for p in sen.getPredicates():
                    vob.add(p.word)

        return vob


    def getFrequentVob(self, type="all"):
        vob={}
        if type == "all":
            for sen in self:
                for w in sen:
                    if not w.word in vob.keys():
                        vob[w.word]=1
                    else:
                        vob[w.word]=vob[w.word]+1


        if type == "pred":
            for sen in self:
                for w in sen.getPredicates():
                    if not w.word in vob.keys():
                        vob[w.word]=1
                    else:
                        vob[w.word]+=1

        return vob

    def toConll2009Format(self, output):
        s = ""
        l = "\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\n"
        for sen in self:
            for w in sen:
                s+= str(sen.index(w)+1) + "\t"+ w.form + l
            s += "\n"
        f= open(output, "w")
        f.write(s)
        f.close()

    def toConll2009POSFormat(self, output):
        s = ""
        l = "\t_\t_\t_\t_\t_\t_\t_\t_\t_\n"
        for sen in self:
            for w in sen:
                s+= str(sen.index(w)+1) + "\t"+ w.form + "\t_\t_\t" + w.pos + l
            s += "\n"
        f= open(output, "w")
        f.write(s)
        f.close()

    def writeConll2009(self, file_name):
        f=open(file_name ,"w")
        for s in self:
            f.write(s.printfConll2009())
            f.write("\n")

        f.close()


    def split(self, percent, out1=None, out2=None):
        num = len(self)
        part_num = num * percent // 100
        lst = []
        txt1 = Text()
        txt2= Text()
        import random
        while len(lst) < part_num:
            n = random.randint(0,num+1)
            if n not in lst:
                lst.append(n)

        for i in range(num):
            if i in lst:
                txt1.append(self[i])
            else:
                txt2.append(self[i])

        if out1 is not None:
            txt1.write(out1)
        if out2 is not None:
            txt2.write(out2)
        return txt1, txt2


if __name__=="__main__":
    txt1 = Text()
    txt1.readConll2009Sentences("/Users/quynhdo/Desktop/train.conll2009.pp.txt")
    txt1.readConll2009Sentences("/Users/quynhdo/Desktop/dev.conll2009.pp.txt")
    txt1.readConll2009Sentences("/Users/quynhdo/Desktop/eval.conll2009.pp.txt")
    txt1.readConll2009Sentences("/Users/quynhdo/Desktop/wsj2.pi.out")
    txt2 = Text()
    txt2.readConll2009Sentences("/Users/quynhdo/Desktop/ood.conll2009.pp.txt")
    txt2.readConll2009Sentences("/Users/quynhdo/Desktop/data/brown23.pi.out.txt")
    vob1 = txt1.getVob()
    vob2 = txt2.getVob()

    f = open("/Users/quynhdo/Desktop/oodonly.txt","w")

    for w in vob2:
        if not w in vob1:
            f.write(w + " ")
    f.close()





    #txt.readConll2009Sentences("/Users/quynhdo/Desktop/brown3.M4sg300.0_3.txt")


    #print ("hello")
    #txt = Text()
    #txt.readConll2009Sentences("/Users/quynhdo/Documents/WORKING/MYWORK/EACL/CoNLL2009-ST-English2/CoNLL2009-ST-English-evaluation-ood.txt")
    '''
    p1 = txt.getFrequentVob(type="all")

    txt=Text()
    txt.readConll2009Sentences("/Users/quynhdo/Documents/WORKING/MYWORK/EACL/CoNLL2009-ST-English2/CoNLL2009-ST-English-evaluation-ood.txt")
    p2 = txt.getFrequentVob(type="all")

    for k in p1.keys():
        if p1[k]>=20  and k in p2.keys():
            if p2[k] >=5:
                print(k)
    '''
    #txt.toConll2009Format("/Users/quynhdo/Desktop/conll2009.ood.words.txt")

    #txt = Text()
    #txt.readConll2009Sentences("/Users/quynhdo/Desktop/train.conll2009.pp.txt")
    #txt.split(50, "/Users/quynhdo/Desktop/train.conll2009.pp.50percent.txt", "/Users/quynhdo/Desktop/train.conll2009.pp.50percentRest.txt")


    '''
    txt = Text()
    txt.readConll2009Sentences("/Users/quynhdo/Desktop/train.conll2009.pp.txt")

    p1 = txt.getVob(type="all")

    txt=Text()
    txt.readConll2009Sentences("/Users/quynhdo/Desktop/ood.conll2009.pp.txt")

    #txt.readConll2009Sentences("/Users/quynhdo/Desktop/brown2.M4sg50.0_3.txt")

    #txt.readConll2009Sentences("/Users/quynhdo/Desktop/brown3.M4sg300.0_3.txt")
    p2 = txt.getVob(type="all")

    print (len(p1))
    print (len(p2))
    print (len(p1.intersection(p2)))

    f = open ("/Users/quynhdo/Desktop/domain.txt", "w")

    for w in p2:
        f.write(w)
        f.write(" ")



        for wn in p1:
            f.write(wn)
            f.write(" ")
        f.write("\n")
    f.close()
    '''

    '''
    f.write("source: ")
    for w in p1:
        f.write(w)
        f.write(" ")


    f.write("\ntarget: ")
    for w in p2:
        f.write(w)
        f.write(" ")
    f.close()

    '''