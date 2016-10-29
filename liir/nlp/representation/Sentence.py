from liir.nlp.representation.Word import Predicate
from liir.nlp.representation.Word import Word

__author__ = 'quynhdo'

import re


# this class define a sentence which is a list of Word
class Sentence(list):
    def __init__(self):   # value should be  a list of Word
        list.__init__(self)

    def getPredicates(self):
        return [w for w in self if isinstance(w, Predicate)]

    def printfConll2009(self):
        s=""
        lst=[]
        for w in self:
            s_w=str(w.id +1) + "\t" + w.form +"\t" +w.lemma +"\t"+ w.lemma+"\t"+w.pos+ "\t" +w.pos + "\t" +"_"+ "\t" +"_"+ "\t" +str(w.head+1) + "\t" +str(w.head+1)+"\t" + w.deprel + "\t" +w.deprel + "\t"
            if isinstance(w, Predicate):
                o = "Y"
                s_w=s_w + o + "\t" + w.lemma + "." +str(w.sense)

            else:
                s_w=s_w + "_" + "\t" + "_"

            lst.append(s_w)

        preds= self.getPredicates()

        for pred in preds:
            poss=[]

            for arg in pred.arguments.keys():
                position=arg.id
                lst[position]=lst[position] +"\t" + pred.arguments[arg]
                poss.append(position)
            for i in range(len(lst)):
                if not i in set(poss):
                    lst[i]=lst[i] + "\t" + "_"

        for sss in lst:
            s=s+sss +  '\n'


        return s



# Sentence in Conll 2009
class SentenceConll2009(Sentence):

    def __init__(self, data_lines=None, read_label=True, use_gold=False):
        '''

        :param data_lines:
        :param read_label: read srl labels or not
        :param use_gold: use gold annotation or not
        :return:
        '''
        Sentence.__init__(self)
        if data_lines is None:
            return
        if not isinstance(data_lines, list):
            return
        pred_id = 0
        dt = []
        for line in data_lines:
            temps=re.split("\\s+", line)
            dt.append(temps)
            w = Word(int(temps[0])-1, temps[1])
            #w.word = temps[1].lower()
            if not use_gold:
                w.lemma = temps[3]
                w.pos = temps[5]
                w.head = int(temps[9])-1
                w.deprel = temps[11]
            else:
                w.lemma = temps[2]
                w.pos = temps[4]
                w.head = int(temps[8])-1
                w.deprel = temps[10]

            if read_label:
                if "Y" in set(temps[12]):
                    w.__class__ = Predicate
                    w.sense = temps[13].split(".")[1]

            self.append(w)
        if read_label:
            # read srl information
            for pred in self:
                if isinstance(pred, Predicate):
                    args={}
                    for j in range(len(data_lines)):
                        tmps = dt[j]
                        lbl = tmps[14+pred_id]
                        if lbl != "_":
                            args[self[int(tmps[0])-1]]=lbl

                    pred.arguments = args
                    pred_id += 1

        for w in self:
            w.sentence = self


class SentenceConll2009POS(Sentence):
    def __init__(self, data_lines=None):
        Sentence.__init__(self)
        if data_lines is None:
            return
        if not isinstance(data_lines, list):
            return
        dt = []
        for line in data_lines:
            temps=re.split("\\s+", line)
            dt.append(temps)
            w = Word(int(temps[0])-1, temps[1])
            w.word = temps[1].lower()
            w.pos = temps[4]
            self.append(w)
        for w in self:
            w.sentence = self




class SentenceConll2005(Sentence):  # Sentence in Conll 2005

    def __init__(self, data_lines=None):
        Sentence.__init__(self)
        if data_lines is None:
            return
        if not isinstance(data_lines, list):
            return
        pred_id = 0
        idx=0
        dt = []
        for line in data_lines:
            temps=re.split("\\s+", line)
            # print (temps)
            dt.append(temps)
            w = Word(idx, temps[0], False, True)
            pos = temps[1]

            if pos == "(":
                pos= "-lrb-"
            if pos == ")":
                pos = "-rrb-"
            w.pos = pos
            w.parsebit = temps[2]
            w.word = temps[0].lower()

            if temps[5] != "-":
                w.__class__= Predicate
                w.sense = temps[5]+"."+temps[4]
            self.append(w)
            idx += 1

        # read srl information
        for pred in self:
            if isinstance(pred, Predicate):
                args=[]
                j = 0
                while j < len(data_lines):
                    tmps = dt[j]
                    lbl = tmps[6+pred_id]
                    match = re.match('\((.+)\*\)', lbl)
                    if match:
                        args.append("B-"+match.group(1))
                        j += 1
                    else:
                        match = re.match('\((.+)\*', lbl)
                        if match:
                            args.append("B-"+match.group(1))
                            for k in range(j+1, len(data_lines)):
                                l1 = data_lines[k]
                                tmps1 = re.split("\\s+",l1)
                                match1 = re.match('\*\)', tmps1[6+pred_id])
                                args.append("I-"+match.group(1))
                                if match1:
                                    j = k+1
                                    break
                        else:
                            args.append("O")
                            j += 1

                pred.arguments = args
                pred_id += 1
        for w in self:
            w.sentence = self


