from liir.nlp.representation.Text import Text

__author__ = 'quynhdo'

class Reader(object):
    def __init__(self, input_file):
        self.input_file=input_file

    def readAll(self):
        raise NotImplementedError("Subclasses should implement this!")

class BatchReader(Reader):
    def __init__(self, batch_size, input_file):
        Reader.__init__(self,input_file)
        self.batch_size = batch_size
        self.current_position = 0
        self.current_file = 0



    def readAll(self):
        raise NotImplementedError("Subclasses should implement this!")

    def next(self):
        raise NotImplementedError("Subclasses should implement this!")

    def reset(self):
        self.current_position = 0
        self.current_file = 0


class SimulateReader(BatchReader):
    def __init__(self, batch_size, data, input_file=None):
        BatchReader.__init__(self,batch_size,input_file)

        self.data = data

    def readAll(self):
        return self.data




class Conll2009BatchReader(BatchReader):

    def __init__(self, batch_size, input_file, read_label=True, use_gold=False):
        BatchReader.__init__(self,batch_size,input_file)

        self.read_label = read_label
        self.use_gold = use_gold

    def next(self):
        txt = Text()
        if self.current_file >= len(self.input_file):
            return txt

        txt.readConll2009SentencesRange(self.input_file[self.current_file], self.current_position, self.current_position + self.batch_size,
                                        self.read_label, self.use_gold)
        self.current_position += len(txt)

        while len(txt) < self.batch_size:
            self.current_position = 0
            self.current_file +=1
            if self.current_file >= len(self.input_file):
                return txt
            s1 = len(txt)
            txt.readConll2009SentencesRange(self.input_file[self.current_file], self.current_position, self.current_position + self.batch_size - len(txt),
                                        self.read_label, self.use_gold)

            self.current_position += len(txt) - s1




        return txt

    def readAll(self):
        txt = Text()
        for f in self.input_file:
            txt.readConll2009Sentences(f)
        return txt

if __name__== "__main__":
    n = "/Users/quynhdo/Desktop/ood.conll2009.pp.txt"
    txt = Text()
    txt.readConll2009Sentences(n)
    txt.readConll2009Sentences("/Users/quynhdo/Desktop/eval.conll2009.pp.txt")
    print (len(txt))
    r = Conll2009BatchReader(20,[n,"/Users/quynhdo/Desktop/eval.conll2009.pp.txt" ])
    num = 0

    txt = r.next()
    while txt is not None:

        num += len(txt)
        txt = r.next()

    print (num)
