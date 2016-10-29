__author__ = 'quynhdo'

import numpy as np
import re

class WEDict:

    def __init__(self, full_dict_path=None):
        f = open(full_dict_path, "r")
        self.full_dict = {}
        self.we_size = -1
        for l in f.readlines(): # read the full dictionary
            l = l.strip()
            tmps = re.split('\s+', l)
            if len(tmps) > 1:
                we = []
                if self.we_size == -1:
                    self.we_size = len(tmps)-1
                for i in range(1, len(tmps)):
                    we.append(float(tmps[i].strip()))

                self.full_dict[tmps[0]]= np.asarray(we)

        f.close()
        self.scale()

    def mergeWEDict(self, d):
        '''
        merge the current dict with another wedict
        :param d:
        :return:
        '''
        for k in self.full_dict.keys():
            arr1 = self.full_dict[k]
            arr2 = d.getWE(k)
            arr= np.concatenate((arr1,arr2))
            self.full_dict[k]=arr

        self.we_size += d.we_size

    def getFullVobWE(self):
        return np.asarray([v for v in self.full_dict.values()])

    def getFullVobWEAndKeys(self):
        k = []
        t = []
        for item in self.full_dict.items():
            k.append(item[0])
            t.append(item[1])

        return np.asarray(k), np.asarray(t)

    def getWE(self, w):
        we = None
        if w in self.full_dict.keys():
            we = self.full_dict[w]
        else:
            we = np.zeros(self.we_size)
        return we

    def extractWEForVob(self, vob, output):
        f = open(output, "w")
        c = 0
        for w in vob:
            if w in self.full_dict.keys():
                f.write(w)
                f.write(" ")
                we = self.full_dict[w]
                c += 1
                for val in we:
                    f.write(str(val))
                    f.write(" ")
                f.write("\n")
        f.close()
        print ( "Words in WE dict: ")
        print (str(c) + "/" + str(len(vob)))

    def writeToFile(self, output):
        f = open(output, "w")
        for w in self.full_dict.keys():
            f.write(w)
            f.write(" ")
            for v in self.full_dict[w]:
                f.write(str(v))
                f.write(" ")
            f.write("\n")
        f.close()

    def scale(self):
        k,t = self.getFullVobWEAndKeys()

        t = 0.1 * t / np.std(t)
        for kk in range(k.size):
            self.full_dict[k[kk]] = t[kk,:]



if __name__ == "__main__":
    d = WEDict("/Users/quynhdo/Desktop/word2vec_with_genia_binary.txt")

    d1= WEDict("/Users/quynhdo/Desktop/word2vec_with_genia_sparse_DA0_3_binary.txt")

    d.mergeWEDict(d1)

    d.writeToFile("/Users/quynhdo/Desktop/word2vec_with_genia_sparse_DA0_3.concatenated_binary.txt")
