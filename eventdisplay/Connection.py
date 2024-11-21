#!/bin/env python

from tqdm import tqdm

class Connection:

    def __init__(self, filepath):
        self.filepath = filepath
        self.data = {}
        self.f = open(self.filepath)
        for line in tqdm(self.f.readlines(), desc="Loading connection data"):
            ls = line.split()
            self.data[int(ls[0])] = [int(x) for x in ls[2:]]

    def getConnection(self, detid):
        return self.data[detid]
