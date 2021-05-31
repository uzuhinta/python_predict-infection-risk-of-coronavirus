# amino = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
#          'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
import pandas as pd
import numpy as np
import checkFasta
import readFasta
import re
import sys
import os
from collections import Counter
# pPath = os.path.split(os.path.realpath(__file__))[0]
# sys.path.append(pPath)

USAGE = """
USAGE:
	python AAP.py input.fasta order <output>
	input.fasta:      the input protein sequence file in fasta format.
	order:            the out order, select from ['alphabetically', 'polarity', 'sideChainVolume' or userDefined] 
	output:           the encoding file, default: 'encodings.tsv'
"""
amino = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
H1 = [0.62, 0.29, - 0.9, - 0.74,	1.19,	0.48, - 0.4,	1.38, - 1.5,	1.06,
      0.64, - 0.78,	0.12, - 0.85, - 2.53, - 0.18, - 0.05,	1.08,	0.81,	0.26]
H2 = [-0.5, - 1,	3,	3, - 2.5	, 0, - 0.5, - 1.8,	3, - 1.8, -
      1.3,	0.2,	0,	0.2	, 3,	0.3, - 0.4, - 1.5, - 3.4, - 2.3]
M = [15, 47, 59,	73,	91,	1,	82,	57,	73,	57,	75,
     58,	42,	72,	101	, 31,	45,	43,	130,	107]


def PseAAC(fastas, lamda=20, w=0.05, ** kw):
    AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    header = ['#']
    for aa in AA:
        header.append(aa)
    encodings.append(header)
    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        length = len(sequence)
        for aa in AA:
            fre = sequence.count(aa) / length
            code.append(fre)
        encodings.append(code)
    return encodings


kw = {'path': r"H_train.txt", 'order': 'ACDEFGHIKLMNPQRSTVWY'}

if __name__ == "__main__":
    fastas1 = readFasta.readFasta(r"data.fasta")
    result = AAC(fastas1, **kw)
    data1 = np.matrix(result[1:])[:, 1:]
    data2 = np.matrix(result)
    data_AAC = pd.DataFrame(data=data1)
    data_AAC_name = pd.DataFrame(data=data2)
    data_AAC.to_csv('AAC_data.csv')
    data_AAC_name.to_csv('AAC_data_name.csv')
