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


def AAC(fastas, gap, **kw):
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
    return encodings, header


kw = {'path': r"H_train.txt", 'order': 'ACDEFGHIKLMNPQRSTVWY'}

if __name__ == "__main__":
    fastas1 = readFasta.readFasta(r"data.fasta")
    result, header = AAC(fastas1, 1, **kw)
    data1 = np.matrix(result[1:])[:, 1:]
    data2 = np.matrix(result[1:])

    # data2 = np.matrix(result)
    data_AAC = pd.DataFrame(data=data1, columns=header[1:])
    data_AAC_name = pd.DataFrame(data=data2, columns=header)
    data_AAC.to_csv('AAC_data.csv')
    data_AAC_name.to_csv('AAC_data_name.csv')
