# amino = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
#          'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
import pandas as pd
import numpy as np
import readFasta
import re
import sys
import os
from collections import Counter
from propy import PyPro
from propy.GetProteinFromUniprot import GetProteinSequence
# pPath = os.path.split(os.path.realpath(__file__))[0]
# sys.path.append(pPath)

USAGE = """
USAGE:
	python AAP.py input.fasta order <output>
	input.fasta:      the input protein sequence file in fasta format.
	order:            the out order, select from ['alphabetically', 'polarity', 'sideChainVolume' or userDefined] 
	output:           the encoding file, default: 'encodings.tsv'
"""


def PseAAC(fastas, feed_lambda=10, feed_weight=0.05, **kw):
    encodings = []
    #
    header = ['#']
    seq = fastas[0][1]
    DesObject = PyPro.GetProDes(seq)
    paac = DesObject.GetPAAC(lamda=feed_lambda, weight=feed_weight)

    for name in paac:
        header.append(name)
    encodings.append(header)
    print(header)
    #
    count = 1
    for i in fastas:
        del DesObject
        name, sequence = i[0], i[1]
        code = [name]
        DesObject = PyPro.GetProDes(sequence)
        paac = DesObject.GetPAAC(lamda=feed_lambda, weight=feed_weight)
        for i in paac:
            code.append(paac[i])
        encodings.append(code)
        print(count)
        count += 1
    return encodings, header


kw = {'path': r"H_train.txt", 'order': 'ACDEFGHIKLMNPQRSTVWY'}


if __name__ == "__main__":
    fastas1 = readFasta.readFasta(r"dev.fasta")
    result, header = PseAAC(fastas1, **kw)
    data1 = np.matrix(result[1:])[:, 1:]
    data2 = np.matrix(result[1:])

    # data2 = np.matrix(result)
    data_PseAAC = pd.DataFrame(data=data1, columns=header[1:])
    data_PseAAC_name = pd.DataFrame(data=data2, columns=header)
    data_PseAAC.to_csv('PseAAC_data.csv')
    data_PseAAC_name.to_csv('PseAAC_data_name.csv')
