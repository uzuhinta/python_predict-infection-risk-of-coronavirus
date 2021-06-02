# amino = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
#          'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
from typing import Sequence
import pandas as pd
import numpy as np
import readFasta
import re
import sys
import os
from collections import Counter
import math
# pPath = os.path.split(os.path.realpath(__file__))[0]
# sys.path.append(pPath)

USAGE = """
USAGE:
	python AAP.py input.fasta order <output>
	input.fasta:      the input protein sequence file in fasta format.
	order:            the out order, select from ['alphabetically', 'polarity', 'sideChainVolume' or userDefined]
	output:           the encoding file, default: 'encodings.tsv'
"""


def Rvalue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)


def PseAAC(fastas, lambdaValue=30, w=0.05, **kw):
    # Read data
    with open("PAAC.txt") as f:
        records = f.readlines()
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}

    for i in range(len(AA)):
        AADict[AA[i]] = i

    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records)):
        array = records[i].rstrip().split(
        ) if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j-meanI)**2 for j in i])/20)
        AAProperty1.append([(j-meanI)/fenmu for j in i])

    #
    encodings = []
    header = ['#']
    for aa in AA:
        header.append('Xc1.' + aa)
    for n in range(1, lambdaValue + 1):
        header.append('Xc2.lambda' + str(n))
    encodings.append(header)
    encodings = []

    header = ['#']
    for aa in AA:
        header.append('Xc1.' + aa)
    for n in range(1, lambdaValue + 1):
        header.append('Xc2.lambda' + str(n))
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        theta = []
        for n in range(1, lambdaValue + 1):
            theta.append(
                sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
                    len(sequence) - n))
            myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)
        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
        encodings.append(code)
    return encodings, header


kw = {'path': r"H_train.txt", 'order': 'ACDEFGHIKLMNPQRSTVWY'}


if __name__ == "__main__":
    fastas1 = readFasta.readFasta(r"data.fasta")
    result, header = PseAAC(fastas1, **kw)
    data1 = np.matrix(result[1:])[:, 1:]
    data2 = np.matrix(result[1:])

    # data2 = np.matrix(result)
    data_PseAAC = pd.DataFrame(data=data1, columns=header[1:])
    data_PseAAC_name = pd.DataFrame(data=data2, columns=header)
    data_PseAAC.to_csv('PseAAC_data.csv')
    data_PseAAC_name.to_csv('PseAAC_data_name.csv')
