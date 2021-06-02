import pandas as pd
import numpy as np
import readFasta
import re
import sys
import os
from collections import Counter


def extract_target(fastas):
    encodings = []
    headers = ['#', 'target']
    encodings.append(headers)
    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        target = 1 if name.split('@')[1] == 'H' else 0
        code.append(target)
        encodings.append(code)
    return encodings, headers


if __name__ == "__main__":
    fastas1 = readFasta.readFasta(r"data.fasta")
    result, header = extract_target(fastas1)
    data1 = np.matrix(result[1:])[:, 1:]
    data2 = np.matrix(result[1:])

    # data2 = np.matrix(result)
    data_PseAAC = pd.DataFrame(data=data1, columns=header[1:])
    data_PseAAC_name = pd.DataFrame(data=data2, columns=header)
    data_PseAAC.to_csv('target_data.csv')
    data_PseAAC_name.to_csv('target_data_name.csv')
