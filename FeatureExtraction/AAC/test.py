from readFasta import readFasta
from checkFasta import checkFasta

if __name__ == "__main__":
    fastas = readFasta("dev.fasta")
    if checkFasta(fastas) == True:
        print("good chop")
