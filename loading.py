import numpy as np


class DataSet:

    def __init__(self, filename):
        data = np.genfromtxt(filename, delimiter=",", dtype="unicode")
        features_str = data[0:, :-1]
        self.features = features_str.astype(np.int)
        self.labels = np.transpose(data[0:, -1])
