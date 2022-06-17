import numpy as np


class Leaf:
    def __init__(self, outputs):
        self.output = np.mean(outputs)


class Node:
    def __init__(self, data, outputs, vars):
        residuals = []
        for i in vars:
            if type(i) is str:
                residuals.append(self.categorical_find_split(i))
            else:
                residuals.append(self.continuous_find_split(i))
        self.split = vars[residuals.index(min(residuals))]
        vars.remove(self.split)
        if type(data[0, self.split]) is str:
        else:

    def continuous_find_split(self, var):
        pass

    def categorical_find_split(self, var):
        pass

class RegressionTree:
    def __init__(self):
        self.root = None
