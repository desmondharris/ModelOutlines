import numpy as np
from ClassificationTree import find_avgs


class Leaf:
    def __init__(self, outputs):
        self.output = np.mean(outputs)


def sum_squared_residuals(expected, predicted):
    sum_residuals = 0
    for exp in expected:
        sum_residuals += (exp - predicted)**2
    return sum_residuals


class Node:
    def __init__(self, data, outputs, vars):
        self.data = data
        self.outputs = outputs
        residuals = []
        for i in vars:
            if type(self.data[:, i]) is str:
                pass
            else:
                residuals.append(self.continuous_find_split(i)[0])
        self.split = vars[residuals.index(min(residuals))]
        vars.remove(self.split)
        if type(self.data[0, self.split]) is str:
            self.splitdict = {}
            for i in set(self.data[:, self.split]):
                if len(np.where(self.data[:, self.split] == i)) > 20 and len(vars) > 1:
                    self.splitdict[i] = Node(self.data, self.outputs, vars)
                else:
                    self.splitdict[i] = Leaf(self.outputs)
        else:
            self.cutoff = self.continuous_find_split(self.split)[1]
            left_outputs = self.outputs[np.where(self.data[:, self.split] <= self.cutoff)]
            right_outputs = self.outputs[np.where(self.data[:, self.split] > self.cutoff)]
            if len(left_outputs) > 20:
                left_data = self.data[np.where(self.data[:, self.split] <= self.cutoff)]
                self.left = Node(left_data, left_outputs, vars)
            else:
                self.left = Leaf(left_outputs)
            if len(right_outputs) > 20:
                right_data = self.data[np.where(self.data[:, self.split] > self.cutoff)]
                self.right = Node(right_data, right_outputs, vars)
            else:
                self.right = Leaf(right_outputs)

    def continuous_find_split(self, var):
        avgs = find_avgs(sorted(set(self.data[:, var])))
        residual_candidates = []
        for i in avgs:
            left_outputs = self.outputs[np.where(self.data[:, var] <= i)]
            left_p_output = np.mean(left_outputs)
            right_outputs = self.outputs[np.where(self.data[:, var] > i)]
            right_p_output = np.mean(right_outputs)
            residual_candidates.append(sum_squared_residuals(left_outputs, left_p_output) +
                                       sum_squared_residuals(right_outputs, right_p_output))
        return [min(residual_candidates), avgs[residual_candidates.index(min(residual_candidates))]]

    def categorical_find_split(self, var):
        classes = set(self.data[:, var])
        total = 0
        for i in classes:
            current = self.outputs[np.where(self.data[:, var] == i)]
            current_output = np.mean(current)
            total += sum_squared_residuals(current, current_output)
        return total


class RegressionTree:
    def __init__(self, data, outputs):
        self.root = Node(data, outputs, list(range(len(data[0]))))

    def predict(self, sample):
        current = self.root
        while type(current) is not Leaf:
            if type(sample[current.split]) is str:
                current = current.split_dict[sample[current.split]]
            else:
                if sample[current.split] > current.cutoff:
                    current = current.right
                else:
                    current = current.left
        return current.output
