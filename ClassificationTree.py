from msilib.schema import Class
import numpy as np
import math
import pandas as pd
from statistics import mode


class Leaf:
    def __init__(self, outputs):
        self.output = mode(outputs)


# For continuous variables, finds all the possible "cutoff points for a split"
def find_avgs(sorted):
        avgs = []
        for i in range(len(sorted)-1):
            avgs.append((sorted[i] + sorted[i+1]) / 2)
        return avgs


# Standard shannon entropy equation
def entropy(vect):
    entropy = 0
    # Don't forget to convert output array into numpy array.
    for i in set(vect):
        prob = len(np.where(vect == i)[0]) / len(vect)
        entropy = entropy - (prob * math.log(prob, 2))
    return entropy


# Standard information gain implementation, calculate weighted entropies and subtract from total entropy
def information_gain(vect, left, right):
    total_len = len(left) + len(right)
    return entropy(vect) - (len(left)/total_len* entropy(left) + len(right)/total_len*entropy(right))


# The node class contains most of the functionality. A node receives data and will determine itself whether to continue
# branching off,or to create a leaf and end the recursion using Shannon Entropy.
class Node:
    def __init__(self, data, outputs, vars, sf=[], path=['rt']):
        self.data = data
        self.outputs = outputs
        self.vars = vars
        sf.append(path)
        self.path = sf
        information_gains = []
        # Go through each eligible variable and calculate the information gain.
        for i in self.vars:
            if type(data[0, i]) is str:
                information_gains.append(self.categorical_find_split(i))
            else:
                information_gains.append(self.continuous_find_split(i)[0])
                # The maximum information gain tells us what variable we use to make a split for this node.
        self.split = self.vars[information_gains.index(max(information_gains))]

        # Remove the current variable from this branch of the tree.
        self.vars.remove(self.split)

        # Handle a categorical variable
        if type(data[0, self.split]) is str:
            # Because this model requires categorical data to be given as a string, we use a dictionary to handle the
            # varied amount of categories a sample can be classified as. the name of the class is the key, and the
            # Node/Leaf it leads to is the value.
            self.split_dict = {}
            for i in set(data[:, self.split]):
                # We only need a new node if the entropy the split is > 1
                if entropy(outputs[np.where(data[:, self.split] == i)]) > 1 and len(vars) > 1:
                    self.split_dict[i] = Node(data[np.where(data[:, self.split] == i)], outputs[
                        np.where(data[:, self.split] == i)], self.vars, self.path, 'ct')
                else:
                    self.split_dict[i] = Leaf(outputs[np.where(data[:, self.split] == i)])
        else:
            # Handle a continuous variable
            # The point where the decision tree splits, i.e "age <= 16?"
            self.cutoff = self.continuous_find_split(self.split)[1]
            leftdata = data[np.where(data[:, self.split] <= self.cutoff)]
            leftoutputs = outputs[np.where(data[:, self.split] <= self.cutoff)]
            rightdata = data[np.where(data[:, self.split] > self.cutoff)]
            rightoutputs = outputs[np.where(data[:, self.split] > self.cutoff)]

            # We only need a new node if the entropy the split is > 1
            if entropy(leftoutputs) > 1 and len(vars) > 1:
                self.left = Node(leftdata, leftoutputs, self.vars, self.path, 'r')
            else:
                self.left = Leaf(leftoutputs)
            if entropy(rightoutputs) > 1 and len(vars) > 1:
                self.right = Node(rightdata, rightoutputs, self.vars, self.path, 'l')
            else:
                self.right = Leaf(rightoutputs)


    def continuous_find_split(self, idx):
        # Consider all possible split points and return the most effective one.
        srtd = sorted(set(self.data[:, idx]))
        avgs = find_avgs(srtd)
        inf_gains = []
        for i in avgs:
            leftoutputs = self.outputs[np.where(self.data[:, idx] <= i)]
            rightoutputs = self.outputs[np.where(self.data[:, idx] > i)]
            inf_gains.append(information_gain(self.outputs, leftoutputs, rightoutputs))
        return max(inf_gains), avgs[inf_gains.index(max(inf_gains))]

    def categorical_find_split(self, idx):
        classes = set(self.data[:, idx])
        inf_gain = entropy(self.outputs)
        for i in classes:
            # Number of datapoints with category i at idx divided by total number of datapoints
            probability = len(np.where(self.data[:, idx] == i)[0]) / len(self.outputs)
            inf_gain -= probability * entropy(self.outputs[np.where(self.data[:, idx] == i)])
        return inf_gain


# I am handling categorical data by requiring categorical data to input as STRINGS, regardless 
# of whether is is as "Tree", "Bush" etc. Convert to "T", "B" and numerical categories can be typecasted
# This also assumes that data is preprocessed.
# Output and Input arrays MUST be non-nested NumPy arrays.
class ClassificationTree:
    def __init__(self, data, outputs):
        # Calls recursive node function
        self.root = Node(data, outputs, list(range(len(data[0]))))

    def predict(self, sample):
        current = self.root
        # Traverse through tree until we reach a Leaf a.k.a an output
        while type(current) is not Leaf:
            if type(sample[current.split]) is str:
                # This handles a certain error, where a Node was trained with only a subset of categories for a
                # variable, and a sample being run through the tree had a category this Node was not trained on.
                # Clunky solution, exploring other options.
                try:
                    current = current.split_dict[sample[current.split]]
                except KeyError:
                    current = Leaf(current.outputs)
            else:
                if sample[current.split] > current.cutoff:
                    current = current.right
                else:
                    current = current.left
        return current.output

    def find_accuracy(self, testdata, testoutputs):
        correct = 0
        for i, j in zip(testdata, testoutputs):
            if self.predict(i) == j:
                correct += 1
        return correct / len(testdata)


if __name__ == '__main__':
    # Extract example dataset from GitHub repository
    Example_TO = pd.read_csv('https://raw.githubusercontent.com/desmondharris/MachineLearningPractice/main/Datasets/car_evaluation.csv', nrows = 400, usecols = [6]).to_numpy()
    Example_TO = np.array([item for sublist in Example_TO for item in sublist])
    Example_TD = pd.read_csv('https://raw.githubusercontent.com/desmondharris/MachineLearningPractice/main/Datasets/car_evaluation.csv', nrows = 400, usecols = range(6)).to_numpy()

    Example_TrO = pd.read_csv('https://raw.githubusercontent.com/desmondharris/MachineLearningPractice/main/Datasets/car_evaluation.csv', skiprows = 400, usecols = [6]).to_numpy()
    Example_TrO = np.array([item for sublist in Example_TrO for item in sublist])
    Example_TrD = pd.read_csv('https://raw.githubusercontent.com/desmondharris/MachineLearningPractice/main/Datasets/car_evaluation.csv', skiprows = 400, usecols = range(6)).to_numpy()

    Example_Tree = ClassificationTree(Example_TrD, Example_TrO)
    print(Example_Tree.find_accuracy(Example_TD, Example_TO))