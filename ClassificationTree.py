import numpy as np
import math
import pandas as pd
from statistics import mode
class Leaf: 
    def __init__(self, outputs):
        self.output = mode(outputs)

#for continuous variables, finds all the possible "cutoff points for a split"
def find_avgs(sorted):
        avgs = []
        for i in range(len(sorted)-1):
            avgs.append((sorted[i] + sorted[i+1]) / 2)
        return avgs

#standard shannon entropy equation
def entropy(vect):
    entropy = 0
    for i in set(vect.tolist()):
        prob = len(np.where(vect == i)[0]) / len(vect)
        entropy = entropy - (prob * math.log(prob, 2))
    return entropy

#standard information gain implementation
def information_gain(vect, left, right):
    total_len = len(left) + len(right)
    return entropy(vect) - (len(left)/total_len* entropy(left) + len(right)/total_len*entropy(right))

#will take different args for character vector
def categorical_information_gain(vect, classes):
    pass

class Node:
    def __init__(self, data, outputs, vars):
        self.data = data
        self.outputs = outputs
        self.vars = vars
        information_gains = []
        #go through each eligible variable and calculate the entropy.
        for i in self.vars:
            if type(data[0, i]) is str:
                information_gains.append(self.categorical_find_split(i))
            else:
                information_gains.append(self.continuous_find_split(i)[0])
        self.split = self.vars[information_gains.index(max(information_gains))]
        #remove the current variable from this branch of the tree.
        self.vars.remove(self.split)

        #handle a categorical variable
        if type(data[0, self.split]) is str:
            # because this model requires categorical data to be given as a string, we use a dictionary to handle the varied amount of categories
            # a sample can be classified as. the name of the class is the key, and the Node/Leaf it leads to is the value.
            self.splitdict = {}
            for i in set(data[:, self.split]):
                if len(data[np.where(data[:, self.split] == i)]) > 25:
                    self.splitdict[i] = Node(data[np.where(data[:, self.split] == i)], outputs[np.where(data[:, self.split] == i)], self.vars)    
                else: 
                    self.splitdict[i] = outputs[np.where(data[:, self.split] == i)]
        else:
            # handle a continuous variable
            # the point where the decision tree splits, i.e "age <= 16?"
            self.cutoff = self.continuous_find_split(self.split)[1]
            leftdata = data[np.where(data[:, self.split] <= self.cutoff)]
            leftoutputs = outputs[np.where(data[:, self.split] <= self.cutoff)]
            rightdata = data[np.where(data[:, self.split] > self.cutoff)]
            rightoutputs = outputs[np.where(data[:, self.split] > self.cutoff)]
            

            if len(leftdata) > 50:
                self.left = Node(leftdata, leftoutputs, self.vars)
            else:
                self.left = Leaf(leftoutputs)
            if len(rightdata) > 50:
                self.right = Node(rightdata, rightoutputs, self.vars)
            else:
                self.right = Leaf(rightoutputs)
               
        
    def continuous_find_split(self, idx):
        #return list as [entropy, cutoffpoint] 
        srtd = sorted(set(self.data[:, idx]))
        avgs = find_avgs(srtd)
        disorders = []
        for cnt, i in enumerate(avgs):
            left = self.data[np.where(self.data[:, idx] <= i)]
            leftoutputs = self.outputs[np.where(self.data[:, idx] <= i)]
            right = self.data[np.where(self.data[:, idx] > i)]
            rightoutputs = self.outputs[np.where(self.data[:, idx] > i)]
            disorders.append(information_gain(self.outputs, leftoutputs, rightoutputs))
        return max(disorders), avgs[disorders.index(max(disorders))]

    def categorical_find_split(self, idx):
        classes = set(self.data[:, idx])
        inf_gain = entropy(self.outputs)
        for i in classes:
            # number of datapoints with category i at idx divided by total number of datapoints
            probability = len(np.where(self.data[:, idx] == i)[0]) / len(self.outputs)
            inf_gain -= probability * entropy(self.outputs[np.where(self.data[:, idx] == i)])
        return inf_gain
    
# I am handling categorical data by requiring categorical data to input as CHARACTERS, regardless 
# of whether is is as "Tree", "Bush" etc. Convert to "T", "B" and numerical categories can be typecasted
# This also assumes that data is preprocessed.
class ClassificationTree:
    def __init__(self, data, outputs):
        self.root = Node(data, outputs, list(range(len(data[0]))))
    
    def predict(self, sample):
        current = self.root
        while type(current) is not Leaf:
            if type(sample[current.split]) is str:
                current = current.splitdict[sample[current.split]]       
            else:
                if sample[current.split] > current.cutoff:
                    current = current.right
                else:
                    current = current.left
        return current.output


""" testdata = np.array([[1,3,4,'C'],
                      [4,6,1,'C'],
                      [1,6,9, 'A'],
                      [5,3,1, 'A']], dtype= 'object')
testoutputs = np.array([1,1,0,0]) """
data = pd.read_excel("C:/Users/dsm84762/Desktop/Data.xlsx",usecols="A:AH", nrows= 633).to_numpy()
outputs_b = pd.read_excel("C:/Users/dsm84762/Desktop/Data.xlsx",usecols="AS:AS", nrows = 633).to_numpy()
outputs = np.array([x for xs in outputs_b for x in xs])
test = ClassificationTree(data, outputs)
g = np.array([1] * 34)
print(test.predict(g))
