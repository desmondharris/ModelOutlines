from msilib.schema import Class
import numpy as np
import math
import pandas as pd
import seaborn as sbs
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
    # Don't forget to convert output array into numpy array.
    for i in set(vect):
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
    def __init__(self, data, outputs, vars, sf = [], path = ['rt']):
        self.data = data
        self.outputs = outputs
        self.vars = vars
        sf.append(path)
        self.path = sf
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
            h = set(data[:, self.split])
            for i in set(data[:, self.split]):
                if entropy(outputs[np.where(data[:, self.split] == i)]) > 1 and len(vars) > 1:
                    self.splitdict[i] = Node(data[np.where(data[:, self.split] == i)], outputs[np.where(data[:, self.split] == i)], self.vars, self.path, 'ct')    
                else: 
                    self.splitdict[i] = Leaf(outputs[np.where(data[:, self.split] == i)])
        else:
            # handle a continuous variable
            # the point where the decision tree splits, i.e "age <= 16?"
            self.cutoff = self.continuous_find_split(self.split)[1]
            leftdata = data[np.where(data[:, self.split] <= self.cutoff)]
            leftoutputs = outputs[np.where(data[:, self.split] <= self.cutoff)]
            rightdata = data[np.where(data[:, self.split] > self.cutoff)]
            rightoutputs = outputs[np.where(data[:, self.split] > self.cutoff)]
            

            if entropy(leftoutputs) > 1 and len(vars) > 1:
                self.left = Node(leftdata, leftoutputs, self.vars, self.path, 'r')
            else:
                self.left = Leaf(leftoutputs)
            if entropy(rightoutputs) > 1 and len(vars) > 1:
                self.right = Node(rightdata, rightoutputs, self.vars, self.path, 'l')
            else:
                self.right = Leaf(rightoutputs)
               
        
    def continuous_find_split(self, idx):
        #return list as [entropy, cutoffpoint] 
        srtd = sorted(set(self.data[:, idx]))
        avgs = find_avgs(srtd)
        disorders = []
        for cnt, i in enumerate(avgs):
            left = self.data[np.where(self.data[:, idx] <= i)]
            g = np.where(self.data[:, idx] <= i)[0][0]
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
                try:
                    current = current.splitdict[sample[current.split]]
                except:
                    current = Leaf(current.outputs)       
            else:
                if sample[current.split] > current.cutoff:
                    current = current.right
                else:
                    current = current.left
        return current.output

    def findAccuracy(self,testdata, testoutputs):
        correct = 0
        for i, j in zip(testdata, testoutputs):
            if self.predict(i) == j:
                correct += 1
        return correct / len(testdata)



"""trainingdata = pd.read_csv(r"C:/Users/dsm84762/Desktop/Python Code/knntestdata/KNNAlgorithmDataset.csv", nrows = 500, usecols= range(2,32)).to_numpy()
trainingoutput = pd.read_csv(r"C:/Users/dsm84762/Desktop/Python Code/knntestdata/KNNAlgorithmDataset.csv", nrows = 500, usecols = [1] ).to_numpy()
trainingoutput = np.array([x for xs in trainingoutput for x in xs])

testdata = pd.read_csv(r"C:/Users/dsm84762/Desktop/Python Code/knntestdata/KNNAlgorithmDataset.csv", skiprows = 500, usecols= range(2,32)).to_numpy()
testoutput = pd.read_csv(r"C:/Users/dsm84762/Desktop/Python Code/knntestdata/KNNAlgorithmDataset.csv", skiprows = 500, usecols = [1] ).to_numpy()
"""
# Extract example dataset from GitHub repository
Example_TO = pd.read_csv('https://raw.githubusercontent.com/desmondharris/MachineLearningPractice/main/Datasets/car_evaluation.csv', nrows = 400, usecols = [6]).to_numpy()
Example_TO = np.array([item for sublist in Example_TO for item in sublist])
Example_TD = pd.read_csv('https://raw.githubusercontent.com/desmondharris/MachineLearningPractice/main/Datasets/car_evaluation.csv', nrows = 400, usecols = range(6)).to_numpy()

Example_TrO = pd.read_csv('https://raw.githubusercontent.com/desmondharris/MachineLearningPractice/main/Datasets/car_evaluation.csv', skiprows = 400, usecols = [6]).to_numpy()
Example_TrO = np.array([item for sublist in Example_TrO for item in sublist])
Example_TrD = pd.read_csv('https://raw.githubusercontent.com/desmondharris/MachineLearningPractice/main/Datasets/car_evaluation.csv', skiprows = 400, usecols = range(6)).to_numpy()

Example_Tree = ClassificationTree(Example_TrD, Example_TrO)
print(Example_Tree.findAccuracy(Example_TD,Example_TO))