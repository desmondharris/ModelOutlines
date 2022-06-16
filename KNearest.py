import numpy as np
import pandas as pd
from statistics import mode
import math


def knn(n, data, labels, sample):
    distances = []
    # Find sample distance from all training data
    for i in data:
        distances.append(euclidian_distance(i, sample))
    distances = np.array(distances)
    # Find n nearest samples, the mode of these samples' classes will be our outputg
    return mode(labels[np.argpartition(distances, n)[:n]])


# Standard Euclidian distance implementation 
def euclidian_distance(vec1, vec2):
    dist = 0
    for i in range(len(vec1)):
        dist += math.sqrt((vec2[i] - vec1[i])**2)
    return dist


# Calculate algorithm accuracy for given n and test dataset
def knn_accuracy(n, data, outputs, samples, sampleoutputs):
    correct = 0
    for input, output in zip(samples, sampleoutputs):
        if knn(n, data, outputs, input) == output:
            correct += 1
    return correct / len(sampleoutputs)


# Using UCI Wisconsin Breast Cancer Dataset 
# https://www.kaggle.com/datasets/zzero0/uci-breast-cancer-wisconsin-original
# Import training data

# Extract training data from GitHub repository
training_outputs = pd.read_csv("https://storage.googleapis.com/kagglesdsdata/datasets/11282/15651/breast-cancer-wisconsin.data.txt?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220615%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220615T224034Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=90ab51eb0a1dcbc2037899fa51916d6c13f909210572873dc8bfa3022194e6dde9689e4b6779f50e4063e72e467dbfcd19649c56c32c4fefd86182e8cc57f9645ae2374f3bfff058eda1f41462d241537825a621910733c24e82e6b56fc5971c99ada202bd0bbb696f4a9e4ce1305ccbbd115a1ec642be449c5decc202445f3d622439c8cd0cc4f042fadb83f348a794eba52de5aac8805426ed0c03d199a8175115a2e78d2b167286968d8b3ea778aa44314bbf6a551ef44d5cf907023662e7ab57d6a60a6b0a3e3b48cc1f470e5778d30171cebfc726a4bacd2d9f00f55d170b5ec50ee9b217ff2eba0cd2c224298939be801e187eaaadce139d6950a6a147", skiprows = 100, usecols = [10]).to_numpy()
# Flatten NP array
training_outputs = np.array([item for sublist in training_outputs for item in sublist]) 
training_data = pd.read_csv("https://storage.googleapis.com/kagglesdsdata/datasets/11282/15651/breast-cancer-wisconsin.data.txt?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220615%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220615T224034Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=90ab51eb0a1dcbc2037899fa51916d6c13f909210572873dc8bfa3022194e6dde9689e4b6779f50e4063e72e467dbfcd19649c56c32c4fefd86182e8cc57f9645ae2374f3bfff058eda1f41462d241537825a621910733c24e82e6b56fc5971c99ada202bd0bbb696f4a9e4ce1305ccbbd115a1ec642be449c5decc202445f3d622439c8cd0cc4f042fadb83f348a794eba52de5aac8805426ed0c03d199a8175115a2e78d2b167286968d8b3ea778aa44314bbf6a551ef44d5cf907023662e7ab57d6a60a6b0a3e3b48cc1f470e5778d30171cebfc726a4bacd2d9f00f55d170b5ec50ee9b217ff2eba0cd2c224298939be801e187eaaadce139d6950a6a147", skiprows = 100, usecols = range(1,10)).to_numpy()
# Find all the missing data and remove for simplicity's sake
remove = []
for j in range(len(training_data)):
    if '?' in training_data[j]:
        remove.append(j)
remove.reverse()
for j in remove:
    training_data = np.delete(training_data, j, 0)
    training_outputs = np.delete(training_outputs, j)
training_data = training_data.astype(int)


# Import test data, reapeating process
test_outputs = pd.read_csv("https://storage.googleapis.com/kagglesdsdata/datasets/11282/15651/breast-cancer-wisconsin.data.txt?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220615%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220615T224034Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=90ab51eb0a1dcbc2037899fa51916d6c13f909210572873dc8bfa3022194e6dde9689e4b6779f50e4063e72e467dbfcd19649c56c32c4fefd86182e8cc57f9645ae2374f3bfff058eda1f41462d241537825a621910733c24e82e6b56fc5971c99ada202bd0bbb696f4a9e4ce1305ccbbd115a1ec642be449c5decc202445f3d622439c8cd0cc4f042fadb83f348a794eba52de5aac8805426ed0c03d199a8175115a2e78d2b167286968d8b3ea778aa44314bbf6a551ef44d5cf907023662e7ab57d6a60a6b0a3e3b48cc1f470e5778d30171cebfc726a4bacd2d9f00f55d170b5ec50ee9b217ff2eba0cd2c224298939be801e187eaaadce139d6950a6a147", nrows = 100, usecols = [10]).to_numpy()
test_outputs = np.array([item for sublist in test_outputs for item in sublist]) 
test_data = pd.read_csv("https://storage.googleapis.com/kagglesdsdata/datasets/11282/15651/breast-cancer-wisconsin.data.txt?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220615%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220615T224034Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=90ab51eb0a1dcbc2037899fa51916d6c13f909210572873dc8bfa3022194e6dde9689e4b6779f50e4063e72e467dbfcd19649c56c32c4fefd86182e8cc57f9645ae2374f3bfff058eda1f41462d241537825a621910733c24e82e6b56fc5971c99ada202bd0bbb696f4a9e4ce1305ccbbd115a1ec642be449c5decc202445f3d622439c8cd0cc4f042fadb83f348a794eba52de5aac8805426ed0c03d199a8175115a2e78d2b167286968d8b3ea778aa44314bbf6a551ef44d5cf907023662e7ab57d6a60a6b0a3e3b48cc1f470e5778d30171cebfc726a4bacd2d9f00f55d170b5ec50ee9b217ff2eba0cd2c224298939be801e187eaaadce139d6950a6a147", nrows = 100, usecols = range(1,10)).to_numpy()
remove = []
for j in range(len(test_data)):
    if '?' in test_data[j]:
        remove.append(j)
remove.reverse()
for j in remove:
    test_data = np.delete(test_data, j, 0)
    test_outputs = np.delete(test_outputs, j)
test_data = test_data.astype(int)

# Display accuracy for several different n values
odds = [1,3,5,7,9,11,13,15,17,19]
for i in odds:
    print(f"The KNN algorithm's accuracy where k = {i} is: {knn_accuracy(i, training_data, training_outputs, test_data, test_outputs)}")


