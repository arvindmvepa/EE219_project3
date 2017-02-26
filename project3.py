from random import shuffle
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
import copy

# PART 1
dataset = 'ratings.csv'
data = pd.read_csv(dataset)

rating_matrix = pd.pivot_table(data, values='rating', index=['userId'], columns=['movieId'], fill_value = 0)
weight_matrix = rating_matrix.copy()
weight_matrix[weight_matrix > 0] = 1
weight_matrix = weight_matrix.astype(int)

rating_matrix = rating_matrix.as_matrix()
weight_matrix = weight_matrix.as_matrix()

for k in [10, 50, 100]:
    nmf = NMF(n_components = k)
    U = nmf.fit_transform(rating_matrix)
    V = nmf.components_
    predicted_rating_matrix = np.dot(U,V)

    error = rating_matrix - predicted_rating_matrix
    squared_error = np.multiply(error,error)
    squared_error = np.multiply(weight_matrix, squared_error)
    sum_squared_error = squared_error.sum().sum()

    print 'Least Squares Error for k = %d: ' %k + str(sum_squared_error)

# PART 2
indices_known_data = zip(*weight_matrix.nonzero()) # (row,column) indices of nonzero elements 
b = dict(enumerate(indices_known_data))  # creating a dictionary {1:(row,column) 2:(row,column) ... }
N = range(len(b))   # shuffling 
shuffle(N)


# dividing known points into ten sets after shuffling, take 10004 points for the last set
lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
m = lol(N,10000)
m[9] = m[9] + m[10]



n_folds = 10
k = 100     # k is chosen to be 100 here, in part 1 we used 10, 50 and 100
error =[]

for i in range(n_folds):   # for all 10 sets(10 folds which has 10000 elements only last set has 10004)
    temp = copy.copy(weight_matrix) 
    keys = m[i]                         # get the keys of known elements in the test set
    for key in keys:                    
        y = indices_known_data[key]     
        p,o = zip(y)                     # get row and column indices of known elements in test set
        temp[p][o] = 0                   # put 0 in the weight_matrix for the elements in test set
        
    new_weight_matrix = temp 

    nmf = NMF(n_components=k)
    temp_rating_matrix = np.multiply(new_weight_matrix,rating_matrix)   # get new rating_matrix with known data, elements in test set is extrated 
    U = nmf.fit_transform(temp_rating_matrix)                 
    V = nmf.components_
    predicted_rating_matrix = np.dot(U, V)   # our prediction rating matrix with know elements(known elements in test set extracted)

    sum = 0
    for key in keys:
        y = indices_known_data[key]
        p,o = zip(y)
        sum = sum + abs(rating_matrix[p][o] - predicted_rating_matrix[p][o])

    sum = sum/(len(m[i]))

    error.append(sum)

    print 'Testing Error in Fold-%d: ' %(i+1) + str(error[i])
    
print 'Highest Cross Validation Error: ' + str(min(error))
print 'Lowest Cross Validation Error: ' + str(max(error))
print 'Average Error of 10 folds: ' + str(np.mean(error))

# PART 3

# PART 4
for k in [10, 50, 100]:
    nmf = NMF(n_components = k)
    U = nmf.fit_transform(weight_matrix)
    V = nmf.components_
    predicted_weight_matrix = np.dot(U,V)

    error = weight_matrix - predicted_weight_matrix
    squared_error = np.multiply(error,error)
    squared_error = np.multiply(weight_matrix, squared_error)
    sum_squared_error = squared_error.sum().sum()

    print 'Least Squares Error for k = %d: ' %k + str(sum_squared_error)

# PART 5

# use the rating matrix as the weight matrix this time
weight_matrix = pd.pivot_table(data, values='rating', index=['userId'], columns=['movieId'], fill_value = 0)

# use matrix of 0s and 1s as rating matrix
R = weight_matrix.copy()
R[R > 0] = 1

k = 100
nmf = NMF(n_components = k)
U = nmf.fit_transform(R)
V = nmf.components_
predicted_rating_matrix = np.dot(U,V)

error = R - predicted_rating_matrix
squared_error = np.multiply(error,error)
squared_error = np.multiply(weight_matrix, squared_error)
sum_squared_error = squared_error.sum().sum()
print 'Least Squares Error for k = %d: ' %k + str(sum_squared_error)
