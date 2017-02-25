from random import shuffle
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

# PART 1
dataset = 'ratings.csv'
data = pd.read_csv(dataset)

rating_matrix = pd.pivot_table(data, values='rating', index=['userId'], columns=['movieId'], fill_value = 0)
weight_matrix = rating_matrix.copy()
weight_matrix[weight_matrix > 0] = 1

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
weight_matrix_array = np.asarray(weight_matrix)
indices_known_data = zip(*weight_matrix_array.nonzero())
b = dict(enumerate(indices_known_data))
N = range(len(b))
shuffle(N)

lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
m = lol(N,10000)

n_folds = 10
k = 100
error =[]
for i in range(n_folds):
    temp = np.asarray(weight_matrix)
    keys = m[i]
    for key in keys:
        y = indices_known_data[key]
        p,o = zip(y)
        temp[p][o] = 0
        new_weight_matrix = pd.DataFrame(temp)

    nmf = NMF(n_components=k)
    rating_matrix = np.multiply(weight_matrix,rating_matrix)
    U = nmf.fit_transform(rating_matrix)
    V = nmf.components_
    predicted_rating_matrix = np.dot(U, V)

    sum = 0
    for key in keys:
        y = indices_known_data[key]
        p,o = zip(y)
        sum = sum + predicted_rating_matrix[p[0]][o[0]]

    error.append(abs(10000 - sum)/10000)

    print 'Testing Error in Fold-%d: ' %(i+1) + str(error[i])

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
