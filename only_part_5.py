import copy
import numpy as np
import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt
from numpy import linalg

def nmf(rating_mat, k, mask, lambda_reg, max_iter=100):

    eps = 1e-5

    rows, columns = rating_mat.shape
    U = np.random.rand(rows, k)
    U = np.maximum(U, eps)

    V = linalg.lstsq(U, rating_mat)[0]
    V = np.maximum(V, eps)

    masked_X = mask * rating_matrix

    for i in range(1, max_iter + 1):

        top = np.dot(masked_X, V.T)
        bottom = (np.add(np.dot((mask * np.dot(U, V)), V.T), lambda_reg * U)) + eps
        U *= top / bottom
        U = np.maximum(U, eps)

        top = np.dot(U.T, masked_X)
        bottom = np.add(np.dot(U.T, mask * np.dot(U, V)), lambda_reg * V) + eps
        V *= top / bottom
        V = np.maximum(V, eps)

    return U,V

# PART 5 
dataset = 'ratings.csv'
data = pd.read_csv(dataset)

rating_matrix = data.pivot_table(index=['userId'], columns=['movieId'], values='rating', fill_value=0)
weight_matrix = rating_matrix.copy()
weight_matrix[weight_matrix > 0] = 1

rating_matrix = rating_matrix.as_matrix()
weight_matrix = weight_matrix.as_matrix()

indices_known_data = zip(*weight_matrix.nonzero())
b = dict(enumerate(indices_known_data))
N = range(len(b))
shuffle(N)
lol = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]
m = lol(N, 10000)
m[9] = m[9] + m[10]

n_folds = 10
k = 100
L = 5

top_movies_ordered = []

precision_10fold = []
hit_rate_10fold = []
false_alarm_rate_10fold = []
total_10fold = []

for i in range(n_folds):
    temp = copy.copy(weight_matrix)
    keys = m[i]
    for key in keys:
        y = indices_known_data[key]
        p, o = zip(y)
        temp[p][o] = 0

    new_rating_matrix = temp
    U,V = nmf(weight_matrix,100,new_rating_matrix,0)
    predicted_matrix = np.dot(U,V)
    predicted_matrix[weight_matrix == 0] = -1

    for i in range(max(data['userId'])):
        user_ratings = predicted_matrix[i]
        top_movies = user_ratings.argsort()[::-1]
        top_movies_ordered.append(top_movies)

    threshold = 3

    hit_val = []
    false_val = []
    total_val = []
    precision_val = []

    for l in range(1, L+1):

        hit = 0
        false = 0
        total = 0

        for i in range(max(data['userId'])):
            recommended_indices = top_movies_ordered[i][0:l]
            for j in range(len(recommended_indices)):
                rating = predicted_matrix[i][recommended_indices[j]]
                if (rating < 0):
                    continue
                if (rating > threshold):
                    if(rating_matrix[i][recommended_indices[j]] > 3):
                        total = total + 1
                        hit = hit + 1
                    else:
                        total = total + 1
                        false = false + 1

        precision_val.append(hit / total)
        hit_val.append(hit)
        total_val.append(total)
        false_val.append(false)

    hit_rate_10fold.append(hit_val)
    false_alarm_rate_10fold.append(false_val)
    total_10fold.append(total_val)
    precision_10fold.append(precision_val)

precision = np.sum(precision_10fold,axis=0)
hits = np.sum(hit_rate_10fold,axis=0)
false_alarm = np.sum(false_alarm_rate_10fold,axis=0)
total = np.sum(total_10fold,axis=0)

hit_rate = hits / (total.astype(float))
false_alarm_rate = false_alarm / (total.astype(float))
precision = precision / 10.0

plt.figure(1)
plt.ylabel('Hit Rate')
plt.xlabel('False Alarm')
plt.title('Hit Rate vs False Alarm')
plt.scatter(false_alarm_rate, hit_rate, s=60, marker='o')
plt.plot(false_alarm_rate,hit_rate)
plt.show()