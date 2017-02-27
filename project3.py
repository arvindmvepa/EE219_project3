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

# PART 2 & PART 3
indices_known_data = zip(*weight_matrix.nonzero())
b = dict(enumerate(indices_known_data))
N = range(len(b))
shuffle(N)

threshold_value = np.arange(1, 6, 1)
n_folds = 10
k = 100
error =[]
precision = np.zeros((len(threshold_value),n_folds))
recall = np.zeros((len(threshold_value),n_folds))

lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
m = lol(N,10000)
m[9] = m[9] + m[10]

for i in range(n_folds):
    temp = copy.copy(weight_matrix)
    keys = m[i]
    for key in keys:
        y = indices_known_data[key]
        p,o = zip(y)
        temp[p][o] = 0

    new_weight_matrix = temp

    nmf = NMF(n_components = k)
    temp_rating_matrix = np.multiply(new_weight_matrix,rating_matrix)

    U = nmf.fit_transform(temp_rating_matrix)
    V = nmf.components_
    #U,V = nmf(rating_matrix,k,new_weight_matrix)
    predicted_rating_matrix = np.dot(U, V)

    #error = su(rating_matrix-predicted_rating_matrix)/ len(m[i])
    #abs( R_predicted[test_data[j][0] - 1, test_data[j][1] - 1] - test_data[j][2] )

    sum = 0
    for key in keys:
        y = indices_known_data[key]
        p,o = zip(y)
        sum = sum + abs(rating_matrix[p][o] - predicted_rating_matrix[p][o])

    sum = sum/(len(m[i]))

    error.append(sum)

    print 'Testing Error in Fold-%d: ' %(i+1) + str(error[i])

    for s, t in enumerate(threshold_value):

        tp = 0  # true positive
        fp = 0  # false positive
        fn = 0  # false negative

        for key in keys:
            y = indices_known_data[key]
            p, o = zip(y)
            if predicted_rating_matrix[p][o]>t:
                if rating_matrix [p][o]>t:
                    tp = tp + 1
                else:
                    fp = fp + 1
            elif rating_matrix [p][o]>t:
                    fn = fn + 1

        precision[s, i] = tp / float(tp + fp)  # calculating precision
        recall[s, i] = tp / float(tp + fn)  # calculating recall

    avg_precision = np.mean(precision, axis=1)
    avg_recall = np.mean(recall, axis=1)

    plt.ylabel('Recall')
    plt.xlabel('Precision')
    plt.title('ROC')
    plt.scatter(avg_precision, avg_recall, s=40, marker='o')
    plt.plot(avg_precision,avg_recall)
    plt.show()

print 'Highest Cross Validation Error: ' + str(min(error))
print 'Lowest Cross Validation Error: ' + str(max(error))
print 'Average Error of 10 folds: ' + str(np.mean(error))

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
