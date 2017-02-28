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

# PART 1
dataset = 'ratings.csv'
data = pd.read_csv(dataset)

rating_matrix = data.pivot_table(index=['userId'], columns=['movieId'], values='rating', fill_value=0)
weight_matrix = rating_matrix.copy()
weight_matrix[weight_matrix > 0] = 1

rating_matrix = rating_matrix.as_matrix()
weight_matrix = weight_matrix.as_matrix()

for k in [10, 50, 100]:
    U,V = nmf(rating_matrix,k,weight_matrix,0)
    predicted_rating_matrix = np.dot(U, V)

    error = rating_matrix - predicted_rating_matrix
    squared_error = np.multiply(error,error)
    squared_error = np.multiply(weight_matrix, squared_error)
    sum_squared_error = sum(sum(squared_error))

    print 'Least Squares Error for k = %d: ' % k + str(sum_squared_error)

# PART 2 & PART 3
indices_known_data = zip(*weight_matrix.nonzero())
b = dict(enumerate(indices_known_data))
N = range(len(b))
shuffle(N)

threshold_value = np.arange(0, 6, 1)
n_folds = 10
k = 10
error = []
precision = np.zeros((len(threshold_value), n_folds))
recall = np.zeros((len(threshold_value), n_folds))
true_positives = np.zeros((len(threshold_value), n_folds))
false_positives = np.zeros((len(threshold_value), n_folds))

lol = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]
m = lol(N, 10000)
m[9] = m[9] + m[10]

for i in range(n_folds):
    temp = copy.copy(weight_matrix)
    keys = m[i]
    for key in keys:
        y = indices_known_data[key]
        p, o = zip(y)
        temp[p][o] = 0

    new_weight_matrix = temp
    U,V = nmf(rating_matrix,k,new_weight_matrix,0)
    predicted_rating_matrix = np.dot(U, V)

    sum = 0
    for key in keys:
        y = indices_known_data[key]
        p, o = zip(y)
        sum = sum + abs(rating_matrix[p][o] - predicted_rating_matrix[p][o])

    sum = sum / (len(m[i]))

    error.append(sum)

    print 'Testing Error in Fold-%d: ' % (i + 1) + str(error[i])

    for s, threshold in enumerate(threshold_value):

        tp = 0  # true positive
        tn = 0  # true negative
        fp = 0  # false positive
        fn = 0  # false negative

        for key in keys:
            y = indices_known_data[key]
            p, o = zip(y)
            if predicted_rating_matrix[p][o] >= threshold:
                if rating_matrix[p][o] > 3:
                    tp = tp + 1
                else:
                    fp = fp + 1
            elif predicted_rating_matrix[p][o] < threshold:
                if rating_matrix[p][o] > 3:
                    fn = fn + 1
                else:
                    tn = tn + 1

        precision[s, i] = tp / float(tp + fp)
        recall[s, i] = tp / float(tp + fn)

avg_precision = np.mean(precision, axis=1)
avg_recall = np.mean(recall, axis=1)

plt.title('ROC for k = %d' %(k))
plt.plot(avg_recall, avg_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

print 'Highest Cross Validation Error: ' + str(min(error))
print 'Lowest Cross Validation Error: ' + str(max(error))
print 'Average Error of 10 folds: ' + str(np.mean(error))

# PART 4
dataset = 'ratings.csv'
data = pd.read_csv(dataset)

rating_matrix = data.pivot_table(index=['userId'], columns=['movieId'], values='rating', fill_value=0)
weight_matrix = rating_matrix.copy()
weight_matrix[weight_matrix > 0] = 1

rating_matrix = rating_matrix.as_matrix()
weight_matrix = weight_matrix.as_matrix()

for k in [10, 50, 100]:
    U,V = nmf(weight_matrix,k,rating_matrix,0)
    predicted_rating_matrix = np.dot(U, V)

    error = rating_matrix - predicted_rating_matrix
    squared_error = np.multiply(error,error)
    squared_error = np.multiply(rating_matrix, squared_error)
    sum_squared_error = squared_error.sum().sum()

    print 'Least Squares Error (Weights - Ratings changed) for k = %d: ' % k + str(sum_squared_error)

indices_known_data = zip(*weight_matrix.nonzero())
b = dict(enumerate(indices_known_data))
N = range(len(b))
shuffle(N)

lol = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]
m = lol(N, 10000)
m[9] = m[9] + m[10]

lambda_values = [0.01,0.1,1]
avg_precision = []
avg_recall = []

for a,lambda_value in enumerate(lambda_values):
    threshold_value = np.arange(0, 6, 1)
    n_folds = 10
    k = 100
    error = []
    precision = np.zeros((len(threshold_value), n_folds))
    recall = np.zeros((len(threshold_value), n_folds))
    print 'For lambda: %f' %lambda_value

    for i in range(n_folds):

        temp = copy.copy(weight_matrix)
        keys = m[i]
        for key in keys:
            y = indices_known_data[key]
            p, o = zip(y)
            temp[p][o] = 0

        new_weight_matrix = temp
        U,V = nmf(rating_matrix,k,new_weight_matrix,lambda_value)
        predicted_rating_matrix = np.dot(U, V)

        sum = 0
        for key in keys:
            y = indices_known_data[key]
            p, o = zip(y)
            sum = sum + abs(rating_matrix[p][o] - predicted_rating_matrix[p][o])

        sum = sum / (len(m[i]))

        error.append(sum)

        print 'Testing Error in Fold-%d: ' %(i + 1) + str(error[i])

        for s, threshold in enumerate(threshold_value):

            tp = 0  # true positive
            tn = 0  # true negative
            fp = 0  # false positive
            fn = 0  # false negative

            for key in keys:
                y = indices_known_data[key]
                p, o = zip(y)
                if predicted_rating_matrix[p][o] >= threshold:
                    if rating_matrix[p][o] > 3:
                        tp = tp + 1
                    else:
                        fp = fp + 1
                elif predicted_rating_matrix[p][o] < threshold:
                    if rating_matrix[p][o] > 3:
                        fn = fn + 1
                    else:
                        tn = tn + 1

            precision[s, i] = tp / float(tp + fp)  # calculating precision
            recall[s, i] = tp / float(tp + fn)  # calculating recall

    avg_precision.append(np.mean(precision, axis=1))
    avg_recall.append(np.mean(recall, axis=1))

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(avg_recall[a], avg_precision[a])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('ROC for k = %d & lambda = %f' % (k, lambda_value))
    plt.show()

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
L = [1,3,5]

precision = np.zeros((len(L),n_folds))
hit_rate = np.zeros((len(L), n_folds))
false_alarm_rate = np.zeros((len(L), n_folds))

for i in range(n_folds):
    temp = copy.copy(rating_matrix)
    keys = m[i]
    for key in keys:
        y = indices_known_data[key]
        p, o = zip(y)
        temp[p][o] = 0

    new_weight_matrix = temp

    U,V = nmf(weight_matrix,k,new_weight_matrix,0)
    predicted_rating_matrix = np.dot(U, V)

    # sort by the predicted ratings and actual ratings matrices for each user
    # store the original indices of the sorted elements
    recs = np.argsort(predicted_rating_matrix, axis=1)
    actual = np.argsort(rating_matrix, axis=1)

    for g in range(len(L)):
        tp = 0  # true positive
        fp = 0  # false positive

        # get the top L recommendations and top L actual values
        top_recs = recs[:,(-1 * L[g]):]
        top_actual = actual[:,(-1 * L[g]):]

        # for each user
        for j in xrange(top_recs.shape[0]):
            # count the number of true positives, i.e. recs correctly guessed
            tp += sum(i in top_actual[j,:] for i in top_recs[j,:])

            # count the number of false positives, i.e. movies that were rated by the user,
            # but do not appear in his top 5, we exclude unknown data from fp count
            fp += sum(i not in top_actual[j,:] and (j,i) in indices_known_data for i in top_recs[j,:])

        precision[g][i] = tp / max(1, float(tp + fp))  # we take max between 1 and the positives to avoid undefined
        # print 'Precision in Fold-%d (L=%d): ' % (i + 1, L[g]) + str(precision[g][i])

        threshold = 4
        tp = 0
        fp = 0
        liked_count = 0
        disliked_count = 0

        # sum all the 'liked' movies and 'disliked' movies in the test data
        # and compute the number of true positives and false positives
        for key in keys:
            y = indices_known_data[key]
            p, o = zip(y)

            if rating_matrix[p][o] >= threshold:
                liked_count += 1
                if o in top_recs[p]:
                    tp = tp + 1
            elif rating_matrix[p][o] > 0: # ignore 0s because they are unknown
                disliked_count += 1
                if o in top_recs[p]:
                    fp = fp + 1

        hit_rate[g][i] = tp / max(1, float(liked_count))
        false_alarm_rate[g][i] = fp / max(1, float(disliked_count))

avg_precision = np.mean(precision, axis=1)
avg_hit_rate = np.mean(hit_rate, axis=1)
avg_false_alarm_rate = np.mean(false_alarm_rate, axis=1)

for g in range(len(L)):
    print 'Average Precision for L = %d: ' %(L[g]) + str(avg_precision[g])
    plt.scatter(avg_false_alarm_rate[g], avg_hit_rate[g], label='Label=%d' %(L[g]))

plt.xlabel('False-Alarm Rate')
plt.ylabel('Hit Rate')
plt.title('Hit Rate Vs. False-Alarm Rate')
plt.legend('upper right')
plt.show()