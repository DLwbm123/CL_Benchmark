# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def backward_transfer(results):
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks - 1):
        li.append((results[-1][i] - results[i][i])/results[i][i])

    return np.mean(li)



'''def forward_transfer(results, random_results):
    n_tasks = len(results)
    li = list()
    for i in range(1, n_tasks):
        #print("task", i)
        li.append(results[i-1][i] - random_results[i])

    return np.mean(li)'''

def forward_transfer(results, random_results):
    n_tasks = len(results)
    li = list()
    for i in range(1, n_tasks):
        #print("task", i)
        for j in range(i, n_tasks):
            li.append(results[i-1][j] - random_results[j])
    return np.mean(li)


def forward_transfer_class(whole_results, whole_random_results):
    n_tasks = len(whole_results)
    li = list()
    for i in range(1, n_tasks):
        li.append(whole_results[i-1][0] - whole_random_results[0])
    return np.mean(li)



def restricted_modelability(results):
    n_tasks = len(results)
    li = list()
    for i in range(1, n_tasks):
        #print("task", i)
        li.append(results[i][i])

    return np.mean(li)

def restricted_modelability_corrected(results1, results2):
    n_tasks = len(results1)
    li = list()
    for i in range(1, n_tasks):
        #print("task", i)
        li.append(results1[i][i]/results2[i][i])

    return np.mean(li)

def task_difficulty(results):
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks-1):
        #print("task", i)
        for j in range(i+1, n_tasks):
            li.append(results[0][0] - results[i][j])
    return np.mean(li)


############################New Metrics###########################
def model_size_efficiency(results, eps = 1e-8):
    #return max(1,np.mean(np.array(results)/results[0]))
    n_tasks = len(results)
    li = list()
    for i in range(1, n_tasks):
        #print("task", i)
        li.append((results[i]-results[i-1])/results[0])
    return np.mean(li)

def samples_storage_size_efficiency(results, ALL_DATA_SIZE):
    return 1-min(1,np.mean(np.array(results)/ALL_DATA_SIZE))


def computational_efficiency(results_onepass, results_wholesets):
    return min(1,np.mean(results_onepass/results_wholesets))
