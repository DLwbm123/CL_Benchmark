# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def backward_transfer(results):
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

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

def task_difficulty(results):
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks-1):
        #print("task", i)
        for j in range(i+1, n_tasks):
            li.append(results[0][0] - results[i][j])
    return np.mean(li)
