# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 21:10:04 2019

@author: David Billingsley
"""

import mnist_loader

import matplotlib.pyplot as plt


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network2L2earlystop as n2ES


# no improvement in k
print "Testing no improvement in k"
net_no_improvement = n2ES.Network([784, 30, 10], cost=n2ES.CrossEntropyCost)

evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net_no_improvement.SGD(training_data, 50, 10, 0.50,
    lmbda = 5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True,
    monitor_evaluation_cost=True,
    monitor_training_accuracy=True,
    monitor_training_cost=True,
    rule = 1,
    k_sample=10,
    epsilon=0.1)

no_imp_acc = evaluation_accuracy
no_imp_final = evaluation_accuracy[-1]
no_imp_stop = len(evaluation_accuracy)-1


#no average improvement in k
print "Testing no average improvement in k"
net_no_avg_improvement = n2ES.Network([784, 30, 10], cost=n2ES.CrossEntropyCost)

evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net_no_avg_improvement.SGD(training_data, 50, 10, 0.50,
    lmbda = 5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True,
    monitor_evaluation_cost=True,
    monitor_training_accuracy=True,
    monitor_training_cost=True,
    rule = 2,
    k_sample=10,
    epsilon=0.1)

no_avg_imp_acc = evaluation_accuracy
no_avg_imp_final = evaluation_accuracy[-1]
no_avg_imp_stop = len(evaluation_accuracy)-1


#local_slope
print "Testing local_slope"
net_local_slope = n2ES.Network([784, 30, 10], cost=n2ES.CrossEntropyCost)

evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net_local_slope.SGD(training_data, 50, 10, 0.50,
    lmbda = 5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True,
    monitor_evaluation_cost=True,
    monitor_training_accuracy=True,
    monitor_training_cost=True,
    rule = 3,
    k_sample=10,
    epsilon=0.1)

local_slope_acc = evaluation_accuracy
local_slope_final = evaluation_accuracy[-1]
local_slope_stop = len(evaluation_accuracy)-1

plt.figure()
plt.plot(no_imp_acc, label="no improvement")
plt.plot(no_avg_imp_acc, label="no average improvement")
plt.plot(local_slope_acc, label="local slope")
plt.legend()
plt.show()

print "k-no-improvement"
print "Final accuracy: " + str(no_imp_final)
print "Stopped after epoch " + str(no_imp_stop)
print " "

print "k-no-avg-improvement"
print "Final accuracy: " + str(no_avg_imp_final)
print "Stopped after epoch " + str(no_avg_imp_stop)
print " "
 
print "local slope"
print "Final accuracy: " + str(local_slope_final)
print "Stopped after epoch " + str(local_slope_stop)




"""results = [evaluation_cost, evaluation_accuracy, training_cost, training_accuracy]

eval_acc_pct = [x / 10000.0 for x in evaluation_accuracy]

print("final accuracy is " + str(eval_acc_pct[-1]))
plt.plot(eval_acc_pct)"""









