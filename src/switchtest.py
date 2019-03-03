# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 14:34:09 2019

@author: David Billingsley
"""
"""
def executed1(arg):
    print "executed1 executed"
    return "this the returned string for executed1 " + str(arg)

def executed2(arg):
    print "executed2 executed"
    return "this the returned string for executed2 " + str(arg)

def executed0(arg):
    print "executed2 executed"
    return "this the returned string for executed0 " + str(arg)

def switcher(arg):
    switch = {
            0:executed0,
            1:executed1,
            2:executed2
            }
    return switch.get(arg, "did not find index")

print switcher(2)("test")
"""

import numpy as np
import matplotlib.pyplot as plt


no_imp_acc = np.random.randint(1, 100, 40)
no_avg_imp_acc = np.random.randint(1, 100, 29)
local_slope_acc = np.random.randint(1, 100, 12)

plt.figure()
plt.plot(no_imp_acc, label="no improvement")
plt.plot(no_avg_imp_acc, label="no average improvement")
plt.plot(local_slope_acc, label="local slope")
plt.legend()
    


