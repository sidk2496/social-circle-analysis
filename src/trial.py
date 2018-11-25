#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 01:03:56 2018

@author: vignesh2496
"""

import pickle
import numpy as np
with open('../../data/facebook/processed/egonet_1.pkl', 'rb') as file:
    h=pickle.load(file)
    f=np.array(h['nodes'][0].attributes)
    print(f.sum())
    print(h['nodes'][0].attributes.shape)
#def f(x):
#    return x*2
#a=[1,2,3]
#print(sum(map(f, a)))