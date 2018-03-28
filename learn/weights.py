# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 17:05:38 2018

@author: Ewan
"""

import numpy as np

def set_W():     # Filter 1
    W = np.array([[[[10, 10, 10],
                   [ 0, 0, 0],
                   [ -10, -10, -10]],
        
                  [[ 10, 10, 10],
                   [ 0, 0, 0],
                   [ -10, -10, -10]],
                   
                  [[ 10, 10, 10],
                   [ 0, 0, 0],
                   [ -10, -10, -10]]],
                   
                 # Filter 2  
                 [[[ 10, 10, 10],
                   [ 10, 10, 10],
                   [ 10, 10, 10]],
    
                  [[ 0, 0, 0],
                   [ 0, 0, 0],
                   [ 0, 0, 0]],
                   
                  [[ -10, -10, -10],
                   [ -10, -10, -10],
                   [ -10, -10, -10]]],
                   
                 # Filter 3 
                 [[[ -1, -1, -1],
                   [ -1, -1, -1],
                   [ -1, -1, -1]],
    
                  [[ -1, -1, -1],
                   [ 8, 8, 8],
                   [ -1, -1, -1]],
                   
                  [[ -1, -1, -1],
                   [ -1, -1, -1],
                   [ -1, -1, -1]]],
                   
                 # Filter 4  
                 [[[ -1, -1, -1],
                   [ -1, -1, -1],
                   [ -1, -1, -1]],
    
                  [[ 2, 2, 2],
                   [ 2, 2, 2],
                   [ 2, 2, 2]],
                   
                  [[ -1, -1, -1],
                   [ -1, -1, -1],
                   [ -1, -1, -1]]]])
    return W

def set_b():
    b = np.array([[[[0.1]]],
                  [[[0.2]]],
                  [[[0.3]]],
                  [[[0.4]]]])
    return b