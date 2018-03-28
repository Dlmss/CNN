# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 12:00:31 2018

@author: Ewan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
import poids


def zero_pad(imgSet, pad):
    
    imgSetPad = np.pad(imgSet, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
    return imgSetPad

def conv_single_step(A_slice, W, b):
    
    p = np.multiply(A_slice, W)
    s = np.sum(p)
    return s + b

def conv_forward(A_prev, W, b, parameters):
    """
    A_prev: (m, n_H_prev, n_W_prev, n_C_prev)
    W     : (n_C, f, f, n_C_prev)
    b     : (n_C, 1, 1, 1)
    """
    
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    n_C, f, f, n_C_prev = W.shape
    pad = parameters["pad"]
    
    n_H = n_H_prev + 2*pad - f + 1
    n_W = n_W_prev + 2*pad - f + 1
    A = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    a_slice = a_prev_pad[h:h+f, w:w+f, :]
                    A[i, h, w, c] = conv_single_step(a_slice, W[c, :, :, :], b[c, :, :, :])
    return A

def pool_forward(A_prev, parameters, mode):
    """
    A_prev: (m, n_H_prev, n_W_prev, n_C_prev)
    """
    
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f = parameters["f"]
    n_H = n_H_prev - f + 1
    n_W = n_W_prev - f + 1
    n_C = n_C_prev
    A = np.zeros((m, n_H, n_W, n_C))
    
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    if (mode == "max"):
                        A[i, h, w, c] = np.max(a_prev[h:h+f, w:w+f, c])
                    elif(mode == "average"):
                        A[i, h, w, c] = np.mean(a_prev[h:h+f, w:w+f, c])
    return A





image = imread("image.jpg")
image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))

image = image/255.

parameters  = {'pad': 1}
parametersB  = {'f': 4}
W = poids.set_W()
b = poids.set_b()

A = conv_forward(image, W, b, parameters)
B = pool_forward(A, parametersB, "average")

fig = plt.figure(1)
plt.imshow(image[0,:,:,:])
fig = plt.figure(2)
plt.subplot(221)
plt.imshow(A[0,:,:,0], cmap="gray", interpolation="nearest")
plt.subplot(222)
plt.imshow(A[0,:,:,1], cmap="gray", interpolation="nearest")
plt.subplot(223)
plt.imshow(A[0, :, :, 2], cmap="gray", interpolation="nearest")
plt.subplot(224)
plt.imshow(A[0, :, :, 3], cmap="gray", interpolation="nearest")

print(A[0,0:4,0:4,0])
print(B[0,0:4,0:4,0])
print(A.shape)
print(B.shape)

plt.imshow(B[0, :, :, 1])

plt.show()