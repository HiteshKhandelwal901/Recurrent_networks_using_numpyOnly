#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[40]:


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def rnn_cell_forward(xt, a_prev, parameters):
    
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    print(" shape of xt output  = ", (np.dot(Wax,xt).shape))
    print(" shape of a output  = ", (np.dot(Waa,a_prev).shape))
    
    a_next = np.tanh(np.dot(Wax,xt) + np.dot(Waa, a_prev) + ba)

    print("shape of final additon  = ", a_next.shape)
    yt_pred = softmax(np.dot(Wya, a_next) + by)
    cache = (a_next, a_prev, xt, parameters)
    
    return a_next, yt_pred, cache


def rnn_forward(x, a0, parameters):
    
    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    a = np.zeros((n_a,m,T_x))
    y_pred = np.zeros((n_y,m,T_x))
    a_next = a0
    
  
    for t in range(T_x):
        xt = x[:,:,t]
        a_next, yt_pred, cache = rnn_cell_forward(xt, a_next, parameters)
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)
        
    caches = (caches, x)
    
    return a, y_pred, caches


# In[41]:


np.random.seed(1)
x_tmp = np.random.randn(3,10,4)
a0_tmp = np.random.randn(5,10)
parameters_tmp = {}
parameters_tmp['Waa'] = np.random.randn(5,5)
parameters_tmp['Wax'] = np.random.randn(5,3)
parameters_tmp['Wya'] = np.random.randn(2,5)
parameters_tmp['ba'] = np.random.randn(5,1)
parameters_tmp['by'] = np.random.randn(2,1)

a_tmp, y_pred_tmp, caches_tmp = rnn_forward(x_tmp, a0_tmp, parameters_tmp)
print("a[4][1] = \n", a_tmp[4][1])
print("a.shape = \n", a_tmp.shape)
print("y_pred[1][3] =\n", y_pred_tmp[1][3])
print("y_pred.shape = \n", y_pred_tmp.shape)
print("caches[1][1][3] =\n", caches_tmp[1][1][3])
print("len(caches) = \n", len(caches_tmp))


# In[23]:





# In[28]:





# In[35]:





# In[36]:





# In[37]:





# In[ ]:




