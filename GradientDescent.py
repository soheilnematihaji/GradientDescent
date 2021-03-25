#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import linalg as LA

import unittest


# In[27]:



class GradientDescent:
    
        
    def __init__(self):
        pass
    
    def gradient(self,function,x,delta_val):
        """
        function: a lambda function as the input to compute function(x) and function(x+delta)
        x: the input to the function it can be numpy array of any length
        delta_val: this is refering to the delta value in calculating the gradient
            for example for 1 dimenstion delta:= d in (f(x+d)-f(x))/d   
        """
        delta = delta_val*np.eye(len(x))# defining len(x) dimention with value of delta_val in the diognal to calculate the gradient
        
        return np.array([ (function(x+delta[i])-function(x))/delta_val for i in range(len(x))])
        
    def gradientDescent(self, function, initial_point, iterations, learning_rate, delta_val):
        theta=initial_point
        for i in range(iterations):
            theta=theta-learning_rate*self.gradient(function,theta,delta_val)
        return theta

    


# In[43]:


class Test(unittest.TestCase):
    
    # This is a variable to generate normal train set with train_size size
    #increasing or decreasing it may effect the test
    train_size=5000
    
    
    # This is a variable to generate huge train set with train_size size
    #increasing this will effect the time of the tests
    efficiency_train_size=1000000
    
    batchsize_eff=100000
    
    def generate_random_noraml_point(self,count,pointCount):
        point=[[(i+count)] for i in range(pointCount)]
        x_train=[[i] for i in range(count)]
        y_train=[[i] for i in range(count)]+np.random.randn(count, 1)
        return point,x_train,y_train 
    
    def f(self,x):
        return x[0]*x[0]+2*x[1]*x[1]
    
    def test_fitLine(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.train_size,1)
        gd=GradientDescent()
        gradient=gd.gradient(self.f,[3,2],0.01)
        assert LA.norm(gradient-[6 ,8]) < 0.03
        min_0=gd.gradientDescent(self.f,np.array([5,5]),200,0.1,0.0001)
        assert LA.norm(min_0-[0 ,0]) < 0.03
        
    
        
        
if __name__== '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# In[ ]:




