#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import linalg as LA

import unittest


# In[2]:



class GradientDescent:
    
    
    def gradient(self,function,x,delta_val):
        """
        function: a lambda function as the input to compute function(x) and function(x+delta)
        x: the input to the function it can be numpy array of any length
        delta_val: this is refering to the delta value in calculating the gradient
            for example for 1 dimenstion delta:= d in (f(x+d)-f(x))/d   
        """
        n=len(x)
        delta = delta_val*np.eye(n)# defining len(x) dimention with value of delta_val in the diognal to calculate the gradient
        
        return np.array([ (function(x+delta[i])-function(x))/delta_val for i in range(n)])
        
    def gradientDescent(self, function, initial_point, iterations=10000, learning_rate=0.1, delta_val=0.01):
        
        if type(initial_point)!=np.ndarray:
            raise ValueError("Only accepting ndarrays as input, please update your function")
        theta=initial_point
        for i in range(iterations):
            theta=theta-learning_rate*self.gradient(function,theta,delta_val)
        return theta

    


# In[4]:


class Test(unittest.TestCase):
    
    iterations=500
    efficiency_iterations=100000
    
    def f(self,x):
        return x[0]*x[0]+2*x[1]*x[1]
    
    def test_gradient(self):
        gd=GradientDescent()
        gradient=gd.gradient(self.f,[3,2],0.01)
        assert LA.norm(gradient-[6 ,8]) < 0.03
        
    def test_gradientDescent(self):
        gd=GradientDescent()
        min_val=gd.gradientDescent(self.f,np.array([5,5]),self.iterations,0.1,0.0001)
        assert LA.norm(min_val-[0 ,0]) < 0.03
        
    def test_gradientDescent_eff(self):
        gd=GradientDescent()
        min_val=gd.gradientDescent(self.f,np.array([20,20]),self.efficiency_iterations,0.1,0.0001)
        assert LA.norm(min_val-[0 ,0]) < 0.03
    def test_ValueError(self):
        gd=GradientDescent()
        with self.assertRaises(ValueError) as context:
            min_val=gd.gradientDescent(self.f,5)
    
        
        
if __name__== '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# In[ ]:




