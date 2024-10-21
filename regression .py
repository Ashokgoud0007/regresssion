#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[3]:


year_of_experience=[2,4,6,8,10,12,14,16,18,20]
salary=[600000,800000,1000000,1200000,1400000,1600000,1800000,2000000,2200000,2400000]
salary


# In[4]:


year_of_experience


# In[5]:


dataset=pd.DataFrame({"year_of_experience":year_of_experience,"salary":salary})
dataset


# In[6]:


dataset.shape


# In[7]:


dataset.columns


# In[8]:


dataset.info()


# In[9]:


dataset.isnull().sum()


# In[10]:


x=dataset.drop("salary",axis=1)
x


# In[11]:


y=dataset["salary"]
y


# In[12]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[13]:


model=LinearRegression()


# In[14]:


model.fit(x_train,y_train)


# In[15]:


predictions=model.predict(x_test)
predictions


# In[16]:


print(mean_squared_error(y_test,predictions))


# In[17]:


print(mean_absolute_error(y_test,predictions))


# In[18]:


print(np.sqrt(mean_squared_error(y_test,predictions)))


# In[ ]:


newdata=[[int(input("enter the years of exprience"))]]
result=model.predict(newdata)
result


# In[ ]:




