#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.tree import export_graphviz
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


climate = pd.read_csv('C://Users//Sonal S//Downloads//temps.csv')
climate


# In[3]:


climate.head(5)


# In[4]:


climate.shape


# In[5]:


climate.columns


# In[6]:


climate.isnull().sum()


# In[7]:


##One hot coding for representation of categorical data to be more expressive
climate= pd.get_dummies(climate)
climate.head(5)


# In[8]:


print('Shape of features after one-hot encoding:', climate.shape)


# In[9]:


#labels are the values we want to predict
labels=climate['actual']


#Remove the labels from the features
climate=climate.drop('actual', axis=1)

#Saving feature names for later use
feature_list=list(climate.columns)


# In[10]:


##Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

#Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(climate, labels, test_size = 0.20,
                                                                        
                                                                            
random_state = 42)


# In[11]:


print('Training features shape:', train_features.shape)
print('Training label shape:', train_labels.shape)
print('Testing features shape:', test_features.shape)
print('Testing labels shape:', test_labels.shape)


# In[12]:


#import the model we are using
from sklearn.ensemble import RandomForestRegressor

#instantiate model
rf= RandomForestRegressor(n_estimators=1000, random_state=42)

#Train the model on Training data
rf.fit(train_features, train_labels);


# In[13]:


##Use forest's predict method on the test data
predictions = rf.predict(test_features)

##Calculate the absolute errors
errors= abs(predictions - test_labels)

##Print out the mean absolute error(mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[14]:


##Calculate mean absolute percentage error(Mape)

Mape = 100*(errors/test_labels)


##Calculate the display accuracy
accuracy=100-np.mean(Mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[24]:


print('The depth of this tree is:', tree.tree_.max_depth)

