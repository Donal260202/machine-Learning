#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Classifier

# In[31]:


import numpy as no
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import tree


# In[2]:


iris=load_iris()
X=iris.data
y=iris.target


# In[3]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)


# In[4]:


model=DecisionTreeClassifier()


# In[5]:


params={
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


# In[6]:


grid_model=GridSearchCV(model,param_grid=params,cv=5,scoring='accuracy')


# In[7]:


grid_model.fit(X_train,y_train)


# In[8]:


grid_model.best_params_


# In[9]:


best_model=grid_model.best_estimator_


# In[10]:


best_model.fit(X_train,y_train)


# In[11]:


y_pred=best_model.predict(X_test)


# In[12]:


from sklearn.metrics import accuracy_score,classification_report


# In[13]:


print(f"Accuracy_score : {accuracy_score(y_pred,y_test)}")
print(f"Claasification Report :\n {classification_report(y_pred,y_test)}")


# In[32]:


plt.figure(figsize=(15,10))
tree.plot_tree(best_model,filled=True)


# # Decision Tree Regressor
# 

# In[14]:


from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_squared_error


# In[15]:


reg_data=fetch_california_housing()
X=reg_data.data
y=reg_data.target


# In[16]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)


# In[17]:


reg_model=DecisionTreeRegressor()


# In[18]:


params=params={
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


# In[19]:


grid_reg=GridSearchCV(reg_model,param_grid=params,scoring='neg_mean_squared_error',cv=5)
grid_reg.fit(X_train,y_train)


# In[20]:


grid_reg.best_params_


# In[21]:


grid_reg.best_score_


# In[22]:


best_reg_model=grid_reg.best_estimator_


# In[23]:


best_reg_model.fit(X_train,y_train)


# In[24]:


y_pred=best_reg_model.predict(X_test)


# In[25]:


y_pred


# In[26]:


print(f"R2 Score of the regressor is : {r2_score(y_test,y_pred)}")
print(f"Mean Squared Error of the regressor is : {mean_squared_error(y_test,y_pred)}")


# 

# In[34]:


plt.figure(figsize=(15,10))
tree.plot_tree(best_reg_model,filled=True)


# In[ ]:





# In[ ]:




