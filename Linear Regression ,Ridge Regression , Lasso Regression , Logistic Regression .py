#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=load_boston()


# In[3]:


df


# In[4]:


data=pd.DataFrame(df.data)
data.columns=df.feature_names


# In[5]:


data.head()


# In[6]:


data['Price']=df.target


# In[7]:


data.head()


# In[8]:


X=data.iloc[:,:-1]
y=data.iloc[:,-1]


# In[9]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
model=LinearRegression()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[22]:


mse=cross_val_score(model,X_train,y_train,scoring='neg_mean_squared_error',cv=10)
mean_mse=np.mean(mse)


# In[23]:


print(mse)
print(mean_mse)


# In[24]:


model.fit(X_train,y_train)


# In[25]:


y_pred=model.predict(X_test)


# In[41]:


from sklearn.metrics import r2_score
r2_score_linear=r2_score(y_pred,y_test)
print(r2score1)


# In[26]:


#Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[15]:


ridge=Ridge()


# In[27]:


params={"alpha":[1e-20,1e-18,1e-15,1e-10,1e-8,1e-3,1e-2,1e-1,1,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100]}
ridge_regressor=GridSearchCV(ridge,params,scoring='neg_mean_squared_error',cv=10)
ridge_regressor.fit(X_train,y_train)


# In[28]:


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[40]:


y_predr=ridge_regressor.predict(X_test)
r2_score_ridge=r2_score(y_predr,y_test)
print(r2_score_ridge)


# In[29]:


#Lasso Regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


# In[30]:


lasso=Lasso()


# In[31]:


params={"alpha":[1e-20,1e-18,1e-15,1e-10,1e-8,1e-3,1e-2,1e-1,1,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100]}
lasso_regressor=GridSearchCV(lasso,params,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(X_train,y_train)


# In[32]:


print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[38]:


y_predl=lasso_regressor.predict(X_test)
r2_score_lasso=r2_score(y_predl,y_test)
print(r2_score_lasso)


# In[43]:


print(f"Linear Regression R2 Score : {r2_score_linear}")
print(f"Ridge Regression R2 Score : {r2_score_ridge}")
print(f"Lasso Regression R2 Score : {r2_score_lasso}")


# In[44]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer


# In[51]:


df=load_breast_cancer()
X=pd.DataFrame(data=df['data'],columns=df['feature_names'])


# In[54]:


X.head()


# In[53]:


y=pd.DataFrame(data=df['target'],columns=["Target"])


# In[55]:


y.head()


# In[58]:


#checking the data is balanced or not
y['Target'].value_counts()


# In[59]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)


# In[60]:


params={"C":[1,5,10,15,20],'max_iter':[50,100,150]}


# In[61]:


model1=LogisticRegression(C=100,max_iter=100)


# In[62]:


final_model=GridSearchCV(model1,param_grid=params,scoring='f1',cv=5)


# In[66]:


print(final_model.best_params_)
print(final_model.best_score_)


# In[64]:


final_model.fit(X_train,y_train)


# In[67]:


y_pred=final_model.predict(X_test)


# In[69]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[70]:


print("Confussion_matrix : ")
confusion_matrix(y_pred,y_test)


# In[71]:


print(f"Accuracy Score : {accuracy_score(y_pred,y_test)}")


# In[72]:


print("Classification_Report")
print(classification_report(y_pred,y_test))


# In[ ]:




