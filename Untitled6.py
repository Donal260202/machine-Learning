#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("Life Expectancy Data.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


rows,columns=df.shape
print(f"Number of rows in dataset : {rows}")
print(f"Number of columns in dataset : {columns}")


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.duplicated().sum()


# # EDA
# 

# In[9]:


df.describe()


# In[10]:


for i in df.select_dtypes(include="number").columns:
    sns.histplot(data=df,x=i)
    plt.show()


# In[11]:


for i in df.select_dtypes(include="number").columns:
    sns.boxplot(data=df,x=i)
    plt.show()


# In[12]:


df.select_dtypes(include="number").columns


# In[13]:


for col in ['Year',  'Adult Mortality', 'infant deaths',
       'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ',
       'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
       ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years',
       ' thinness 5-9 years', 'Income composition of resources', 'Schooling']:
    sns.scatterplot(data=df,x=col,y='Life expectancy ')
    plt.show()


# In[14]:


s=df.select_dtypes(include="number").corr()


# In[15]:


plt.figure(figsize=(15,15))
sns.heatmap(data=s,annot=True)


# # MISSING VALUE TREATMENT
# 

# In[ ]:





# In[16]:


df.isnull().sum()


# In[17]:


df.columns


# In[18]:


for i in [' BMI ','Polio','Income composition of resources']:
    df[i].fillna(df[i].median(),inplace=True)


# In[19]:


from sklearn.impute import KNNImputer
impute=KNNImputer()


# In[20]:


for i in df.select_dtypes(include="number").columns:
    df[i]=impute.fit_transform(df[[i]])


# In[21]:


df.isnull().sum()


# # Outlier Treatment

# In[22]:


# Define whisker calculation function
def whisker(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lw = Q1 - 1.5 * IQR
    uw = Q3 + 1.5 * IQR
    return lw, uw

# Calculate whiskers
lw, uw = whisker(df['GDP'])

# Capping values below lw and above uw
df['GDP'] = np.where(df['GDP'] < lw, lw, df['GDP'])  # Cap below lw
df['GDP'] = np.where(df['GDP'] > uw, uw, df['GDP'])  # Cap above uw


# In[23]:


sns.boxplot(df['GDP'])
plt.show()


# In[ ]:





# In[24]:



sns.boxplot(df['GDP'])
plt.show()


# In[ ]:





# In[ ]:




