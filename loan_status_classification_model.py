#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[52]:


df = pd.read_csv('Credit_Card Defaulters_Dataset.csv')


# In[53]:


df.head()


# In[54]:


df.isnull().sum()


# In[55]:


df.columns


# In[56]:


lis = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
       'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
       'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']


# # Fill all that null values with mean using for loops

# In[57]:


for i in lis:
    df[i]=df[i].fillna(df[i].mean())


# In[58]:


df.isnull().sum()


# In[59]:


df.info()


# In[60]:


df.describe()


# # Display box plot for LIMIT_BAL

# In[61]:


plt.boxplot(df['LIMIT_BAL'])


# # Display box plot for AGE

# In[62]:


plt.boxplot(df['AGE'])


# # Plot scatter plot for AGE & LIMIT_BAL

# In[63]:


plt.scatter(df['AGE'],df['LIMIT_BAL'])


# # Plot scatter plot for AGE & LIMIT_BAL

# In[64]:


plt.scatter(df['LIMIT_BAL'],df['PAY_AMT6'])


# # Find shape of data

# In[65]:


df.shape


# # Perform label encoding on Default Status

# In[66]:


from sklearn.preprocessing import LabelEncoder


# In[67]:


enc =LabelEncoder()


# In[68]:


df['Default Status'] = enc.fit_transform(df['Default Status'])


# In[69]:


df.head()


# # working with model

# # 1. create a features and target set

# In[70]:


X = df.drop('Default Status',axis=1)
y = df['Default Status']


# # 2. split data into training and testing

# In[71]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[72]:


from sklearn.naive_bayes import GaussianNB


# # 3. Apply navie bayes classfier

# In[73]:


clf = GaussianNB()
clf.fit(X_train,y_train)


# # 4.  testing score

# In[74]:


clf.score(X_test,y_test)


# # 5. training score

# In[75]:


clf.score(X_train,y_train)


# In[76]:


y_pred = clf.predict(X_test)
   


# In[77]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,classification_report


# # 6. accuracy score, 

# In[78]:


accuracy_score(y_test,y_pred)


# # 7. precision_score

# In[79]:


precision_score(y_test,y_pred)


# # 8. recall_score

# In[80]:


recall_score(y_test,y_pred)


# # 9. confusion_matrix

# In[81]:


confusion_matrix(y_test,y_pred)


# # 10 .classification_report

# In[82]:


print(classification_report(y_test,y_pred))

