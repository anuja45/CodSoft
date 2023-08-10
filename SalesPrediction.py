#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# In[2]:


sales_data=pd.read_csv('advertising.csv')


# In[3]:


sales_data.head()


# In[4]:


sales_data.shape


# In[5]:


sales_data.columns


# In[6]:


sales_data.info()


# In[7]:


sns.heatmap(sales_data.isnull(),yticklabels=False)


# In[8]:


sales_data.isnull().sum()


# # Visualization

# In[9]:


sns.displot(sales_data['Sales'])


# In[10]:


plt.subplot(2,2,1)
sns.scatterplot(sales_data['TV'],sales_data['Sales'])
plt.subplot(2,2,2)
sns.scatterplot(sales_data['Radio'],sales_data['Sales'])
plt.subplot(2,2,3)
sns.scatterplot(sales_data['Newspaper'],sales_data['Sales'])


# In[11]:


sales_data.plot()


# In[12]:


sales_data.plot(kind='area',subplots=True,layout=(2,2),sharex=False,sharey=False)


# In[13]:


sales_data.plot(kind='line',subplots=True,layout=(2,2),sharex=False,sharey=False)


# In[14]:


sns.lmplot('TV','Sales',data=sales_data)


# In[15]:


sns.lmplot('Radio','Sales',data=sales_data)


# In[16]:


sns.lmplot('Newspaper','Sales',data=sales_data)


# # Model Training

# In[17]:


x=sales_data.iloc[:,[0,1,2]]
y=sales_data.iloc[:,3]


# In[18]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[20]:


x_train.shape,x_test.shape


# In[21]:


reg=LinearRegression()


# In[22]:


model=reg.fit(x_train,y_train)


# In[23]:


train_predict=model.predict(x_train)
train_predict


# In[24]:


r2_score(y_train,train_predict)


# In[25]:


test_predict=model.predict(x_test)
test_predict


# In[26]:


r2_score(y_test,test_predict)


# In[27]:


df=pd.DataFrame({'Actual':y_test.values.flatten(),
                'Predicted':test_predict.flatten()})
df.head()


# In[28]:


df.sample(10).plot(kind='bar')


# # Prediction

# In[29]:


new_data=[[230.1,37.8,69.2]]

model.predict(new_data)

