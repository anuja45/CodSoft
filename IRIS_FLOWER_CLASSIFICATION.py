#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[6]:


iris_data=pd.read_csv('Iris.csv')


# In[7]:


iris_data.head()


# In[8]:


iris_data.shape


# In[9]:


iris_data.describe()


# In[10]:


iris_data.info()


# In[11]:


sns.heatmap(iris_data.isnull(),yticklabels=False)


# In[12]:


iris_data.isnull().sum()


# In[13]:


iris_data=iris_data.drop('Id',axis=1)


# In[14]:


iris_data.head()


# In[15]:


iris_data['Species'].value_counts()


# # Viaualization

# In[16]:


iris_data.plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue')


# In[17]:


iris_data.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)


# In[18]:


iris_data.plot(kind='line',subplots=True,layout=(2,2),sharex=False,sharey=False)


# In[19]:


iris_data.plot(kind='bar',subplots=True,layout=(2,2),sharex=False,sharey=False)


# In[20]:


iris_data.plot(kind='area',subplots=True,layout=(2,2),sharex=False,sharey=False)


# In[21]:


iris_data.plot(kind='hist',subplots=True,layout=(2,2),sharex=False,sharey=False)


# In[22]:


iris_data.plot()


# In[26]:


plt.subplot(2,2,1)
sns.lineplot('SepalLengthCm','Species',data=iris_data)
plt.subplot(2,2,2)
sns.lineplot('SepalWidthCm','Species',data=iris_data)
plt.subplot(2,2,3)
sns.lineplot('PetalLengthCm','Species',data=iris_data)
plt.subplot(2,2,4)
sns.lineplot('PetalLengthCm','Species',data=iris_data)


# # Model Training

# In[27]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix


# In[28]:


x=iris_data.iloc[:,[0,1,2,3]]
y=iris_data.iloc[:,4]


# In[29]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[30]:


x_train.shape,x_test.shape


# In[31]:


regressor=LogisticRegression()


# In[32]:


model=regressor.fit(x_train,y_train)


# In[33]:


train_prediction=model.predict(x_train)


# In[34]:


train_prediction


# In[35]:


train_accuracy=accuracy_score(y_train,train_prediction)
train_accuracy


# In[36]:


confusion_matrix(y_train,train_prediction)


# In[37]:


test_prediction=model.predict(x_test)
test_prediction


# In[38]:


test_accuracy=accuracy_score(y_test,test_prediction)
test_accuracy


# In[39]:


confusion_matrix(y_test,test_prediction)


# In[44]:


df=pd.DataFrame({'Actual':y_test,'Predicted':test_prediction})
df.head()


# # Prediction

# In[40]:


new_data=np.array([[4.6,3.1,1.4,0.2]])
prediction=model.predict(new_data)
prediction[0]

