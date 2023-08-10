#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns


# In[2]:


data=pd.read_csv('tested.csv')


# In[3]:


data.head()


# In[4]:


data[['Sex','Survived']]


# In[5]:


data.shape


# In[6]:


data.describe()


# In[7]:


data.info()


# In[8]:


sns.heatmap(data.isnull(),yticklabels=False)


# In[9]:


data.isnull().sum()


# In[10]:


sns.countplot(x='Survived',data=data,hue='Sex')


# In[11]:


data.Age.fillna(data.Age.mean(),inplace=True)


# In[12]:


data.isnull().sum()


# In[13]:


data['Age'].plot.hist(bins=30)


# In[14]:


data.Fare.fillna(data.Fare.mean(),inplace=True)


# In[15]:


data.isnull().sum()


# In[16]:


data.Embarked.mode()[0]


# In[17]:


data.Sex=data.Sex.map({'male':0,'female':1})


# In[18]:


data.head()


# In[19]:


data.Embarked.unique()


# In[20]:


data.Embarked=data.Embarked.map({'S':1,'Q':2,'C':3})


# In[21]:


data.head()


# In[22]:


sns.countplot(x='Survived',data=data,hue='Embarked')


# In[23]:


data=data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)


# In[24]:


data.head()


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


x=data.iloc[:,[1,2,3,4,5,6,7]]
y=data.iloc[:,0]


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[28]:


x_train.shape,x_test.shape


# In[29]:


from sklearn.linear_model import LogisticRegression


# In[30]:


regressor=LogisticRegression()


# In[31]:


model=regressor.fit(x_train,y_train)


# In[32]:


x_train_predict=model.predict(x_train)


# In[33]:


x_train_predict


# In[34]:


x_test_predict=model.predict(x_test)


# In[35]:


x_test_predict


# In[36]:


from sklearn.metrics import accuracy_score


# In[37]:


train_accuracy=accuracy_score(y_train,x_train_predict)


# In[38]:


train_accuracy


# In[39]:


test_accuracy=accuracy_score(y_test,x_test_predict)


# In[40]:


test_accuracy


# # Prediction on new data

# In[41]:


new_data=[[3,0,27.0,0,0,8.6625,1]]
prediction=model.predict(new_data)
if(prediction[0]==0):
    print('not survived')
else:
    print('survived')


# In[ ]:




