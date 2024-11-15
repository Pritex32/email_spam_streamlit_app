#!/usr/bin/env python
# coding: utf-8

# # Analysis By prisca

# In[9]:


# this data is to categprize spam emails and non spam emails to reduce peope falling for malicious emails


# In[10]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # Data cleaning and exploration

# In[11]:


df=pd.read_csv(r'C:\Users\USER\Documents\dataset\email.csv') 


# In[12]:


df.head(6)


# In[13]:


df.shape # size of the data


# In[14]:


df.isnull().sum()


# In[15]:


df['Message']=df['Message'].str.lower()


# In[16]:


dfr=df.drop_duplicates(inplace=True)


# In[17]:


from sklearn.preprocessing import LabelEncoder


# In[18]:


le=LabelEncoder()


# In[19]:


df['Category']=le.fit_transform(df['Category']) # change object values to numeric


# In[20]:


df.duplicated().sum() # checking for duplicates


# In[21]:


df7=df.drop_duplicates(inplace=True) # handled duplicates


# In[22]:


df.head()


# In[23]:


# data structure


# In[24]:


x=df['Message']
y=df['Category']


# In[25]:


from sklearn.feature_extraction.text import CountVectorizer


# In[26]:


v=CountVectorizer()


# # split data

# In[27]:


from sklearn.model_selection import  train_test_split


# In[28]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[29]:


x_train=v.fit_transform(x_train)
x_test=v.transform(x_test)      # please note, dont put fit transform on x_test, just use Transform(x-test)


# In[30]:


y_train.shape,y_test.shape,x_train.shape,x_test.shape


# # model building

# In[31]:


from sklearn.naive_bayes import MultinomialNB


# In[32]:


nb=MultinomialNB()


# In[33]:


nb.fit(x_train,y_train)


# In[34]:


nb.score(x_train,y_train)


# In[35]:


ypred=nb.predict(x_test)


# In[36]:


ypred


# In[37]:


tf=pd.DataFrame({'actual':y_test,
                 'predicted':ypred})


# In[38]:


tf


# # model evaluation

# In[39]:


from sklearn import metrics
from sklearn .metrics import confusion_matrix


# In[40]:


metrics.accuracy_score(y_test,ypred)


# In[41]:


gh=confusion_matrix(y_test,ypred)


# In[42]:


sns.heatmap(gh,annot=True)


# In[43]:


gh


# # validating the model performance

# In[44]:


from sklearn .model_selection import KFold,cross_val_score


# In[45]:


kf=KFold(n_splits=5)


# In[46]:


score=cross_val_score(nb,x_train,y_train,cv=kf)


# In[47]:


score


# In[48]:


# from the kfold validation , this model did so good overall


# In[49]:


from sklearn .model_selection import learning_curve


# In[50]:


train_sizes, train_scores, test_scores = learning_curve(nb, x_train, y_train, cv=5, n_jobs=-1)

plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training score')
plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', label='Cross-validation score')
plt.xlabel('Training size')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend()
plt.show()


# # model testing

# In[51]:


rt=['six chances to win cash! from 100 to 20,000 po...']


# In[52]:


rt=v.transform(rt)


# In[53]:


nb.predict(rt) # perfectly predicted


# In[54]:


bn=['i have a date on sunday with will!!',
    'urgent! you have won a 1 week free membership ...']


# In[55]:


bn=v.transform(bn)


# In[56]:


nb.predict(bn) #perfectly predicted


# # this model learned so well, i got a train score of 0.991 and test score of 0.98, 
# 

# In[57]:


import joblib


# In[60]:


model=joblib.dump(nb,'email_spam.joblib')


# In[ ]:




