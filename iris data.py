
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris


# In[2]:


iris=load_iris()


# In[3]:


iris


# In[4]:


print("keys {}".format(type(iris['data'])))


# In[5]:


print("{}".format(iris['target']))


# In[6]:


print("{}".format(iris['data'][:5]))


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train,X_test,y_train,y_test=train_test_split(iris['data'],iris['target'],random_state=0 )


# In[9]:


print("{}".format(X_train.shape))


# In[10]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[11]:


import numpy as np 


# In[12]:


iris_d=pd.DataFrame(X_train,columns=iris['feature_names'])


# In[13]:


iris_d


# In[14]:


sm=pd.plotting.scatter_matrix(iris_d)


# In[15]:


sx=pd.plotting.scatter_matrix(iris_d,c=y_train,figsize=(8,8),marker='o',hist_kwds={'bins':30},s=60,alpha=11.8)


# In[16]:


from sklearn.neighbors import KNeighborsClassifier


# In[17]:


knn=KNeighborsClassifier(n_neighbors=1)


# In[18]:


knn.fit(X_train,y_train)


# In[19]:


xnew=np.array([[5,2.9,2.5,1.9]])


# In[20]:


print(xnew.shape)


# In[21]:


xnew


# In[22]:


print(iris['target'][0])


# In[23]:


predict=knn.predict(xnew)


# In[24]:


print(iris['target_names'][predict])


# In[25]:


a=knn.predict(X_test)


# In[26]:


print(a)


# In[27]:


print(np.mean(a==y_test))


# In[28]:


print(knn.score(X_test,y_test))


# In[29]:


print(y_test)

