#!/usr/bin/env python
# coding: utf-8

# # K Nearest Neighbors Project 
# 
# Welcome to the KNN Project! This will be a simple project very similar to the lecture, except you'll be given another data set. Go ahead and just follow the directions below.
# ## Import Libraries
# **Import pandas,seaborn, and the usual libraries.**

# In[5]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data
# ** Read the 'KNN_Project_Data csv file into a dataframe **

# In[6]:


df=pd.read_csv("KNN_Project_Data")


# **Check the head of the dataframe.**

# In[7]:


df.head()


# # EDA
# 
# Since this data is artificial, we'll just do a large pairplot with seaborn.
# 
# **Use seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column.**

# In[8]:


sns.pairplot(data=df,hue='TARGET CLASS')


# # Standardize the Variables
# 
# Time to standardize the variables.
# 
# ** Import StandardScaler from Scikit learn.**

# In[9]:


from sklearn.preprocessing import StandardScaler


# ** Create a StandardScaler() object called scaler.**

# In[10]:


scaler=StandardScaler()


# ** Fit scaler to the features.**

# In[12]:


scaler.fit(X= df.drop('TARGET CLASS',axis = 1))


# **Use the .transform() method to transform the features to a scaled version.**

# In[13]:


X=scaler.transform(X=df.drop('TARGET CLASS',axis = 1))


# **Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**

# In[14]:


tra_df = pd.DataFrame(X, columns=df.columns[:-1])
tra_df.head()


# # Train Test Split
# 
# **Use train_test_split to split your data into a training set and a testing set.**

# In[15]:


from sklearn.model_selection import train_test_split


# In[17]:


target= df['TARGET CLASS']
train_X,test_X,train_Y,test_Y=train_test_split(X,target,test_size=0.3,random_state=101)


# # Using KNN
# 
# **Import KNeighborsClassifier from scikit learn.**

# In[19]:


from sklearn.neighbors import KNeighborsClassifier as KNN


# **Create a KNN model instance with n_neighbors=1**

# In[20]:


model= KNN(n_neighbors = 1)


# **Fit this KNN model to the training data.**

# In[21]:


model.fit(train_X,train_Y)


# # Predictions and Evaluations
# Let's evaluate our KNN model!

# **Use the predict method to predict values using your KNN model and X_test.**

# In[22]:


y_predict=model.predict(test_X)


# ** Create a confusion matrix and classification report.**

# In[23]:


from sklearn.metrics import confusion_matrix, classification_report


# In[24]:


print(confusion_matrix(test_Y,y_predict))


# In[25]:


print(classification_report(test_Y,y_predict))


# # Choosing a K Value
# Let's go ahead and use the elbow method to pick a good K Value!
# 
# ** Create a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list. Refer to the lecture if you are confused on this step.**

# In[26]:


err_rates = []
for idx in range(1,40):
    model=KNN(n_neighbors = idx)
    model.fit(train_X,train_Y)
    pred_idx=model.predict(test_X)
    err_rates.append(np.mean(test_Y!=pred_idx))


# **Now create the following plot using the information from your for loop.**

# In[45]:


plt.style.use('ggplot')
plt.subplots(figsize=(10,6))
plt.plot(range(1,40),err_rates,linestyle='--',color="blue",marker='o',markerfacecolor='red',markersize=10)
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.title('Error rate vs. k-value')


# ## Retrain with new K Value
# 
# **Retrain your model with the best K value (up to you to decide what you want) and re-do the classification report and the confusion matrix.**

# In[54]:


knn=KNN(n_neighbors=30)
knn.fit(train_X,train_Y)
pred=knn.predict(test_X)
print('WITH K=31')
print("")
print(classification_report(test_Y,pred))
print("Accuracy: ",knn.score(test_X,test_Y))

