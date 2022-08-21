#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Logistic Regression Project 
# 
# In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad
# 
# ## Import Libraries
# 
# **Import a few libraries you think you'll need (Or just import them as you go along!)**

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data
# **Read in the advertising.csv file and set it to a data frame called ad_data.**

# In[5]:


df=pd.read_csv("advertising.csv")


# **Check the head of ad_data**

# In[6]:


df.head()


# ** Use info and describe() on ad_data**

# In[7]:


df.info()


# In[8]:


df.describe()


# ## Exploratory Data Analysis
# 
# Let's use seaborn to explore the data!
# 
# Try recreating the plots shown below!
# 
# ** Create a histogram of the Age**

# In[12]:


df['Age'].hist(bins=30,alpha=0.7,edgecolor="black")


# ### create a pair plot for the age and area income

# In[16]:


sns.jointplot(x='Age',y='Area Income',data=df)


# **Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.**

# In[21]:


sns.jointplot(x='Age',y='Daily Time Spent on Site',data=df,kind="kde",space=0,color="r",fill=True)


# ** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**

# In[23]:


sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=df,color='g')


# ** Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**

# In[25]:


sns.pairplot(data=df,hue='Clicked on Ad')


# # Logistic Regression
# 
# Now it's time to do a train test split, and train our model!
# 
# You'll have the freedom here to choose columns that you want to train on!

# ** Split the data into training set and testing set using train_test_split**

# In[26]:


from sklearn.model_selection import train_test_split
X=df[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y=df['Clicked on Ad']
X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.3,random_state=101)


# ** Train and fit a logistic regression model on the training set.**

# In[28]:


from sklearn.linear_model import LogisticRegression


# In[32]:


model=LogisticRegression()
model.fit(X_train,y_train)


# ## Predictions and Evaluations
# ** Now predict values for the testing data.**

# In[33]:


pred=model.predict(X_test)


# In[34]:


pred


# ** Create a classification report for the model.**

# In[39]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# ## Great Job!
