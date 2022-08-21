#!/usr/bin/env python
# coding: utf-8

# ## Imports
# ** Import pandas, numpy, matplotlib,and seaborn. Then set %matplotlib inline 
# (You'll import sklearn as you need it.)**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data
# 
# We'll work with the Ecommerce Customers csv file from the company. It has Customer info, suchas Email, Address, and their color Avatar. Then it also has numerical value columns:
# 
# * Avg. Session Length: Average session of in-store style advice sessions.
# * Time on App: Average time spent on App in minutes
# * Time on Website: Average time spent on Website in minutes
# * Length of Membership: How many years the customer has been a member. 
# 
# ** Read in the Ecommerce Customers csv file as a DataFrame called customers.**

# In[4]:


customers = pd.read_csv('Ecommerce Customers')


# **Check the head of customers, and check out its info() and describe() methods.**

# In[6]:


customers.head()


# In[7]:


customers.describe()


# In[8]:


customers.info()


# ## Exploratory Data Analysis
# 
# **Let's explore the data!**
# 
# For the rest of the exercise we'll only be using the numerical data of the csv file.
# ___
# **Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?**

# In[10]:


sns.jointplot(x='Time on Website',y ='Yearly Amount Spent', data = customers)


# ** Do the same but with the Time on App column instead. **

# In[11]:


sns.jointplot(x='Time on App',y ='Yearly Amount Spent',data = customers)


# ** Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.**

# In[12]:


sns.jointplot(x='Time on App',y ='Length of Membership', data = customers, kind='hex')


# **Let's explore these types of relationships across the entire data set. Use [pairplot](https://stanford.edu/~mwaskom/software/seaborn/tutorial/axis_grids.html#plotting-pairwise-relationships-with-pairgrid-and-pairplot) to recreate the plot below.(Don't worry about the the colors)**

# In[13]:


sns.pairplot(customers)


# **Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?**

# In[14]:


print("Length of Membership")


# **Create a linear model plot (using seaborn's lmplot) of  Yearly Amount Spent vs. Length of Membership. **

# In[15]:


sns.set(color_codes=True)
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent',data=customers)


# ## Training and Testing Data
# 
# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.
# ** Set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column. **

# In[16]:


X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


# In[17]:


y= customers['Yearly Amount Spent']


# ** Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101**

# In[20]:


from sklearn.model_selection import train_test_split
train_X,test_X,train_Y,test_Y=train_test_split(X,y,test_size=0.3,random_state=101)


# ## Training the Model
# 
# Now its time to train our model on our training data!
# 
# ** Import LinearRegression from sklearn.linear_model **

# In[21]:


from sklearn.linear_model import LinearRegression


# **Create an instance of a LinearRegression() model named lm.**

# In[22]:


lm = LinearRegression()


# ** Train/fit lm on the training data.**

# In[23]:


lm.fit(train_X,train_Y )


# **Print out the coefficients of the model**

# In[24]:


print(lm.coef_)


# ## Predicting Test Data
# Now that we have fit our model, let's evaluate its performance by predicting off the test values!
# 
# ** Use lm.predict() to predict off the X_test set of the data.**

# In[25]:


predictions = lm.predict(test_X)


# ** Create a scatterplot of the real test values versus the predicted values. **

# In[27]:


plt.pyplot.scatter(test_Y, predictions)
plt.pyplot.ylabel('Predicted')
plt.pyplot.xlabel('Y test')


# ## Evaluating the Model
# 
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).
# 
# ** Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas**

# In[29]:


import sklearn.metrics as metrics
print('MAE: {}'.format(metrics.mean_absolute_error(test_Y, predictions)))
print('MSE: {}'.format(metrics.mean_squared_error(test_Y, predictions)))
print('RMSE: {}'.format(np.sqrt(metrics.mean_squared_error(test_Y, predictions))))


# ## Residuals
# 
# You should have gotten a very good model with a good fit. Let's quickly explore the residuals to make sure everything was okay with our data. 
# 
# **Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().**

# In[31]:


sns.distplot((test_Y-predictions))


# ## Conclusion
# We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important.  Let's see if we can interpret the coefficients at all to get an idea.
# 
# ** Recreate the dataframe below. **

# In[32]:


pd.DataFrame(lm.coef_ , X.columns, columns=['Coeffecient'])


# ** How can you interpret these coefficients? **

# The greater the value the more related it is to the target, in this case yearly amount spent

# **Do you think the company should focus more on their mobile app or on their website?**

# The company should focus on the mobile app
