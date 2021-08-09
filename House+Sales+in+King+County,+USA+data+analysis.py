
# coding: utf-8

# **Dataset Description**
# 
# This dataset contains house sale prices for King County, 
# 
# which includes Seattle. It includes homes sold between May 2014 and May 2015.

# In[1]:

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().magic('matplotlib inline')


# In[2]:

#read the dataset
df=pd.read_csv('kc_house_data.csv')
df.head(5)


# **assess dataset**

# In[3]:

df.shape


# In[4]:

#checking data types for columns
df.dtypes


# In[5]:

#checking missing value for columns
df.isnull().sum()


# In[6]:

#checking dublicated data
df.duplicated().sum()


# In[7]:

df['view'].nunique()


# In[8]:

df['view'].unique()


# In[9]:

#statistical summary
df.describe()


# **clean dataset**
# 
# 1- There are some datatypes of some columns that should be changed like (date , yr_built , yr_removed)
# 
# to datetime type , and (bathrooms , floors) to int.
# 
# 
# 

# In[10]:

#change datatype for column (date) object------>Datetime

#code
df['date']=pd.to_datetime(df['date'])

#test
df['date'].head(3)


# In[11]:

#change datatype for column (yr_built , yr_removed) int------>Datetime

#code
df['yr_built']=pd.to_datetime(df['yr_built'])

#code
df['yr_renovated']=pd.to_datetime(df['yr_renovated'])


# In[12]:

#change datatype for column (bathrooms , floors) float------>int

#code
df['bathrooms']=df['bathrooms'].astype(int)

#code
df['floors']=df['floors'].astype(int)

#test
df.dtypes


# In[13]:

#drop column id

#code
df=df.drop(['id'],axis=1)


# In[14]:

#test
df.head(1)


# **Perform EDA (Exploratory data analysis)**

# In[15]:

sdf=df[['bedrooms','bathrooms','floors','grade','waterfront']]

sdf.hist(figsize=(12,12))


# In[18]:

#The percentage of houses which have waterfront.
df.groupby('waterfront').size().plot(kind='pie',autopct='%.2f%%',labels = ['0','1'],figsize=(7.5,7.5))


# In[19]:

df.groupby('floors').size().plot(kind = 'pie' , title = 'Percentage of floors numbers' , autopct = "%.2f%%" , labels = set(df.floors) , figsize =(7.5,7.5) , fontsize=15)


# 
# **Conclusion**
# 
# 1- Waterfront : Most houses don't have waterfront.
#     
# 
# 2- Grades : Most houses have garde between (7 or 8) , and the other houses their grades values are low.
# 
# 3- Floors : More than 55% of houses contains only one floor , about 40% of houses contains two floors and a very few percentage of house that have 3 floors.
# 
# 4- Bathrooms : More than 10000 houses have 2 bathroooms, More than 8000 houses have 1 bathroooms, almost 2100 houses  have 3 bathrooms and very few houses have more than 3 bathrooms.
# 
# 5-Bedrooms :most houses have bedrooms betwwen(1 to 5) and few houses have more than 5 bedrooms.
# 

# In[59]:

#relationship between price & Bedrooms
print(df[['price','bedrooms']].corr())

plt.scatter(data=df,x='bedrooms',y='price')
plt.show()


# Conclusion : correlation value=0.3 , the relationship between number of bedrooms and price of houses is weak.

# In[60]:

#relationship between price & Bedrooms
print(df[['price','bathrooms']].corr())

plt.scatter(data=df,x='bathrooms',y='price')
plt.show()


# Conclusion : correlation value=0.5 , the relationship between number of bathrooms and price of houses is moderate

# In[62]:

#The relationship between sqft_living and price

print(df[['price','sqft_living']].corr())

plt.scatter(data=df,x='sqft_living',y='price')
plt.show()


# Conclusion :correlation value=0.7 , the relationship between number of bathrooms and price of houses is strong somehow, it means that any change in the value of the square footage affects on the price of house.

# In[64]:

#The relationship between Longitude coordinate and price
print(df[['price','long']].corr())

plt.scatter(data=df,x='long',y='price')
plt.show()


# Conclusion : correlation value=0.02 , the relationship between number of long and price of houses is very very week

# **copy the data in new dataframe**

# In[28]:

df_1=df.copy()


# **some predictive analysis**

# In[20]:

from sklearn.linear_model import LinearRegression
line_fitter=LinearRegression()


# In[29]:

x=df_1[['sqft_living']]
y=df_1[['price']]
line_fitter.fit(x,y)
print("The price value increases by =" , line_fitter.coef_ , "when the square footage increases by 1")
print("When sqft_living = 0 , the price value decreases by" , line_fitter.intercept_)


# **predict the price based on sqft_living  , bedrooms  , bathrooms  , floors ,  grade  , waterfront**

# In[33]:

x=df[['sqft_living','bedrooms','bathrooms','floors','grade','waterfront']]
y=df[['price']]
line_fitter.fit(x,y)
y_predicted=line_fitter.predict(x)


# In[34]:

print(y_predicted)


# In[35]:

#add predicted_price column to dataframe
df_1['predicted_price']=y_predicted


# In[36]:

df_1.head(50)


# # E    N     D
