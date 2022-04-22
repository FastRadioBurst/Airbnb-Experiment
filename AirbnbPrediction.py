#!/usr/bin/env python
# coding: utf-8

# ## <b> -------------------------------------Regression: Airbnb price prediction---------------------------------- 

# Source: https://www.kaggle.com/rudymizrahi/airbnb-listings-in-major-us-cities-deloitte-ml
#         
# Train dataset has been taken from this link which contains 74111 rows and 27 columns. We have randomly selected 4692 rows as our main dataset. 

##for showing all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import warnings
warnings.filterwarnings("ignore")

#Importing basic libraries used throughout
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Reading the datset stored in the csv file format.
ab = pd.read_csv(r'airbnb.csv')
ab.head()

#Viewing the summary of dataset for understanding the data we are dealing with.
ab.info()
ab.describe()
ab.describe(include='O')

#Making a copy of the orignal dataset
airbnb=ab.copy()

#Visualizing the null values in the dataset. As can be noticed from the blue dash lines, lot of null values in few columns,
#which will be treated consequently.
sns.heatmap(airbnb.isnull(), yticklabels = False, cbar = False, cmap="Blues")

#Our target variable is log_price. After converting the target variable by taking the inverse of log, 
#the variable is highly right skewed. So, keeping the target variable as it is because of normal distribution pattern.
plt.figure(figsize=[6,12])
plt.subplot(211)
np.exp(airbnb['log_price']).hist()
plt.subplot(212)
airbnb['log_price'].hist()

#We cannot make assumption from our side whether the host is verified or not, hence dropping the null values in the column.
airbnb.dropna(subset=['host_identity_verified'], inplace=True)
airbnb.info()


# ### Before data processing and data imputation, we will split the data into train and test datset.
#Target Label
y= airbnb['log_price']

#Independent attributes
X = airbnb.iloc[:,2:]
X1 = airbnb.iloc[:,:1]


#Merging X1 and X
X = pd.concat([X1, X], axis=1, sort=False)
#X.info()

#Splitting into train and test set
from sklearn.model_selection import train_test_split
X_train_org, X_test_org, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.2)

#Checking the size of the each split datasets
X_train_org.shape, y_train.shape, X_test_org.shape, y_test.shape

#making a copy of the X_train_org, so that we can process and prepare the data before running the models.
airbnb = X_train_org.copy()
airbnb.info()


# ## Processing the columns, treating Missing values, and dropping irrelavent data. This will done for train set first and then test set. 
# 

#Property column has many categories. Following is the distribution of the Property column
airbnb.groupby('property_type')['id'].count()


# #As most of the instances fall under Apartment, House category. So creating 3 categories for the Property_type column
def conv(airbnb):
    if airbnb['property_type'] in ['Apartment', 'House', 'Other']:
        return airbnb['property_type']
    else:
        return 'Other'

airbnb['property_type'] = airbnb.apply(lambda airbnb:conv(airbnb), axis = 1)
airbnb['property_type'].unique()

#Room_type column
airbnb.groupby('room_type')['id'].count()

#Amenities column has text type data. Hence dropping the column. 
airbnb.drop(['amenities'], axis=1, inplace=True)

#Accomodates column. As it has discreet values, creating bins and converting the column to categorical variable.
airbnb.groupby('accommodates')['id'].count()


def conv(airbnb):
    if airbnb['accommodates']==1:
        return '1'
    elif airbnb['accommodates']==2:
        return '2'
    elif airbnb['accommodates'] in (3,4):
        return '3-4'
    else:
        return 'Greater than 4'


airbnb['accommodates'] = airbnb.apply(lambda airbnb:conv(airbnb), axis = 1)
airbnb.groupby('accommodates')['id'].count()

#Bathrooms column. As it has discreet values, creating bins and converting the column to categorical variable.
airbnb.groupby('bathrooms')['id'].count()

#As the bathroom columns has missing values, Using accomodates and bathrooms column and using that distribution 
# for imputing in the bathrooms column. 
airbnb.groupby('accommodates')['bathrooms'].median()

#Using median, as it gives exact number rather than mean, because 1.15 bathroom won't mean anything.

def impute(airbnb):
    if airbnb['accommodates'] in (['1','2','3-4']):
        return 1
    else:
        return 2

airbnb['bathrooms'] = airbnb.apply(lambda airbnb:impute(airbnb), axis = 1)
airbnb.groupby('bathrooms')['id'].count()

#Similarly for bed_type column
airbnb.groupby('bed_type')['id'].count()

def conv(airbnb):
    if airbnb['bed_type'] in ['Real Bed']:
        return airbnb['bed_type']
    else:
        return 'Other'

airbnb['bed_type'] = airbnb.apply(lambda airbnb:conv(airbnb), axis = 1)
airbnb.groupby('bed_type')['id'].count()

#Reducing categories for cancellation_policy column
airbnb.groupby('cancellation_policy')['id'].count()

def conv(airbnb):
    if airbnb['cancellation_policy'] in ['flexible', 'strict', 'moderate']:
        return airbnb['cancellation_policy']
    else:
        return 'strict'

airbnb['cancellation_policy'] = airbnb.apply(lambda airbnb:conv(airbnb), axis = 1)
airbnb.groupby('cancellation_policy')['id'].count()

#Checking cleaning_fee column
airbnb.groupby('cleaning_fee')['id'].count()

#Distribution of City column
airbnb.groupby('city')['id'].count()

#First_review and Last_review column has lot of missing values. Dealing with them. 
#Converting them to datetime format.
airbnb['first_review'] = pd.to_datetime(airbnb['first_review'])
airbnb['last_review'] = pd.to_datetime(airbnb['last_review'])

#Creating a new variable to understand the difference between first and last reviews
airbnb['Diff_fnl_review']= airbnb['last_review'].dt.year - airbnb['first_review'].dt.year

airbnb['Diff_fnl_review'].unique()

airbnb['Diff_fnl_review'].hist(bins=40)

airbnb.groupby('Diff_fnl_review')['id'].count()


# #As most of the instances fall under 0 category, that is first review and last review are in same year.
# #Even if we impute the first and last review with host_since column(wherever NaN value), it will not make much sense. 
#Dropping these columns
airbnb.drop(['first_review','last_review', 'Diff_fnl_review'],axis=1, inplace=True)
airbnb.info()

#Distribution of 'host_has_profile_pic' column
airbnb.groupby('host_has_profile_pic')['id'].count()

#As most of the lisitng has profile pic of the host, this attribute will not make much difference, so dropping it
airbnb.drop(['host_has_profile_pic'],axis=1, inplace=True)

#Distribution of 'host_identity_verified' column
airbnb.groupby('host_identity_verified')['id'].count()

#Converting to datetime format
airbnb['host_since']=pd.to_datetime(airbnb['host_since'])

#Checking the latest date
airbnb['host_since'].max()


# #### Creating a new variable in number of years for time the host has started on airbnb 
airbnb['host_since_timeinyear']=2017 - airbnb['host_since'].dt.year
airbnb['host_since_timeinyear'].hist()

airbnb.groupby('host_since_timeinyear')['id'].count()

#Converting  host_since_timeinyear into 3 categories
def hosttime(airbnb):
    if airbnb['host_since_timeinyear'] in [0]:
        return 'New_Host'
    elif airbnb['host_since_timeinyear'] in [1,2]:
        return '1-2 yrs'
    elif airbnb['host_since_timeinyear'] in [3,4]:
        return '3-4 yrs'
    else:
        return 'Greater than 4 yrs'

airbnb['host_since_timeinyear'] = airbnb.apply(lambda airbnb:hosttime(airbnb), axis = 1)
airbnb.groupby('host_since_timeinyear')['id'].count()

#Dropping the original column
airbnb.drop(['host_since'], axis=1, inplace=True)

#Distribution of 'instant_bookable' column
airbnb.groupby('instant_bookable')['id'].count()

#Keeping the data at granular level to city, and dropping all the geo variables including the name.
airbnb.drop(['latitude', 'longitude', 'name', 'neighbourhood','zipcode'], axis=1, inplace=True)

#Again visualizing the null values
sns.heatmap(airbnb.isnull(), yticklabels = False, cbar = False, cmap="Blues")

#Distribution of number of reviews as per city
airbnb.groupby('city')['number_of_reviews'].sum()

#Removing the % sign from the column and converting to float category
airbnb['host_response_rate'] = airbnb['host_response_rate'].str.replace(r'\D', '')
airbnb['host_response_rate'] = airbnb['host_response_rate'].astype(float)
airbnb['host_response_rate'].hist()

airbnb['host_response_rate'].describe()
airbnb.groupby('host_response_rate')['id'].count()

#Replacing the host_response_rate with mean i.e 95 to be in the safe side. Eventhough median tends towards 100.
airbnb['host_response_rate']= airbnb['host_response_rate'].fillna(value=95.0)


#Distribution of 'review_scores_rating' column
airbnb.groupby('review_scores_rating')['id'].count()

airbnb['review_scores_rating'].describe()
airbnb['review_scores_rating'].hist()

#Replacing the missing values with median=96
airbnb['review_scores_rating']= airbnb['review_scores_rating'].fillna(value=96.0)

#Visualizing null values left
sns.heatmap(airbnb.isnull(), yticklabels = False, cbar = False, cmap="Blues")

#Distribution of beds column. And imputing for null values
airbnb['beds'].describe()
airbnb.groupby('bedrooms')['beds'].mean()
airbnb.info()
airbnb.groupby('bedrooms')['beds'].count()

#Creating a subset of the maindatset to understhan the distribution
d = {'property_type':airbnb['property_type'], 'room_type':airbnb['room_type'],'bedrooms':airbnb['bedrooms'],'beds':airbnb['beds']}
a= pd.DataFrame(data = d)
a.head()

#Creating a subset for where only null values of bedrooms
df1 = a[a['bedrooms'].isnull()]
df1
pd.set_option('display.max_rows', None)

#As per above, substituting bedrooms as 1 where missing value in the column, because 1 number of beds will be there in 1 bedroom
airbnb['bedrooms'] = airbnb['bedrooms'].fillna(value=1.0)

#Similarly creating the subset for beds
df2 = a[a['beds'].isnull()]
df2['bedrooms'].unique()
df2.info()

airbnb.groupby('bedrooms')['beds'].median()

#As per the above, we can see that bedrooms 0,1,2,3,4,5 has 1,1,2,3,4,6 beds as median respectively. 
#So, for beds column, we are replacing null values with respective bedrooms number

#Creating a copy as a checkpoint
airbnb1 = airbnb.copy()
airbnb1.info()


# Function for imputing the number of beds and then applying it. Doing it step by step.
def beds(airbnb1):
    if np.isnan(airbnb1['beds']) and airbnb1['bedrooms']==0:
        return 1
    else:
        return airbnb1['beds']

airbnb1['beds'] = airbnb1.apply(lambda airbnb1:beds(airbnb1), axis = 1)

def beds(airbnb1):
    if np.isnan(airbnb1['beds']) and airbnb1['bedrooms']==1:
        return 1
    else:
        return airbnb1['beds']

airbnb1['beds'] = airbnb1.apply(lambda airbnb1:beds(airbnb1), axis = 1)


def beds(airbnb1):
    if np.isnan(airbnb1['beds']) and airbnb1['bedrooms']==2:
        return 2
    else:
        return airbnb1['beds']


airbnb1['beds'] = airbnb1.apply(lambda airbnb1:beds(airbnb1), axis = 1)


# In[89]:


def beds(airbnb1):
    if np.isnan(airbnb1['beds']) and airbnb1['bedrooms']==3:
        return 3
    else:
        return airbnb1['beds']


# In[90]:


airbnb1['beds'] = airbnb1.apply(lambda airbnb1:beds(airbnb1), axis = 1)


# In[91]:


def beds(airbnb1):
    if np.isnan(airbnb1['beds']) and airbnb1['bedrooms']==4:
        return 4
    else:
        return airbnb1['beds']


# In[92]:


airbnb1['beds'] = airbnb1.apply(lambda airbnb1:beds(airbnb1), axis = 1)


# In[93]:


def beds(airbnb1):
    if np.isnan(airbnb1['beds']) and airbnb1['bedrooms']==5:
        return 6
    else:
        return airbnb1['beds']


# In[94]:


airbnb1['beds'] = airbnb1.apply(lambda airbnb1:beds(airbnb1), axis = 1)


# In[95]:


#Checking for null values
sns.heatmap(airbnb1.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# All null values are dealt with.

# In[96]:


airbnb = airbnb1.copy()


# In[97]:


airbnb.info()


# ## Now encoding the columns  and converting it into numerical form for running the model

# In[98]:


airbnb.drop(['id'], axis=1, inplace= True)
airbnb.head()


# In[99]:


airbnb['property_type'].unique() #encoding
airbnb['room_type'].unique() #encoding
airbnb['accommodates'].unique() #mapping
airbnb['bathrooms'].unique() #mapping
airbnb['bed_type'].unique() #mapping
airbnb['cancellation_policy'].unique() #mapping


# In[100]:


airbnb['cleaning_fee'].unique() #boolen no change needed
airbnb['city'].unique() # encoding
airbnb['host_identity_verified'].unique() #mapping
airbnb['host_response_rate'].unique() #convert % to 0.1 or something
airbnb['instant_bookable'].unique() #mapping
airbnb['number_of_reviews'].unique() #continous variable
airbnb['review_scores_rating'].unique() #convert % to number
airbnb['bedrooms'].unique() #mapping
airbnb['beds'].unique() # beds are ordinal mapping needed
airbnb['host_since_timeinyear'].unique() #mapping ordinal


# In[101]:


#Converting the columns in % form
airbnb['host_response_rate']=airbnb['host_response_rate']/100.00
airbnb['review_scores_rating']=airbnb['review_scores_rating']/100.00


# # One hot encoding for columns "property_type" & "room_type"

# In[102]:


airbnb =pd.get_dummies(airbnb,columns=['property_type','room_type','city'], prefix=['property_type','room_type','city'])


# # Mappping for respective columns

# In[103]:


airbnb['accommodates'].unique() #mapping
airbnb['accommodates'] = airbnb['accommodates'].map( {'1': 0, '2': 1, '3-4': 2, 'Greater than 4': 3} ).astype(int)
airbnb['bathrooms'].unique() #mapping
airbnb['bathrooms'] = airbnb['bathrooms'].map( {1: 0, 2: 1} ).astype(int)
airbnb['bed_type'].unique() #mapping
airbnb['bed_type'] = airbnb['bed_type'].map( {'Real Bed': 1, 'Other': 0} ).astype(int)
airbnb['cancellation_policy'].unique() #mapping
airbnb['cancellation_policy'] = airbnb['cancellation_policy'].map( {'flexible': 0, 'moderate': 1, 'strict':2} ).astype(int)
airbnb['instant_bookable'].unique() #mapping
airbnb['instant_bookable'] = airbnb['instant_bookable'].map( {'t': 1, 'f': 0} ).astype(int)
airbnb['host_since_timeinyear'].unique() #mapping
airbnb['host_since_timeinyear'] = airbnb['host_since_timeinyear'].map( {'New_Host': 0, '1-2 yrs': 1, '3-4 yrs': 2, 'Greater than 4 yrs': 3} ).astype(int)
airbnb['host_identity_verified'].unique() #mapping
airbnb['host_identity_verified'] = airbnb['host_identity_verified'].map( {'t': 1, 'f': 0} ).astype(int)
airbnb['bedrooms'].unique() #mapping
airbnb['bedrooms'] = airbnb['bedrooms'].map( {0: 0, 1: 1, 2: 2, 3: 3, 4: 4,5: 5, 6: 6, 7: 7, 8: 8, 9: 9} ).astype(int)


# In[104]:


airbnb.info()


# # The train set is prepared. Now all the above code is run for test dataset 
# # to be on the same page while testing our models. 

# In[105]:


#Creating the copy for test set
airbnb_test=X_test_org.copy()


# In[106]:


#Property column has many categories. Following is the distribution of the Property column
airbnb_test.groupby('property_type')['id'].count()


# #As most of the instances fall under Apartment, House category. So creating 3 categories for the Property_type column

# In[107]:


#Defining the function
def conv(airbnb_test):
    if airbnb_test['property_type'] in ['Apartment', 'House', 'Other']:
        return airbnb_test['property_type']
    else:
        return 'Other'


# In[108]:


#Applying the function
airbnb_test['property_type'] = airbnb_test.apply(lambda airbnb_test:conv(airbnb_test), axis = 1)
airbnb_test['property_type'].unique()


# In[109]:


#Room_type column
airbnb_test.groupby('room_type')['id'].count()


# In[110]:


#Amenities column has text type data. Hence dropping the column. 
airbnb_test.drop(['amenities'], axis=1, inplace=True)


# In[111]:


#Accomodates column. As it has discreet values, creating bins and converting the column to categorical variable.
airbnb_test.groupby('accommodates')['id'].count()


# In[112]:


#Defining the function
def conv(airbnb_test):
    if airbnb_test['accommodates']==1:
        return '1'
    elif airbnb_test['accommodates']==2:
        return '2'
    elif airbnb_test['accommodates'] in (3,4):
        return '3-4'
    else:
        return 'Greater than 4'


# In[113]:


#Applying the function
airbnb_test['accommodates'] = airbnb_test.apply(lambda airbnb_test:conv(airbnb_test), axis = 1)
airbnb_test.groupby('accommodates')['id'].count()


# In[114]:


#Bathrooms column. As it has discreet values, creating bins and converting the column to categorical variable.


# In[115]:


airbnb_test.groupby('bathrooms')['id'].count()


# In[116]:


#As the bathroom columns has missing values, Using accomodates and bathrooms column and using that distribution 
# for imputing in the bathrooms column. 
airbnb_test.groupby('accommodates')['bathrooms'].median()


# In[117]:


#Using median, as it gives exact number rather than mean, because 1.15 bathroom won't mean anything.


# In[118]:


def impute(airbnb_test):
    if airbnb_test['accommodates'] in (['1','2','3-4']):
        return 1
    else:
        return 2


# In[119]:


airbnb_test['bathrooms'] = airbnb_test.apply(lambda airbnb_test:impute(airbnb_test), axis = 1)
airbnb_test.groupby('bathrooms')['id'].count()


# In[120]:


#Similarly for bed_type column


# In[121]:


airbnb_test.groupby('bed_type')['id'].count()


# In[122]:


def conv(airbnb_test):
    if airbnb_test['bed_type'] in ['Real Bed']:
        return airbnb_test['bed_type']
    else:
        return 'Other'


# In[123]:


airbnb_test['bed_type'] = airbnb_test.apply(lambda airbnb_test:conv(airbnb_test), axis = 1)
airbnb_test.groupby('bed_type')['id'].count()


# In[124]:


#Reducing categories for cancellation_policy column


# In[125]:


airbnb_test.groupby('cancellation_policy')['id'].count()


# In[126]:


def conv(airbnb_test):
    if airbnb_test['cancellation_policy'] in ['flexible', 'strict', 'moderate']:
        return airbnb_test['cancellation_policy']
    else:
        return 'strict'


# In[127]:


airbnb_test['cancellation_policy'] = airbnb_test.apply(lambda airbnb_test:conv(airbnb_test), axis = 1)
airbnb_test.groupby('cancellation_policy')['id'].count()


# In[128]:


#Checking cleaning_fee column
airbnb_test.groupby('cleaning_fee')['id'].count()


# In[129]:


#Distribution of City column
airbnb_test.groupby('city')['id'].count()


# In[130]:


#First_review and Last_review column has lot of missing values. Dealing with them. 
#Converting them to datetime format.


# In[131]:


airbnb_test['first_review'] = pd.to_datetime(airbnb_test['first_review'])


# In[132]:


airbnb_test['last_review'] = pd.to_datetime(airbnb_test['last_review'])


# In[133]:


#Creating a new variable to understand the difference between first and last reviews
airbnb_test['Diff_fnl_review']= airbnb_test['last_review'].dt.year - airbnb_test['first_review'].dt.year


# In[134]:


airbnb_test['Diff_fnl_review'].unique()


# In[135]:


airbnb_test['Diff_fnl_review'].hist(bins=40)


# In[136]:


airbnb_test.groupby('Diff_fnl_review')['id'].count()


# #As most of the instances fall under 0 category, that is first review and last review are in same year.
# #Even if we impute the first and last review with host_since column(wherever NaN value), it will not make much sense. 
# #Hence removing these columns

# In[137]:


#Dropping these columns
airbnb_test.drop(['first_review','last_review', 'Diff_fnl_review'],axis=1, inplace=True)
airbnb_test.info()


# In[138]:


#Distribution of 'host_has_profile_pic' column
airbnb_test.groupby('host_has_profile_pic')['id'].count()


# In[139]:


#As most of the lisitng has profile pic of the host, this attribute will not make much difference, so dropping it


# In[140]:


#Dropping the column
airbnb_test.drop(['host_has_profile_pic'],axis=1, inplace=True)


# In[141]:


#Distribution of 'host_identity_verified' column
airbnb_test.groupby('host_identity_verified')['id'].count()


# In[142]:


#Converting to datetime format
airbnb_test['host_since']=pd.to_datetime(airbnb_test['host_since'])


# In[143]:


#Checking the latest date
airbnb_test['host_since'].max()


# #### Creating a new variable in number of years for time the host has started on airbnb_test 

# In[144]:


airbnb_test['host_since_timeinyear']=2017 - airbnb_test['host_since'].dt.year


# In[145]:


airbnb_test['host_since_timeinyear'].hist()


# In[146]:


airbnb_test.groupby('host_since_timeinyear')['id'].count()


# In[147]:


#Converting  host_since_timeinyear into 3 categories


# In[148]:


#Creating the function
def hosttime(airbnb_test):
    if airbnb_test['host_since_timeinyear'] in [0]:
        return 'New_Host'
    elif airbnb_test['host_since_timeinyear'] in [1,2]:
        return '1-2 yrs'
    elif airbnb_test['host_since_timeinyear'] in [3,4]:
        return '3-4 yrs'
    else:
        return 'Greater than 4 yrs'


# In[149]:


#Applying the funtion
airbnb_test['host_since_timeinyear'] = airbnb_test.apply(lambda airbnb_test:hosttime(airbnb_test), axis = 1)
airbnb_test.groupby('host_since_timeinyear')['id'].count()


# In[150]:


#Dropping the original column
airbnb_test.drop(['host_since'], axis=1, inplace=True)


# In[151]:


#Distribution of 'instant_bookable' column
airbnb_test.groupby('instant_bookable')['id'].count()


# In[152]:


#Keeping the data at granular level to city, and dropping all the geo variables including the name.
airbnb_test.drop(['latitude', 'longitude', 'name', 'neighbourhood','zipcode'], axis=1, inplace=True)


# In[153]:


#Again visualizing the null values
sns.heatmap(airbnb_test.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# In[154]:


#Distribution of number of reviews as per city
airbnb_test.groupby('city')['number_of_reviews'].sum()


# In[155]:


#Removing the % sign from the column and converting to float category
airbnb_test['host_response_rate'] = airbnb_test['host_response_rate'].str.replace(r'\D', '')
airbnb_test['host_response_rate'] = airbnb_test['host_response_rate'].astype(float)
airbnb_test['host_response_rate'].hist()


# In[156]:


airbnb_test['host_response_rate'].describe()


# In[157]:


airbnb_test.groupby('host_response_rate')['id'].count()


# In[158]:


#Replacing the host_response_rate with mean i.e 95 to be in the safe side. Eventhough median tends towards 100.
airbnb_test['host_response_rate']= airbnb_test['host_response_rate'].fillna(value=95.0)


# In[159]:


#Distribution of 'review_scores_rating' column
airbnb_test.groupby('review_scores_rating')['id'].count()


# In[160]:


airbnb_test['review_scores_rating'].describe()
airbnb_test['review_scores_rating'].hist()


# In[161]:


#Replacing the missing values with median=96
airbnb_test['review_scores_rating']= airbnb_test['review_scores_rating'].fillna(value=96.0)


# In[162]:


#Visualizing null values left
sns.heatmap(airbnb_test.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# In[163]:


#Distribution of beds column. And imputing for null values
airbnb_test['beds'].describe()
airbnb_test.groupby('bedrooms')['beds'].mean()


# In[164]:


airbnb_test.info()


# In[165]:


airbnb_test.groupby('bedrooms')['beds'].count()


# In[166]:


#Creating a subset of the maindatset to understhan the distribution
d = {'property_type':airbnb_test['property_type'], 'room_type':airbnb_test['room_type'],'bedrooms':airbnb_test['bedrooms'],'beds':airbnb_test['beds']}
a= pd.DataFrame(data = d)
a.head()


# In[167]:


#Creating a subset for where only null values of bedrooms
df1 = a[a['bedrooms'].isnull()]
df1
pd.set_option('display.max_rows', None)


# In[ ]:





# In[168]:


#As per above, substituting bedrooms as 1 where missing value in the column, because 1 number of beds will be there in 1 bedroom
airbnb_test['bedrooms'] = airbnb_test['bedrooms'].fillna(value=1.0)


# In[169]:


#Similarly creating the subset for beds
df2 = a[a['beds'].isnull()]
df2['bedrooms'].unique()
df2.info()


# In[170]:


airbnb_test.groupby('bedrooms')['beds'].median()


# In[171]:


#As per the above, we can see that bedrooms 0,1,2,3,4,5 has 1,1,2,3,4,6 beds as median respectively. 
#So, for beds column, we are replacing null values with respective bedrooms number
airbnb_test1 = airbnb_test.copy()
airbnb_test1.info()


# Creating function for imputing the number of beds and then applying it. Doing it step by step.

def beds(airbnb_test1):
    if np.isnan(airbnb_test1['beds']) and airbnb_test1['bedrooms']==0:
        return 1
    else:
        return airbnb_test1['beds']

airbnb_test1['beds'] = airbnb_test1.apply(lambda airbnb_test1:beds(airbnb_test1), axis = 1)

def beds(airbnb_test1):
    if np.isnan(airbnb_test1['beds']) and airbnb_test1['bedrooms']==1:
        return 1
    else:
        return airbnb_test1['beds']

airbnb_test1['beds'] = airbnb_test1.apply(lambda airbnb_test1:beds(airbnb_test1), axis = 1)

def beds(airbnb_test1):
    if np.isnan(airbnb_test1['beds']) and airbnb_test1['bedrooms']==2:
        return 2
    else:
        return airbnb_test1['beds']


airbnb_test1['beds'] = airbnb_test1.apply(lambda airbnb_test1:beds(airbnb_test1), axis = 1)

def beds(airbnb_test1):
    if np.isnan(airbnb_test1['beds']) and airbnb_test1['bedrooms']==3:
        return 3
    else:
        return airbnb_test1['beds']

airbnb_test1['beds'] = airbnb_test1.apply(lambda airbnb_test1:beds(airbnb_test1), axis = 1)

def beds(airbnb_test1):
    if np.isnan(airbnb_test1['beds']) and airbnb_test1['bedrooms']==4:
        return 4
    else:
        return airbnb_test1['beds']

airbnb_test1['beds'] = airbnb_test1.apply(lambda airbnb_test1:beds(airbnb_test1), axis = 1)

def beds(airbnb_test1):
    if np.isnan(airbnb_test1['beds']) and airbnb_test1['bedrooms']==5:
        return 6
    else:
        return airbnb_test1['beds']

airbnb_test1['beds'] = airbnb_test1.apply(lambda airbnb_test1:beds(airbnb_test1), axis = 1)

#Checking for null values
sns.heatmap(airbnb_test1.isnull(), yticklabels = False, cbar = False, cmap="Blues")

airbnb_test = airbnb_test1.copy()
airbnb_test.info()


# ## Now encoding the columns  and converting it into numerical form for running the model
airbnb_test.drop(['id'], axis=1, inplace= True)
airbnb_test.head()
airbnb_test['property_type'].unique() #encoding
airbnb_test['room_type'].unique() #encoding
airbnb_test['accommodates'].unique() #mapping
airbnb_test['bathrooms'].unique() #mapping
airbnb_test['bed_type'].unique() #mapping
airbnb_test['cancellation_policy'].unique() #mapping
airbnb_test['cleaning_fee'].unique() #boolen no change needed
airbnb_test['city'].unique() # encoding
airbnb_test['host_identity_verified'].unique() #mapping
airbnb_test['host_response_rate'].unique() #convert % to 0.1 or something
airbnb_test['instant_bookable'].unique() #mapping
airbnb_test['number_of_reviews'].unique() #continous variable
airbnb_test['review_scores_rating'].unique() #convert % to number
airbnb_test['bedrooms'].unique() #mapping
airbnb_test['beds'].unique() # beds are ordinal mapping needed
airbnb_test['host_since_timeinyear'].unique() #mapping ordinal

#Converting the columns in % form
airbnb_test['host_response_rate']=airbnb_test['host_response_rate']/100.00
airbnb_test['review_scores_rating']=airbnb_test['review_scores_rating']/100.00


# # One hot encoding for columns "property_type" & "room_type"
airbnb_test =pd.get_dummies(airbnb_test,columns=['property_type','room_type','city'], prefix=['property_type','room_type','city'])


# # Mappping for respective columns
airbnb_test['accommodates'].unique() #mapping
airbnb_test['accommodates'] = airbnb_test['accommodates'].map( {'1': 0, '2': 1, '3-4': 2, 'Greater than 4': 3} ).astype(int)
airbnb_test['bathrooms'].unique() #mapping
airbnb_test['bathrooms'] = airbnb_test['bathrooms'].map( {1: 0, 2: 1} ).astype(int)
airbnb_test['bed_type'].unique() #mapping
airbnb_test['bed_type'] = airbnb_test['bed_type'].map( {'Real Bed': 1, 'Other': 0} ).astype(int)
airbnb_test['cancellation_policy'].unique() #mapping
airbnb_test['cancellation_policy'] = airbnb_test['cancellation_policy'].map( {'flexible': 0, 'moderate': 1, 'strict':2} ).astype(int)
airbnb_test['instant_bookable'].unique() #mapping
airbnb_test['instant_bookable'] = airbnb_test['instant_bookable'].map( {'t': 1, 'f': 0} ).astype(int)
airbnb_test['host_since_timeinyear'].unique() #mapping
airbnb_test['host_since_timeinyear'] = airbnb_test['host_since_timeinyear'].map( {'New_Host': 0, '1-2 yrs': 1, '3-4 yrs': 2, 'Greater than 4 yrs': 3} ).astype(int)
airbnb_test['host_identity_verified'].unique() #mapping
airbnb_test['host_identity_verified'] = airbnb_test['host_identity_verified'].map( {'t': 1, 'f': 0} ).astype(int)
airbnb_test['bedrooms'].unique() #mapping
airbnb_test['bedrooms'] = airbnb_test['bedrooms'].map( {0: 0, 1: 1, 2: 2, 3: 3, 4: 4,5: 5, 6: 6, 7: 7, 8: 8, 9: 9} ).astype(int)

airbnb_test.info()


# ### Applying MinMaxScaling techniques produces values of range [0,1]. Our dataset has features with hard boundaries.  It does not have outliers
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(airbnb)
X_test = scaler.transform(airbnb_test)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# # Regression by Bagging and Pasting
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

#Cross-Validation
scores = cross_val_score(KNeighborsRegressor(n_neighbors=19), X_train, y_train,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))
scores = cross_val_score(KNeighborsRegressor(n_neighbors=19), X_test, y_test,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))

# accuracy for best fit KNNRegressor
knn = KNeighborsRegressor(n_neighbors=19)
knn.fit(X_train, y_train)
print(knn.score(X_train, y_train))
print(knn.score(X_test, y_test))


# RMSE for both train and test is 0.5

# ## KNN with Bagging

# Grid search for KNN for bagging hyperparameters

grid = {"max_samples": [50, 100,150, 200],
                          "n_estimators": [100,200,400,500],
                          "bootstrap": [True]}

grid_search = GridSearchCV(BaggingRegressor(KNeighborsRegressor(n_neighbors=19), random_state=0), grid, cv=10, return_train_score=True)

grid_search.fit(X_train, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Cross validation
knn = KNeighborsRegressor(n_neighbors=19)

scores = cross_val_score(BaggingRegressor(knn, n_estimators=400, max_samples=200, bootstrap=True, random_state=0), X_train, y_train,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))

scores = cross_val_score(BaggingRegressor(knn, n_estimators=400, max_samples=200, bootstrap=True, random_state=0), X_test, y_test,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))


# ### Bagging is not performing well with KNN

# # Ridge Regression
from  sklearn.linear_model import Ridge
scores = cross_val_score(Ridge(alpha=.01), X_train, y_train,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))
scores = cross_val_score(Ridge(alpha=.01), X_test, y_test,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))


# RMSE for Train is 0.48 and test is 0.49

# ## Ridge with Bagging
# Grid search for Ridge for bagging hyperparameters

grid = {"max_samples": [50, 100,150, 200],
                          "n_estimators": [100,200,400,500],
                          "bootstrap": [True]}

grid_search = GridSearchCV(BaggingRegressor(Ridge(alpha=.01), random_state=0), grid, cv=10, return_train_score=True)

grid_search.fit(X_train, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

scores = cross_val_score(BaggingRegressor(Ridge(alpha=.01), n_estimators=400, max_samples=200, bootstrap=True, random_state=0), X_train, y_train,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))
scores = cross_val_score(BaggingRegressor(Ridge(alpha=.01), n_estimators=400, max_samples=200, bootstrap=True, random_state=0), X_test, y_test,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))


# No effect on using bagging with Ridge

# ## 2 models with Pasting

# # Lasso Regression
from sklearn.linear_model import Lasso

#Grid search

scores = cross_val_score(Lasso(alpha=0.001), X_train, y_train,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))

scores = cross_val_score(Lasso(alpha=0.001), X_test, y_test,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))


# RMSE for train and test is 0.48
# Grid search for Lasso for pasting hyperparameters

grid = {"max_samples": [50, 100,150, 200],
                          "n_estimators": [100,200,400,500],
                          "bootstrap": [False]}

grid_search = GridSearchCV(BaggingRegressor(Lasso(alpha=0.001), random_state=0), grid, cv=10, return_train_score=True)

grid_search.fit(X_train, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

#Using cross-validation
scores = cross_val_score(BaggingRegressor(Lasso(alpha=0.001), n_estimators=500, max_samples=100, bootstrap=False, random_state=0), X_train, y_train,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))
scores = cross_val_score(BaggingRegressor(Lasso(alpha=0.001), n_estimators=500, max_samples=100, bootstrap=False, random_state=0), X_test, y_test,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))


# Not much effect using pasting on Lasso
# # Pasting with KNN
grid = {"max_samples": [50, 100,150, 200],
                          "n_estimators": [100,200,400,500],
                          "bootstrap": [False]}

grid_search = GridSearchCV(BaggingRegressor(KNeighborsRegressor(n_neighbors=19), random_state=0), grid, cv=10, return_train_score=True)

grid_search.fit(X_train, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

knn = KNeighborsRegressor(n_neighbors=19)

scores = cross_val_score(BaggingRegressor(knn, n_estimators=100, max_samples=200, bootstrap=False, random_state=0), X_train, y_train,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))

scores = cross_val_score(BaggingRegressor(knn, n_estimators=100, max_samples=127, bootstrap=False, random_state=0), X_test, y_test,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))


# RMSE increasing with using pasting on KNN
# # Adaboosting

# ## Linear SVR
from sklearn.svm import LinearSVR

scores = cross_val_score(LinearSVR(C=1), X_train, y_train,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))
scores = cross_val_score(LinearSVR(C=1), X_test, y_test,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))


# RMSE for train and test is 0.48 and 0.49

# Grid search for LinearSVR for Adaboost hyperparameters
from sklearn.ensemble import AdaBoostRegressor
grid = {"learning_rate": [0.05, 0.1, 0.3],
                          "n_estimators": [5,10,50,100,200],
                          "loss": ['linear','square','exponential']}

grid_search = GridSearchCV(AdaBoostRegressor(LinearSVR(C=1), random_state=0), grid, cv=10, return_train_score=True)

grid_search.fit(X_train, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Cross validation
scores = cross_val_score(AdaBoostRegressor(LinearSVR(C=1), n_estimators=50, loss='linear', learning_rate=0.05, random_state=0), X_train, y_train,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))

scores = cross_val_score(AdaBoostRegressor(LinearSVR(C=1), n_estimators=50, loss='linear', learning_rate=0.05, random_state=0), X_test, y_test,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))


# ## Adaboost for svr rbf kernel
from sklearn.svm import SVR
scores = cross_val_score(SVR(kernel='rbf',gamma=.1,C=1), X_train, y_train,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))

scores = cross_val_score(SVR(kernel='rbf',gamma=0.1,C=1), X_test, y_test,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))

# Grid search for polynomial Regression for Adaboost hyperparameters

grid = {"learning_rate": [0.01,0.1,0.5,1],
                          "n_estimators": [10,50,100,200],
                          }

grid_search = GridSearchCV(AdaBoostRegressor(SVR(kernel='rbf',gamma=0.1,C=1),random_state=0), grid, cv=5, return_train_score=True)

grid_search.fit(X_train, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

scores = cross_val_score(AdaBoostRegressor(SVR(kernel='rbf',gamma=0.1,C=1), n_estimators=200, learning_rate=0.01, random_state=0), X_train, y_train,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))

scores = cross_val_score(AdaBoostRegressor(SVR(kernel='rbf',gamma=0.1,C=1), n_estimators=200, learning_rate=0.01, random_state=0), X_test, y_test,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))


# # Gradient Boosting
from  sklearn.ensemble import GradientBoostingRegressor
# Grid search for gradient Boosting hyperparameters

grid = {"learning_rate": [0.01,0.1,0.2,0.5],
                          "n_estimators": [100,200,300],
       "criterion":['mse'],
       "max_depth":[2,3,4]}

grid_search = GridSearchCV(GradientBoostingRegressor(random_state=5), grid, cv=10, return_train_score=True)

grid_search.fit(X_train, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

scores = cross_val_score(GradientBoostingRegressor(random_state=5, criterion= 'mse', learning_rate= 0.1, max_depth= 2, n_estimators= 300), X_train, y_train,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))
scores = cross_val_score(GradientBoostingRegressor(random_state=5,criterion= 'mse', learning_rate= 0.1, max_depth= 2, n_estimators= 300), X_test, y_test,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))


# Gradient Boosting is lowering the RMSE on train but not with a big difference

# ## PCA
from sklearn.decomposition import PCA
#Data Preparation for PCA
scaler = MinMaxScaler()
X_train1 = scaler.fit_transform(airbnb)
X_test1 = scaler.transform(airbnb_test)
y_train1 = y_train
y_test1 = y_test

pca = PCA().fit(X_train1)
#Plotting the Cumulative Summation of the Explained Variance
#plt.figure()
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('Number of Components')
#plt.ylabel('Variance (%)') #for each component
#plt.title('Pulsar Dataset Explained Variance')
#plt.scatter(10,.90)
#plt.show()

pca = PCA(n_components=0.90)
X_reduced_train = pca.fit_transform(X_train1)
X_reduced_test=pca.transform(X_test1)

# This will select the number of components such that the amount of variance that needs to be explained is greater than the percentage specified by n_components. 

# ## KNN Regression with PCA 
param_grid = {'n_neighbors': [ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,14,15,16,17,18,19,20]}
grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=10, return_train_score=True)

grid_search.fit(X_reduced_train, y_train1)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

train_score_array = []
test_score_array = []

for k in range(1,20):
    knn = KNeighborsRegressor(k)
    knn.fit(X_reduced_train, y_train1)
    train_score_array.append(knn.score(X_reduced_train, y_train1))
    test_score_array.append(knn.score(X_reduced_test, y_test1))

x_axis = range(1,20)
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(x_axis, train_score_array, label = 'Train Score', c = 'g')
plt.plot(x_axis, test_score_array, label = 'Test Score', c='b')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()
from sklearn.model_selection import cross_val_score

scores = cross_val_score(KNeighborsRegressor(n_neighbors=18), X_reduced_train, y_train1,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))

scores = cross_val_score(KNeighborsRegressor(n_neighbors=18), X_reduced_test, y_test1,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))


# ## Linear Regression with PCA
lreg = LinearRegression()
lreg.fit(X_reduced_train, y_train1)
print(lreg.score(X_reduced_train, y_train1))
print(lreg.score(X_reduced_test, y_test1))

# plotting the best fit line
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

X_train_rm = X_reduced_train[:,2].reshape(-1,1)
lreg.fit(X_train_rm, y_train1)
y_predict = lreg.predict(X_train_rm)

plt.plot(X_train_rm, y_predict, c = 'r')
plt.scatter(X_train_rm,y_train1)
plt.xlabel('RM')

# cross validation

scores = cross_val_score(LinearRegression(), X_reduced_train, y_train1,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))

scores = cross_val_score(LinearRegression(), X_reduced_test, y_test1,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))


# ### Lasso Regression with PCA
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
grid_search = GridSearchCV(Lasso(random_state=0), param_grid, cv=10, return_train_score=True)

grid_search.fit(X_reduced_train, y_train1)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

x_range = [0.01, 0.1, 1, 10, 100]
train_score_list = []
test_score_list = []

for alpha in x_range: 
    lasso = Lasso(alpha,random_state=0)
    lasso.fit(X_reduced_train,y_train1)
    train_score_list.append(lasso.score(X_reduced_train,y_train1))
    test_score_list.append(lasso.score(X_reduced_test, y_test1))
    
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(x_range, train_score_list, c = 'g', label = 'Train Score')
plt.plot(x_range, test_score_list, c = 'b', label = 'Test Score')
plt.xscale('log')
plt.legend(loc = 3)
plt.xlabel(r'$\alpha$')

scores = cross_val_score(Lasso(alpha=0.001), X_reduced_train, y_train1,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))

scores = cross_val_score(Lasso(alpha=0.001), X_reduced_test, y_test1,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))


# ### Ridge Regression with PCA
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10,100]}
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
grid_search = GridSearchCV(Ridge(random_state=0), param_grid, cv=10, return_train_score=True)

grid_search.fit(X_reduced_train, y_train1)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

x_range = [0.001,0.01, 0.1, 1, 10,100,1000]
train_score_list = []
test_score_list = []

for alpha in x_range: 
    ridge = Ridge(alpha)
    ridge.fit(X_reduced_train,y_train1)
    train_score_list.append(ridge.score(X_reduced_train,y_train1))
    test_score_list.append(ridge.score(X_reduced_test, y_test1))
    
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(x_range, train_score_list, c = 'g', label = 'Train Score')
plt.plot(x_range, test_score_list, c = 'b', label = 'Test Score')
plt.xscale('log')
plt.legend(loc = 3)
plt.xlabel(r'$\alpha$')

scores = cross_val_score(Ridge(alpha=10), X_reduced_train, y_train1,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))

scores = cross_val_score(Ridge(alpha=10), X_reduced_test, y_test1,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))


# ### Polynomial with PCA

from  sklearn.preprocessing  import PolynomialFeatures
from sklearn.linear_model import LinearRegression
train_score_list = []
test_score_list = []
lreg=LinearRegression()
for n in range(1,4):
    poly = PolynomialFeatures(n)
    X_train_poly_p = poly.fit_transform(X_reduced_train)
    X_test_poly_p = poly.transform(X_reduced_test)
    lreg.fit(X_train_poly_p, y_train1)
    train_score_list.append(lreg.score(X_train_poly_p, y_train1))
    test_score_list.append(lreg.score(X_test_poly_p, y_test1))

print(train_score_list)
print(test_score_list)


# Based on the above we can say this will be a polynomial of power 1 or linear
poly = PolynomialFeatures(1)
X_train_poly_p = poly.fit_transform(X_reduced_train)
X_test_poly_p = poly.transform(X_reduced_test)

scores = cross_val_score(LinearRegression(), X_train_poly_p, y_train1,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))

scores = cross_val_score(LinearRegression(), X_test_poly_p, y_test1,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))


# ## Linear SVR with PCA
from sklearn.svm import LinearSVR

param_grid = {'C': [0.001,0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LinearSVR(), param_grid, cv=10, return_train_score=True)

grid_search.fit(X_reduced_train, y_train1)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

x_range = [0.001,0.01, 0.1, 1, 10, 100]
train_score_list = []
test_score_list = []

for c in x_range: 
    linear_svm = LinearSVR(c)
    linear_svm.fit(X_reduced_train,y_train1)
    train_score_list.append(linear_svm.score(X_reduced_train,y_train1))
    test_score_list.append(linear_svm.score(X_reduced_test, y_test1))
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(x_range, train_score_list, c = 'g', label = 'Train Score')
plt.plot(x_range, test_score_list, c = 'b', label = 'Test Score')
plt.xscale('log')
plt.legend(loc = 3)
plt.xlabel(r'$\alpha$')

scores = cross_val_score(LinearSVR(C=1), X_reduced_train, y_train1,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))

scores = cross_val_score(LinearSVR(C=1), X_reduced_test, y_test1,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))


# ### SVM Kernels with PCA
from sklearn.svm import SVR
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100,1000],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
             }
print("Parameter grid:\n{}".format(param_grid))

grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=10, return_train_score=True)
grid_search.fit(X_reduced_train, y_train1)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

scores = cross_val_score(SVR(kernel='rbf',gamma=0.01,C=1000), X_reduced_train, y_train1,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))

scores = cross_val_score(SVR(kernel='rbf',gamma=0.01,C=1000), X_reduced_test, y_test1,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))


# ## Poly kernel
from sklearn.svm import SVR
param_grid = {'C': [ 10, 100,1000],
              'gamma': [0.0001, 0.001, .01],
              'degree': [1,2,3]
             }
print("Parameter grid:\n{}".format(param_grid))

grid_search = GridSearchCV(SVR(kernel='poly'), param_grid, cv=10, return_train_score=True)
grid_search.fit(X_reduced_train, y_train1)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

scores = cross_val_score(SVR(kernel='poly',gamma=0.01,C=1000,degree=1), X_reduced_train, y_train1,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))

scores = cross_val_score(SVR(kernel='poly',gamma=0.01,C=1000,degree=1), X_reduced_test, y_test1,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))


# ### Linear Kernel
from sklearn.svm import SVR
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100,1000],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
             }
print("Parameter grid:\n{}".format(param_grid))

grid_search = GridSearchCV(SVR(kernel='linear'), param_grid, cv=10, return_train_score=True)
grid_search.fit(X_reduced_train, y_train1)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

scores = cross_val_score(SVR(kernel='linear',gamma=.001,C=10), X_reduced_train, y_train1,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))

scores = cross_val_score(SVR(kernel='linear',gamma=0.001,C=10), X_reduced_test, y_test1,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("rmse score: {:.2f}".format(sqrt(abs(scores.mean()))))


# ### Results comparission of PCA with simple models
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
def score_print_reg(y,y_pred,count):
    b=[]
    b.append(sqrt(mean_squared_error(y,y_pred[count])))
    b.append(r2_score(y,y_pred[count]))
    b.append(mean_absolute_error(y,y_pred[count]))
    return b

knn_pca=KNeighborsRegressor(n_neighbors=18)
knn_pca.fit(X_reduced_train,y_train1)
linear_pca=LinearRegression()
linear_pca.fit(X_reduced_train,y_train1)
svm_lin_pca=SVR(kernel='linear',gamma=0.001,C=10)
svm_lin_pca.fit(X_reduced_train,y_train1)

svm_poly_pca=SVR(kernel='poly',gamma=0.01,C=1000,degree=1)
svm_poly_pca.fit(X_reduced_train,y_train1)

svm_rbf_pca=SVR(kernel='rbf',gamma=0.01,C=1000)
svm_rbf_pca.fit(X_reduced_train,y_train1)
linear_svr_pca=LinearSVR(C=1)
linear_svr_pca.fit(X_reduced_train,y_train1)

lasso_pca=Lasso(alpha=0.001)
lasso_pca.fit(X_reduced_train,y_train1)
ridge_pca=Ridge(alpha=10)
ridge_pca.fit(X_reduced_train,y_train1)
poly_pca=LinearRegression()
poly_pca.fit(X_train_poly_p,y_train1)

models_pca=[knn_pca,linear_pca,linear_svr_pca,svm_lin_pca,svm_poly_pca,svm_rbf_pca,lasso_pca,ridge_pca,poly_pca]
y_pred=[]
count=0
scores=[]
for i in models_pca:
    print(count)
    if (i!=poly_pca):
        y_pred.append(i.predict(pca.transform(X_test)))
        scores.append(score_print_reg(y_test1,y_pred,count))
    else:
        y_pred.append(i.predict(X_test_poly_p))
        scores.append(score_print_reg(y_test1,y_pred,count))
    count=count+1

pca_models=pd.DataFrame()
pca_models['models']=['knn_pca','linear_pca','linear_svr_pca','svm_lin_pca','svm_poly_pca','svm_rbf_pca','lasso_pca','ridge_pca','poly_pca']
pca_models=pd.concat([pca_models,pd.DataFrame(scores)], axis=1)
pca_models.columns=['models','root_mean_squared_error','r2_score','mean_absolute_error']
pca_models

knn=KNeighborsRegressor(n_neighbors=18)
knn.fit(X_train,y_train)
linear=LinearRegression()
linear.fit(X_train,y_train)


svm_lin=SVR(kernel='linear',gamma=0.001,C=10)
svm_lin.fit(X_train,y_train)

svm_poly=SVR(kernel='poly',gamma=0.01,C=1000,degree=1)
svm_poly.fit(X_train,y_train)

svm_rbf=SVR(kernel='rbf',gamma=0.01,C=1000)
svm_rbf.fit(X_train,y_train)
linear_svr=LinearSVR(C=1)
linear_svr.fit(X_train,y_train)

lasso=Lasso(alpha=0.001)
lasso.fit(X_train,y_train)
ridge=Ridge(alpha=10)
ridge.fit(X_train,y_train)

lreg=LinearRegression()

poly = PolynomialFeatures(1)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
poly=LinearRegression()
poly.fit(X_train_poly,y_train)

models=[knn,linear,linear_svr,svm_lin,svm_poly,svm_rbf,lasso,ridge,poly]
y_pred=[]
count=0
scores=[]
for i in models:
    print(count)
    if (i!=poly):
        y_pred.append(i.predict(X_test))
        scores.append(score_print_reg(y_test,y_pred,count))
    else:
        y_pred.append(i.predict(X_test_poly))
        scores.append(score_print_reg(y_test,y_pred,count))
    count=count+1

simple_models=pd.DataFrame()
simple_models['models']=['knn','linear','linear_svr','svm_lin','svm_poly','svm_rbf','lasso','ridge','poly']
simple_models=pd.concat([simple_models,pd.DataFrame(scores)], axis=1)
simple_models.columns=['models','root_mean_squared_error','r2_score','mean_absolute_error']
simple_models

concat_simple_pca=pd.concat([pca_models.transpose(),simple_models.transpose()],axis=1)
concat_simple_pca.columns=['knn_pca','linear_pca','linear_svr_pca','svm_lin_pca','svm_poly_pca','svm_rbf_pca','lasso_pca','ridge_pca','poly_pca','knn','linear','linear_svr','svm_lin','svm_poly','svm_rbf','lasso','ridge','poly']

concat_simple_pca.drop(['models'],inplace=True)

concat_reg=concat_simple_pca.sort_index(axis=1)
concat_reg


# # Neural Network for Regression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model():
    model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_dim=25),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

model10 = build_model()

model10.summary()

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 1000
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)

history = model10.fit(
  X_train, np.asarray(y_train),
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[early_stop,PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail(25)

import matplotlib.pyplot as plt
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
  
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
    plt.ylim([0,1])
    plt.legend()
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
    plt.ylim([0,2])
    plt.legend()
    plt.show()

plot_history(history)

test_predictions = model10.predict(X_test).flatten()

plt.scatter(y_test, test_predictions)
plt.xlabel('True Values Audit_Risk')
plt.ylabel('Predictions Audit_Risk')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([0, 150], [0, 150])

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from math import sqrt

print("R2=",r2_score(y_test,test_predictions))
print("RMSE",sqrt(mean_squared_error(y_test,test_predictions)))

