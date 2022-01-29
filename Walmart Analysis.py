#!/usr/bin/env python
# coding: utf-8

# # Walmart analysis

# In[1]:


# Before running, check file path for uploading CSVs and downloadning model CSV results are correct.


# ## Importing data sets

# In[2]:


# Importing libraries I will be using
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.preprocessing import LabelEncoder # To help transform the data
from sklearn.model_selection import train_test_split # To help split the data

# For ML models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# For unsupervised ML models 
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# To evaluate models
from sklearn import metrics # To check accuracy
from sklearn.metrics import confusion_matrix 
#from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# In[3]:


# The four needed csv files imported with pandas
features_df = pd.read_csv(r"C:\Users\chris\Downloads\Uni\EHU year 2\Data mining\Course work\Coursework 2\features.csv")
stores_df = pd.read_csv(r"C:\Users\chris\Downloads\Uni\EHU year 2\Data mining\Course work\Coursework 2\stores.csv")
test_df = pd.read_csv(r"C:\Users\chris\Downloads\Uni\EHU year 2\Data mining\Course work\Coursework 2\test(2).csv")
train_df = pd.read_csv(r"C:\Users\chris\Downloads\Uni\EHU year 2\Data mining\Course work\Coursework 2\train(2).csv")


# In[4]:


# Prints head, top 2 elements in table along with column names
print(features_df.head(2))
print(stores_df.head(2))
print(test_df.head(2))
print(train_df.head(2))


# ## Merge training CSV files

# In[5]:


# Merged train and features together.
# Used on = store, date and IsHoliday to ensure these columns were used to merge tables as both data
# sets had these columns. Used pandas to merge.

df1 = pd.merge(features_df, train_df, on = ("Store", "Date", "IsHoliday"))
df1


# In[6]:


# Merged stores and df1 together.
# Used on = store to ensure this column was used to merge tables as both data
# sets had this column. Used pandas to merge.
df2 = pd.merge(df1, stores_df, on = ("Store"))
df2


# ## Fill missing values

# In[7]:


# isna().sum() was used to check missing values, learnt in class
df2.isna().sum()


# In[8]:


# Here all missing values are set to 0, as mark downs are a promotional event
# and should not have the average inputted.
df = df2.fillna(0)


# In[9]:


df.isna().sum()


# ## Data analysis

# In[10]:


# Creates graphs with columns against the weekly sales which helps see correlations if any
columns = ["Date","Temperature", "Fuel_Price", "MarkDown1", "MarkDown4", "MarkDown5", "CPI", "Unemployment", "IsHoliday", "Dept", "Type", "Size", "Store"]
z = "Weekly_Sales"
for x in columns:
    # Seaborn pairplot histogram
    sns.pairplot(df, x_vars = x, y_vars = z, kind = "hist", height = 4)


# In[11]:


# Create own month and year column with pandas to see if month has any correlation with the data set
df["Year"] = pd.to_datetime(df["Date"]).dt.year
df["Month"] = pd.to_datetime(df["Date"]).dt.month
df["Week"] = pd.to_datetime(df["Date"]).dt.isocalendar().week
df


# In[12]:


# Data to go in the graph, done by year and then specficallly week compared to weekly sales column
# Gives the mean of each week in the year
weekly_sales_year1 = df[df["Year"] == 2010]["Weekly_Sales"].groupby(df["Week"]).mean()
weekly_sales_year2 = df[df["Year"] == 2011]["Weekly_Sales"].groupby(df["Week"]).mean()
weekly_sales_year3 = df[df["Year"] == 2012]["Weekly_Sales"].groupby(df["Week"]).mean()

# Creates graph to show weekly sales in the weeks to see if there is a correlation
plt.figure(figsize=(20,8))
sns.lineplot(weekly_sales_year1.index, weekly_sales_year1.values)
sns.lineplot(weekly_sales_year2.index, weekly_sales_year2.values)
sns.lineplot(weekly_sales_year3.index, weekly_sales_year3.values)

# Graph adjustments
plt.grid() # To make the graph easier to read
plt.legend(["2010", "2011", "2012"], fontsize = 20)
plt.title("Mean Weekly Sales across the years", fontsize = 30)
plt.ylabel("Weekly Sales", fontsize = 20)
plt.xlabel("Week in year", fontsize = 20)

# Displays graph
plt.show()


# In[13]:


# Data to go in the graph, done by year and then specficallly month compared to weekly sales column
# Gives the mean of each month in the year
Month_sales_year1 = df[df["Year"] == 2010]["Weekly_Sales"].groupby(df["Month"]).mean()
Month_sales_year2 = df[df["Year"] == 2011]["Weekly_Sales"].groupby(df["Month"]).mean()
Month_sales_year3 = df[df["Year"] == 2012]["Weekly_Sales"].groupby(df["Month"]).mean()

# Creates graph to show weekly sales in the monthhs to see if there is a correlation
plt.figure(figsize=(20,8))
sns.lineplot(Month_sales_year1.index, Month_sales_year1.values)
sns.lineplot(Month_sales_year2.index, Month_sales_year2.values)
sns.lineplot(Month_sales_year3.index, Month_sales_year3.values)

# Graph adjustments
plt.grid() # To make the graph easier to read
plt.legend(["2010", "2011", "2012"], fontsize=20)
plt.title("Mean Monthly Sales across the years", fontsize=30)
plt.ylabel("Monthly Sales", fontsize=20)
plt.xlabel("Month in year", fontsize=20)

# Displays graph
plt.show()


# ## Check corresponding columns

# In[14]:


correspondent = df.corr() # Checks correlation in df
plt.subplots(figsize = (15, 15)) # Size of heat map
sns.heatmap(correspondent, cmap = "tab20b", vmin = -1.0, annot = True) # Creates heatmap
plt.show() # Shows heatmap


# In[15]:


# Drops columns with low correlation and limited helpful insight on earlier graphs
df = df.drop(columns = ["Date", "CPI", "Unemployment", "Fuel_Price", "Temperature", "Year"])

# Check data frame is as expected
df


# In[16]:


# Here info used to check data types in df and entries which match the data set.
df.info()


# In[17]:


# Checks amount of categories in weekly sales
print("Amount of categories in Sales coulmn:", df["Weekly_Sales"].value_counts().count())


# ## Data transformation

# In[18]:


# Turns continous data into categorical, pandas used to help
df["Weekly_Sales"] = pd.cut(df.Weekly_Sales, bins = [-5000, -1, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 60000, 100000, 700000],
                            labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])


# In[19]:


df # Check if weekly sales assigned to different categories


# In[20]:


# Turning column values into integer from other type
# One hot encoding used to convert columns 
df["IsHoliday"] = pd.get_dummies(df["IsHoliday"]) 
df["Type"] = pd.get_dummies(df["Type"])

# Done to make sure store id number does not change output
df["Store"] = pd.get_dummies(df["Store"])

# Done to make sure Weekly_sales does not have a higher or lower weight
df["Weekly_Sales"] = pd.get_dummies(df["Weekly_Sales"])
 
# Used to check data type is not of numerical type
df.info()


# In[21]:


"""
# Wanted to see what would happen if I used label encoding instead, worsens accuracy and 
# can not read precision, recall and f1 score.
# Turning column values into integer from object
l1 = LabelEncoder() # Label encoder used from sklearn to normalize data 
l1.fit(df["Store"]) # This is the column that is normailized
# The column is updated on the df, each is repeated for each non numerical data type.
df.Store = l1.transform(df.Store) 

l1 = LabelEncoder() 
l1.fit(df["IsHoliday"]) 
df.IsHoliday = l1.transform(df.IsHoliday) 

l1 = LabelEncoder() 
l1.fit(df["Type"]) 
df.Type = l1.transform(df.Type) 

l1 = LabelEncoder() 
l1.fit(df["Weekly_Sales"]) 
df.Weekly_Sales = l1.transform(df.Weekly_Sales) 

# Checks data types in columns, all should be numerical
df.info()
"""


# In[22]:


# Here just printed df to check all rows and columns and values are as expected
df


# ## Data split

# In[23]:


# Here data is split into test, validate and train

X = df.drop("Weekly_Sales", 1) # Features assinged to X, output hidden
y = df.Weekly_Sales # Output or result hidden in seperate variable y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.25, random_state = 1)

print(X_train) # Prints to check data is successfully  hid output
print(X_test)
print(y_train) # Prints to check output is sperate and as expected
print(y_test)
print(X_valid) # Prints to check valid is sperate and as expected
print(y_valid)


# ## Test data prep

# In[24]:


# Same prep used as for the training data
df3 = pd.merge(features_df, test_df, on = ("Store", "Date", "IsHoliday"))
df3


# In[25]:


df4 = pd.merge(df3, stores_df, on = ("Store"))
df4


# In[26]:


df4.isna().sum()


# In[27]:


test_df = df4.fillna(0)
test_df


# In[28]:


# Add same columns as train data
test_df["Year"] = pd.to_datetime(test_df["Date"]).dt.year
test_df["Month"] = pd.to_datetime(test_df["Date"]).dt.month
test_df["Week"] = pd.to_datetime(test_df["Date"]).dt.isocalendar().week
test_df


# In[29]:


# Drops same columns
test_df = test_df.drop(columns = ["Date", "CPI", "Unemployment", "Fuel_Price", "Temperature", "Year"])
test_df


# In[30]:


# Turning column values into integer from other type
# One hot encoding used to convert columns
test_df["IsHoliday"] = pd.get_dummies(test_df["IsHoliday"]) 
test_df["Type"] = pd.get_dummies(test_df["Type"])

# Done to make sure store id number does not change output
test_df["Store"] = pd.get_dummies(test_df["Store"])

test_df.info()


# ## Decision Tree Model

# In[31]:


def decisionTreeModel():
    # Creates tree
    dtc = DecisionTreeClassifier().fit(X_train, y_train)

    # Predictions of test stored
    y_val = dtc.predict(X_valid) # Validation data
    y_predic = dtc.predict(X_test) # Test data

    # Shows how often model is correct
    print("The decision tree models accuracy is")
    print(metrics.accuracy_score(y_valid, y_val)*100, "%", "Validation data")
    print(metrics.accuracy_score(y_test, y_predic)*100, "%", "Test data")
    
    # View confusion matrix for test data and predictions
    print("Confusion matrix")
    print(confusion_matrix(y_test, y_predic))
    
    # Gets matrix
    matrix = confusion_matrix(y_test, y_predic) # Matrix for these 2
    matrix = matrix.astype("float") / matrix.sum(axis = 1)[:, np.newaxis]

    # Build matrix
    plt.figure(figsize = (5,5)) # Graph size
    sns.set(font_scale = 1.5) # Title and axis size

    # Creates matrix, allows numbers of size 30 and colour mako
    sns.heatmap(matrix, annot = True, annot_kws = {"size" : 30}, cmap = "mako")

    # Labels for the confusion matrix
    plt.title("Confusion matrix for the decision tree classifier")
    plt.xlabel("Predicted")
    plt.ylabel("Real label")
    plt.show() # Shows plot
    
    # Evaluation metrics for model, precision recall and f1
    precision = precision_score(y_test, y_predic)
    print("Precision: ", precision)
    recall = recall_score(y_test, y_predic)
    print("Recall: ", recall)
    f1 = f1_score(y_test, y_predic)
    print("F1 score: ", f1)
    
    # Download final results csv
    results = dtc.predict(test_df) # Final result for test data
    #results = pd.DataFrame(results, columns=["Weekly_Sales_Predictions"]).to_csv("C:/Users/chris/Downloads/DecisionTreeResults.csv") 


# In[32]:


# Here the inputted data is checked to train and evaluate the model
decisionTreeModel()


# ## K-nearest neigbour 

# In[33]:


def knn():
    # Creates Knn model, trains
    knn = KNeighborsClassifier(n_neighbors = 3).fit(X_train, y_train)

    # Predictions of test stored
    y_val = knn.predict(X_valid) # Validation data
    y_predic = knn.predict(X_test) # Test data

    # Shows how often model is correct
    print("The knn models accuracy is")
    print(metrics.accuracy_score(y_valid, y_val)*100, "%", "Validation data")
    print(metrics.accuracy_score(y_test, y_predic)*100, "%", "Test data")
    
    # View confusion matrix for test data and predictions
    print(confusion_matrix(y_test, y_predic))
    
    # Gets the matrix
    matrix = confusion_matrix(y_test, y_predic) # Matrix for these 2
    matrix = matrix.astype("float") / matrix.sum(axis = 1)[:, np.newaxis]

    # Builds the  matrix
    plt.figure(figsize = (5,5)) # Graph size
    sns.set(font_scale = 1.5) # Title and axis size

    # Creates matrix, allows numbers of size 30 and colour winter
    sns.heatmap(matrix, annot = True, annot_kws = {"size" : 30}, cmap = "winter")
    
    # Labels for the confusion matrix
    plt.title("Confusion matrix for the KNN model")
    plt.xlabel("Predicted")
    plt.ylabel("Real label")
    plt.show() # Shows plot
    
    # Evaluation metrics for model, precision recall and f1
    precision = precision_score(y_test, y_predic)
    print("Precision: ", precision)
    recall = recall_score(y_test, y_predic)
    print("Recall: ", recall)
    f1 = f1_score(y_test, y_predic)
    print("F1 score: ", f1)
    
    # Download final results csv
    results = knn.predict(test_df) # Final result for test data
    # results = pd.DataFrame(results, columns=["Weekly_Sales_Predictions"]).to_csv("C:/Users/chris/Downloads/KnnResults.csv") 


# In[34]:


knn()


# ## Random Forest Model 

# In[35]:


def randomForestModel():
    # Creates multiple random trees, called a forest, trains model
    rfc = RandomForestClassifier(n_estimators = 40).fit(X_train, y_train)

    # Predictions of test stored
    y_val = rfc.predict(X_valid) # Validation data
    y_predic = rfc.predict(X_test) # Test data

    # Shows how often model is correct, predictions checked against actual output
    print("The random forest models accuracy is")
    print(metrics.accuracy_score(y_valid, y_val)*100, "%", "Validation data")
    print(metrics.accuracy_score(y_test, y_predic)*100, "%", "Test data")
    
    # View confusion matrix for test 
    print(confusion_matrix(y_test, y_predic))
    
    # Gets matrix
    matrix = confusion_matrix(y_test, y_predic) # Matrix for these 2
    matrix = matrix.astype("float") / matrix.sum(axis = 1)[:, np.newaxis]

    # Builds the matrix
    plt.figure(figsize = (5,5)) # Graph size
    sns.set(font_scale = 1.5) # Title and axis size

    # Creates the matrix, allows numbers of size 30 and colour mako
    sns.heatmap(matrix, annot = True, annot_kws = {"size" : 30}, cmap = "magma")

    # Labels for the confusion matrix
    plt.title("Confusion matrix for the random forest classifier")
    plt.xlabel("Predicted")
    plt.ylabel("Real label")
    plt.show() # Shows plot
    
    # Evaluation metrics for model, precision recall and f1
    precision = precision_score(y_test, y_predic)
    print("Precision: ", precision)
    recall = recall_score(y_test, y_predic)
    print("Recall: ", recall)
    f1 = f1_score(y_test, y_predic)
    print("F1 score: ", f1)
    
    # Download final results csv
    results = rfc.predict(test_df) # Final result for test data
    #results = pd.DataFrame(results, columns=["Weekly_Sales_Predictions"]).to_csv("C:/Users/chris/Downloads/RandomForestResults.csv") 


# In[36]:


randomForestModel()


# ## Kmeans

# In[109]:


# Centres not added to see if three clusters would form like elbow method indicates.
# Size of test data so train data can be checked to same extenet
X, y = make_blobs(n_samples= 115064, random_state = 0, cluster_std = .2)
plt.scatter(X[:, 0], X[:, 1], s = 10) # S = size, rest makes the graph


# In[110]:


def elbowMethod():
# Elbow method, graph output says three clusters is optimal 

# To store
    wcss = []
    # Loop to run multiple Kmeans and store errors in list
    for i in range(1, 10):
        # Calls ML method, different clusters each time
        kmeans = KMeans(n_clusters = i, random_state = 0)
        # Fits data to ML method
        kmeans.fit(X) 
        wcss.append(kmeans.inertia_)
    
    # Plots the graph
    plt.plot(range(1, 10), wcss) 
    plt.title("The Elbow Method")
    plt.xlabel("The number of clusters")
    plt.ylabel("Within Cluster Sum of Squares (WCSS)")
    plt.show() # Shows the graph

elbowMethod()


# In[113]:


# Code learnt from uni material, Kmeans model
est = KMeans(n_clusters = 3)  # 3 clusters identified from elbow method
est.fit(test_df)
y_kmeans = est.predict(test_df)
plt.scatter(X[:, 0], X[:, 1], c = y_kmeans, s = 20, cmap = "autumn");
plt.show()


# In[67]:


# Final Kmean results
# Download final results csv
res = est.fit(test_df) # Final result for test data
results = res.predict(test_df)
# results = pd.DataFrame(results, columns=["Weekly_Sales_Predictions"]).to_csv("C:/Users/chris/Downloads/KmeansResults.csv")


# ## Kmean: Expectation Maximization

# In[ ]:


# Not enough time to iterate over so centroid is accurate 

