#!/usr/bin/env python
# coding: utf-8

# # Import Dataset

# In[663]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[664]:


df = pd.read_csv("./term-deposit-marketing-2020.csv")


# # Exploratory Data Analysis

# In[665]:


df.shape


# In[666]:


df.dtypes


# In[667]:


df.head(10)


# In[668]:


df.tail(10)


# In[669]:


df.info()


# In[670]:


df.describe()


# In[671]:


df["age"].isnull().unique()
df["balance"].isnull().unique()


# We can see that there is no missing values as all columns have 40,000 data each.
# <br>In addition, the five integer columns above have all integer data only.</br>

# ### Browsing non-integer columns

# In[672]:


df["job"].value_counts()


# In[673]:


df["marital"].value_counts()


# In[674]:


df["education"].value_counts()


# In[675]:


df["default"].value_counts()


# In[676]:


df["housing"].value_counts()


# In[677]:


df["loan"].value_counts()


# In[678]:


df["contact"].value_counts()


# In[679]:


df["month"].value_counts()


# September doesn't exists.

# In[680]:


df["y"].value_counts()


# # Data Visualization

# In[681]:


# Create fig, ax objects
fig, ax = plt.subplots(figsize=(16, 8))

sns.countplot(data=df, x="age", hue="y")
ax.set(title='age', xlabel='age', ylabel='y')
ax.legend()
plt.show()


# In[682]:


# Create fig, ax objects
fig, ax = plt.subplots(figsize=(16, 8))

sns.countplot(data=df, x="job", hue="y")
ax.set(title='job', xlabel='job', ylabel='y')
ax.legend()
plt.show()


# In[683]:


# Create fig, ax objects
fig, ax = plt.subplots(figsize=(16, 8))

sns.countplot(data=df, x="marital", hue="y")
ax.set(title='marital', xlabel='marital', ylabel='y')
ax.legend()
plt.show()


# In[684]:


# Create fig, ax objects
fig, ax = plt.subplots(figsize=(16, 8))

sns.countplot(data=df, x="education", hue="y")
ax.set(title='education', xlabel='education', ylabel='y')
ax.legend()
plt.show()


# In[685]:


# Create fig, ax objects
fig, ax = plt.subplots(figsize=(12, 8))

sns.countplot(data=df, x="default", hue="y")
ax.set(title='default', xlabel='default', ylabel='y')
ax.legend()
plt.show()


# In[686]:


# Create fig, ax objects
fig, ax = plt.subplots(figsize=(12, 8))

sns.countplot(data=df, x="housing", hue="y")
ax.set(title='housing', xlabel='housing', ylabel='y')
ax.legend()
plt.show()


# In[687]:


# Create fig, ax objects
fig, ax = plt.subplots(figsize=(12, 8))

sns.countplot(data=df, x="loan", hue="y")
ax.set(title='loan', xlabel='loan', ylabel='y')
ax.legend()
plt.show()


# In[688]:


# Create fig, ax objects
fig, ax = plt.subplots(figsize=(12, 8))

sns.countplot(data=df, x="contact", hue="y")
ax.set(title='contact', xlabel='contact', ylabel='y')
ax.legend()
plt.show()


# In[689]:


# Create fig, ax objects
fig, ax = plt.subplots(figsize=(12, 8))

sns.countplot(data=df, x="day", hue="y")
ax.set(title='day', xlabel='day', ylabel='y')
ax.legend()
plt.show()


# In[690]:


# Create fig, ax objects
fig, ax = plt.subplots(figsize=(12, 8))

sns.countplot(data=df, x="month", hue="y")
ax.set(title='month', xlabel='month', ylabel='y')
ax.legend()
plt.show()


# In[691]:


# Create fig, ax objects
#fig, ax = plt.subplots(figsize=(12, 8))

#sns.countplot(data=df, x="duration", hue="y")
#ax.set(title='duration', xlabel='duration', ylabel='y')
#ax.legend()
#plt.show()


# In[692]:


# Create fig, ax objects
fig, ax = plt.subplots(figsize=(12, 8))

sns.countplot(data=df, x="campaign", hue="y")
ax.set(title='campaign', xlabel='campaign', ylabel='y')
ax.legend()
plt.show()


# From the countplots above, I cannot see a specific correlation between each column and y.
# <br> I will try a Heatmap after data wrangling.</br>

# # Data Wrangling

# #### For column "marital", "education", "job", and "contact", I will use one-hot encoding for better performance.

# In[693]:


df = pd.get_dummies(df, columns=["marital"], prefix="marital")


# In[694]:


df = pd.get_dummies(df, columns=["education"], prefix="education")


# In[695]:


df = pd.get_dummies(df, columns=["job"], prefix="job")


# In[696]:


df = pd.get_dummies(df, columns=["contact"], prefix="contact")


# In[697]:


df.head()


# #### For columns "default", "housing", and "loan", I will replace the values to 0 for 'no' and 1 for 'yes'.

# In[698]:


df.loc[df["default"] == "no", "default"] = 0
df.loc[df["default"] == "yes", "default"] = 1


# In[699]:


df.loc[df["housing"] == "no", "housing"] = 0
df.loc[df["housing"] == "yes", "housing"] = 1


# In[700]:


df.loc[df["loan"] == "no", "loan"] = 0
df.loc[df["loan"] == "yes", "loan"] = 1


# In[701]:


df.loc[df["y"] == "no", "y"] = 0
df.loc[df["y"] == "yes", "y"] = 1


# In[702]:


df.loc[df["month"] == "jan", "month"] = 1
df.loc[df["month"] == "feb", "month"] = 2
df.loc[df["month"] == "mar", "month"] = 3
df.loc[df["month"] == "apr", "month"] = 4
df.loc[df["month"] == "may", "month"] = 5
df.loc[df["month"] == "jun", "month"] = 6
df.loc[df["month"] == "jul", "month"] = 7
df.loc[df["month"] == "aug", "month"] = 8
df.loc[df["month"] == "sep", "month"] = 9
df.loc[df["month"] == "oct", "month"] = 10
df.loc[df["month"] == "nov", "month"] = 11
df.loc[df["month"] == "dec", "month"] = 12


# In[703]:


df["default"].value_counts()


# In[704]:


df["housing"].value_counts()


# In[705]:


df["loan"].value_counts()


# In[706]:


df["y"].value_counts()


# In[707]:


df["month"].value_counts()


# In[708]:


# Change object types to integer types to be included when drawing a Heatmap
df[["default","housing","loan","month","y"]] = df[["default","housing","loan","month","y"]].astype(str).astype(int)


# In[709]:


df.head()


# In[710]:


df.info()


# In[711]:


cols = df.iloc[:,0:-1].columns
df_int = df.iloc[:,0:-1]


# In[712]:


df_int.info()


# In[713]:


df_int.describe()


# As the values have various ranges per column, I'm going to use Robust Scaler for preprocessing the dataset as some columns look like having some outliers, such as 'balance' and 'duration'.

# In[714]:


from sklearn.preprocessing import RobustScaler


# In[715]:


robust_scale = RobustScaler()
df_robust = robust_scale.fit_transform(df_int)


# In[716]:


df_robust = pd.DataFrame(df_robust, columns = cols)


# In[717]:


df_robust.head()


# In[718]:


df_robust.describe()


# In[719]:


# Create correlation matrix with standard scaler
corr = df_robust.corr()


# In[720]:


#corr = df_int.corr()


# In[721]:


corr['y'].sort_values(ascending=False)


# In[722]:


# Draw a Heatmap with Seaborn
plt.figure(figsize = (30, 24))
sns.heatmap(corr, annot = True)
plt.show()


# As we can see, "duration" (last contact duration, in seconds) is quite correlated with "y" as the correlation coefficient is 0.46.
# <br>However, all other columns are barely correlated with "y" as their absolute value of correlation coefficient is lower than 0.2.</br>

# In[723]:


# Create a feature
#feature = ["duration", "balance", "marital_divorced", "marital_single", "education_tertiary"]
feature = ["duration"]
X = np.array(df[feature])
y = np.array(df["y"])


# In[724]:


print(X.shape, y.shape)


# In[725]:


print(y)


# # Split train and test

# In[726]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=34)


# # Modeling

# In[727]:


# Loadling libraries for K-Fold Cross Validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.model_selection import KFold


# ### 1. Decision Tree

# In[728]:


# Create decision tree model
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()


# ### 2. Logistic Regression

# In[729]:


# Create logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# ### 3. Random Forest

# In[730]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)


# ### 4. K-Nearest Neighbors

# In[731]:


# Create K-Nearest Neighbors model
from sklearn.neighbors import KNeighborsClassifier
knn1 = KNeighborsClassifier(n_neighbors = 1)
knn3 = KNeighborsClassifier(n_neighbors = 3)
knn5 = KNeighborsClassifier(n_neighbors = 5)


# ### 5. Naive Bayes

# In[732]:


# Create Naive Bayes model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()


# ### 6. Support Vector Machines

# In[733]:


# Create SVM model
from sklearn.svm import SVC
svc_rbf = SVC()
svc_sigmoid = SVC(kernel='sigmoid')


# # Validation

# ### K-fold Cross Validation

# In[734]:


scoring = "f1"
dt_score = cross_validate(dt, X_train, y_train, scoring=scoring, return_estimator=True)
lr_score = cross_validate(lr, X_train, y_train, scoring=scoring, return_estimator=True)
#rf_score = cross_validate(rf, X_train, y_train, scoring=scoring, return_estimator=True)
knn1_score = cross_validate(knn1, X_train, y_train, scoring=scoring, return_estimator=True)
knn3_score = cross_validate(knn3, X_train, y_train, scoring=scoring, return_estimator=True)
knn5_score = cross_validate(knn5, X_train, y_train, scoring=scoring, return_estimator=True)
gnb_score = cross_validate(gnb, X_train, y_train, scoring=scoring, return_estimator=True)
svc_rbf_score = cross_validate(svc_rbf, X_train, y_train, scoring=scoring, return_estimator=True)
svc_sigmoid_score = cross_validate(svc_sigmoid, X_train, y_train, scoring=scoring, return_estimator=True)


# In[735]:


# Print the mean score of each model
print("Decision Tree score:", dt_score["test_score"].mean())
print("Logistic Regression score:", lr_score["test_score"].mean())
#print("Random Forest score:", rf_score["test_score"].mean())
print("K-Nearst Neighbors 1 score:", knn1_score["test_score"].mean())
print("K-Nearst Neighbors 3 score:", knn3_score["test_score"].mean())
print("K-Nearst Neighbors 5 score:", knn5_score["test_score"].mean())
print("Naive Bayes score:", gnb_score["test_score"].mean())
print("Support Vector Machines with rbf score:", svc_rbf_score["test_score"].mean())
print("Support Vector Machines with sigmoid score:", svc_sigmoid_score["test_score"].mean())


# Since Naive Bayes showed the best score, I will re-train the model and test it using test data.

# In[740]:


# Retrain the model and evaluate
import sklearn
gnb_re = sklearn.base.clone(gnb)
gnb_re.fit(X_train, y_train)
print("Test set Accuracy:", accuracy_score(y_test, gnb_re.predict(X_test), normalize=False))
print("Test set RMSE:", mean_squared_error(y_test, gnb_re.predict(X_test), squared=False))
print("Mean validation RMSE:", -gnb_score["test_score"].mean())


# # Conclusion

# - NEED TO MODIFY LATER
# <br>We could see that the only 'duration' (last contact duration, in seconds) has positive correlation with the term deposit subscription. This means that no matter how old the customer is, where they lives, or what they do, there are more chances for the person who responds to the call longer to subscribe the term deposit.</br>

"""

# In[ ]:





# In[ ]:





# # Misc

# In[341]:


cv = KFold(5, shuffle=True, random_state=135)
scores = cross_val_score(dt_clf, X_train, y_train, scoring="accuracy", cv=cv)
print('Scores: \n{}'.format(scores))
print('Mean of scores: \n{:.4f}'.format(scores.mean()))


# In[ ]:





# In[ ]:





# # Stratified K-fold Cross Validation

# In[1]:


from sklearn.model_selection import StratifiedKFold


# In[144]:


cv_accuracy = []


# In[145]:


skf = StratifiedKFold(n_splits=5)


# In[146]:


for train_index, test_index in skf.split(X, y):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model_LR.fit(X_train, y_train)
    predict = model_LR.predict(X_test)
    
    accuracy = np.round(accuracy_score(y_test, predict), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    
    print('CV accuracy: {:.4f}, Train data size: {}, Test data size: {}'.format(accuracy, train_size, test_size))
    cv_accuracy.append(accuracy)


# In[147]:


print("Accuracy_mean: {:.4f}".format(np.mean(cv_accuracy)))


# 


"""