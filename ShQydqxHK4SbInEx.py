# %% [markdown]
# # Import Dataset

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv("./term-deposit-marketing-2020.csv")

# %% [markdown]
# # Exploratory Data Analysis

# %%
df.shape

# %%
df.dtypes

# %%
df.head(10)

# %%
df.tail(10)

# %%
df.info()

# %%
df.describe()

# %%
df["age"].isnull().unique()
df["balance"].isnull().unique()

# %% [markdown]
# We can see that there is no missing values as all columns have 40,000 data each.
# <br>In addition, the five integer columns above have all integer data only.</br>

# %% [markdown]
# ### Browsing non-integer columns

# %%
df["job"].value_counts()

# %%
df["marital"].value_counts()

# %%
df["education"].value_counts()

# %%
df["default"].value_counts()

# %%
df["housing"].value_counts()

# %%
df["loan"].value_counts()

# %%
df["contact"].value_counts()

# %%
df["month"].value_counts()

# %% [markdown]
# September doesn't exists.

# %%
df["y"].value_counts()

# %% [markdown]
# # Data Visualization

# %%
# Create fig, ax objects
fig, ax = plt.subplots(figsize=(16, 8))

sns.countplot(data=df, x="age", hue="y")
ax.set(title='age', xlabel='age', ylabel='y')
ax.legend()
plt.show()

# %%
# Create fig, ax objects
fig, ax = plt.subplots(figsize=(16, 8))

sns.countplot(data=df, x="job", hue="y")
ax.set(title='job', xlabel='job', ylabel='y')
ax.legend()
plt.show()

# %%
# Create fig, ax objects
fig, ax = plt.subplots(figsize=(16, 8))

sns.countplot(data=df, x="marital", hue="y")
ax.set(title='marital', xlabel='marital', ylabel='y')
ax.legend()
plt.show()

# %%
# Create fig, ax objects
fig, ax = plt.subplots(figsize=(16, 8))

sns.countplot(data=df, x="education", hue="y")
ax.set(title='education', xlabel='education', ylabel='y')
ax.legend()
plt.show()

# %%
# Create fig, ax objects
fig, ax = plt.subplots(figsize=(12, 8))

sns.countplot(data=df, x="default", hue="y")
ax.set(title='default', xlabel='default', ylabel='y')
ax.legend()
plt.show()

# %%
# Create fig, ax objects
fig, ax = plt.subplots(figsize=(12, 8))

sns.countplot(data=df, x="housing", hue="y")
ax.set(title='housing', xlabel='housing', ylabel='y')
ax.legend()
plt.show()

# %%
# Create fig, ax objects
fig, ax = plt.subplots(figsize=(12, 8))

sns.countplot(data=df, x="loan", hue="y")
ax.set(title='loan', xlabel='loan', ylabel='y')
ax.legend()
plt.show()

# %%
# Create fig, ax objects
fig, ax = plt.subplots(figsize=(12, 8))

sns.countplot(data=df, x="contact", hue="y")
ax.set(title='contact', xlabel='contact', ylabel='y')
ax.legend()
plt.show()

# %%
# Create fig, ax objects
fig, ax = plt.subplots(figsize=(12, 8))

sns.countplot(data=df, x="day", hue="y")
ax.set(title='day', xlabel='day', ylabel='y')
ax.legend()
plt.show()

# %%
# Create fig, ax objects
fig, ax = plt.subplots(figsize=(12, 8))

sns.countplot(data=df, x="month", hue="y")
ax.set(title='month', xlabel='month', ylabel='y')
ax.legend()
plt.show()

# %%
# Create fig, ax objects
#fig, ax = plt.subplots(figsize=(12, 8))

#sns.countplot(data=df, x="duration", hue="y")
#ax.set(title='duration', xlabel='duration', ylabel='y')
#ax.legend()
#plt.show()

# %%
# Create fig, ax objects
fig, ax = plt.subplots(figsize=(12, 8))

sns.countplot(data=df, x="campaign", hue="y")
ax.set(title='campaign', xlabel='campaign', ylabel='y')
ax.legend()
plt.show()

# %% [markdown]
# From the countplots above, I cannot see a specific correlation between each column and y.
# <br> I will try a Heatmap after data wrangling.</br>

# %% [markdown]
# # Data Wrangling

# %% [markdown]
# #### For column "marital", "education", "job", and "contact", I will use one-hot encoding for better performance.

# %%
df = pd.get_dummies(df, columns=["marital"], prefix="marital")

# %%
df = pd.get_dummies(df, columns=["education"], prefix="education")

# %%
df = pd.get_dummies(df, columns=["job"], prefix="job")

# %%
df = pd.get_dummies(df, columns=["contact"], prefix="contact")

# %%
df.head()

# %% [markdown]
# #### For columns "default", "housing", and "loan", I will replace the values to 0 for 'no' and 1 for 'yes'.

# %%
df.loc[df["default"] == "no", "default"] = 0
df.loc[df["default"] == "yes", "default"] = 1

# %%
df.loc[df["housing"] == "no", "housing"] = 0
df.loc[df["housing"] == "yes", "housing"] = 1

# %%
df.loc[df["loan"] == "no", "loan"] = 0
df.loc[df["loan"] == "yes", "loan"] = 1

# %%
df.loc[df["y"] == "no", "y"] = 0
df.loc[df["y"] == "yes", "y"] = 1

# %%
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

# %%
df["default"].value_counts()

# %%
df["housing"].value_counts()

# %%
df["loan"].value_counts()

# %%
df["y"].value_counts()

# %%
df["month"].value_counts()

# %%
# Change object types to integer types to be included when drawing a Heatmap
df[["default","housing","loan","month","y"]] = df[["default","housing","loan","month","y"]].astype(str).astype(int)

# %%
df.head()

# %%
df.info()

# %% [markdown]
# As the numbers in column 'day' and 'month' don't mean any order or prioirty that can affect to 'y', I will drop the columns.

# %%
df.drop(columns=['day', 'month'], inplace=True)

# %% [markdown]
# 

# %%
cols = df.iloc[:,0:-1].columns
df_int = df.iloc[:,0:-1]

# %%
df_int.info()

# %%
df_int.describe()

# %% [markdown]
# As the values have various ranges per column, I'm going to use Robust Scaler for preprocessing the dataset as some columns look like having some outliers, such as 'balance' and 'duration'.

# %%
from sklearn.preprocessing import RobustScaler

# %%
robust_scale = RobustScaler()
df_robust = robust_scale.fit_transform(df_int)

# %%
df_robust = pd.DataFrame(df_robust, columns = cols)

# %%
df_robust.head()

# %%
df_robust.describe()

# %%
# Create correlation matrix with standard scaler
corr = df_robust.corr()

# %%
#corr = df_int.corr()

# %%
corr['y'].sort_values(ascending=False)

# %%
# Draw a Heatmap with Seaborn
plt.figure(figsize = (30, 24))
sns.heatmap(corr, annot = True)
plt.show()

# %% [markdown]
# As we can see, "duration" (last contact duration, in seconds) is quite correlated with "y" as the correlation coefficient is 0.46.
# <br>However, all other columns are barely correlated with "y" as their absolute value of correlation coefficient is lower than 0.2.</br>

# %%
# Create a feature
#feature = ["duration", "balance", "marital_divorced", "marital_single", "education_tertiary"]
feature = ["duration"]
X = np.array(df[feature])
y = np.array(df["y"])

# %%
print(X.shape, y.shape)

# %%
print(y)

# %% [markdown]
# # Split train and test

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=34)

# %%
print(X_train.shape, y_train.shape)

# %% [markdown]
# # Handling Imbalanced Data

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

# %%
## No Imbalance Handling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate

# Define model
model_ori=RandomForestClassifier(criterion='entropy')
# Define evaluation procedure (here we use Repeated Stratified K-Fold CV)
cv_ori=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# Evaluate model
scoring=['accuracy','precision_macro','recall_macro']
scores_ori = cross_validate(model_ori, X_train, y_train, scoring=scoring, cv=cv_ori, n_jobs=-1)
# summarize performance
print('Mean Accuracy: %.4f' % np.mean(scores_ori['test_accuracy']))
print('Mean Precision: %.4f' % np.mean(scores_ori['test_precision_macro']))
print('Mean Recall: %.4f' % np.mean(scores_ori['test_recall_macro']))

# %% [markdown]
# As we can see here, the accuracy metric score is very high, while the recall metric score is quite low. This means that the model failed to learn the minority class well, thus failed to correctly predict the minority class label.
# As the data is imbalanced, I will apply the sampling strategy to improve the model's performance, especially the SMOTE-Tomek Links method, which is combined SMOTE (over-sampling) and Tomek's Link (under-sampling).

# %%
## With SMOTE-Tomek Links method
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

# Define model
model=RandomForestClassifier(criterion='entropy')
# Define SMOTE-Tomek Links
resample=SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
# Define pipeline
pipeline=Pipeline(steps=[('r', resample), ('m', model)])
# Define evaluation procedure (here we use Repeated Stratified K-Fold CV)
cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# Evaluate model
scoring=['accuracy','precision_macro','recall_macro']
scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)
# summarize performance
print('Mean Accuracy: %.4f' % np.mean(scores['test_accuracy']))
print('Mean Precision: %.4f' % np.mean(scores['test_precision_macro']))
print('Mean Recall: %.4f' % np.mean(scores['test_recall_macro']))

# %% [markdown]
# The accuracy and precision metrics might decrease, but we can see that the recall metric are higher, it means that the model performs better to correctly predict the minority class label by using SMOTE-Tomek Links to handle the imbalanced data.

# %% [markdown]
# # Detecting Outliers

# %% [markdown]
# ### Isolation Forest

# %%
from sklearn.ensemble import IsolationForest
# identify outliers in the training dataset
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(X_train)

# %%
# select all rows that are not outliers - removing outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]

# %%
print(X_train.shape, y_train.shape)

# %% [markdown]
# # Modeling with sampling

# %%
# Loadling libraries for K-Fold Cross Validation, Grid Search and etc.
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

# %%
# Load libraries for machine learning models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# %% [markdown]
# ### 1. Decision Tree

# %%
# Create decision tree model
from sklearn.tree import DecisionTreeClassifier
# Define model
model = DecisionTreeClassifier()
# Define SMOTE-Tomek Links
resample=SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
# Define pipeline
pipeline=Pipeline(steps=[('r', resample), ('m', model)])
# Define evaluation procedure (here we use Repeated Stratified K-Fold CV)
cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# Evaluate model
scoring=['accuracy','precision_macro','recall_macro']
scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)
# Save the performance results
dt_accu = np.mean(scores['test_accuracy'])
dt_prec = np.mean(scores['test_precision_macro'])
dt_reca = np.mean(scores['test_recall_macro'])

'''
# %% [markdown]
# ### 2. Logistic Regression (with Hyperparameter tuning)

# %%
# Create logistic regression model
from sklearn.linear_model import LogisticRegression
# Define model
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train,)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
'''

# %%
# Define model with the best parameter values
model = LogisticRegression(C=100, penalty='l2', solver='newton-cg')
# Define SMOTE-Tomek Links
resample=SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
# Define pipeline
pipeline=Pipeline(steps=[('r', resample), ('m', model)])
# Define evaluation procedure (here we use Repeated Stratified K-Fold CV)
cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# Evaluate model
scoring=['accuracy','precision_macro','recall_macro']
scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)
# Save the performance results
lr_accu = np.mean(scores['test_accuracy'])
lr_prec = np.mean(scores['test_precision_macro'])
lr_reca = np.mean(scores['test_recall_macro'])

'''
# %% [markdown]
# ### 3. Random Forest (with Hyperparameter tuning)

# %%
from sklearn.ensemble import RandomForestRegressor
# Define model
model = RandomForestRegressor(random_state = 42)
n_estimators = [10, 100, 1000]
max_features = ['sqrt', 'log2']
# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
'''

# %%
# Define model with the best parameter values
model = RandomForestRegressor(max_features='sqrt', n_estimators=10, random_state = 42)
# Define SMOTE-Tomek Links
resample=SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
# Define pipeline
pipeline=Pipeline(steps=[('r', resample), ('m', model)])
# Define evaluation procedure (here we use Repeated Stratified K-Fold CV)
cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# Evaluate model
scoring=['accuracy','precision_macro','recall_macro']
scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)
# Save the performance results
rf_accu = np.mean(scores['test_accuracy'])
rf_prec = np.mean(scores['test_precision_macro'])
rf_reca = np.mean(scores['test_recall_macro'])

'''
# %% [markdown]
# ### 4. K-Nearest Neighbors (with Hyperparameter tuning)

# %%
# Create K-Nearest Neighbors model
from sklearn.neighbors import KNeighborsClassifier
# Define model
model = KNeighborsClassifier()
n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']
# define grid search
grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
'''

# %%
# Define model with the best parameter values
model = KNeighborsClassifier(metric='euclidean', n_neighbors=19, weights='uniform')
# Define SMOTE-Tomek Links
resample=SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
# Define pipeline
pipeline=Pipeline(steps=[('r', resample), ('m', model)])
# Define evaluation procedure (here we use Repeated Stratified K-Fold CV)
cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# Evaluate model
scoring=['accuracy','precision_macro','recall_macro']
scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)
# Save the performance results
knn_accu = np.mean(scores['test_accuracy'])
knn_prec = np.mean(scores['test_precision_macro'])
knn_reca = np.mean(scores['test_recall_macro'])

# %% [markdown]
# ### 5. Naive Bayes

# %%
# Create Naive Bayes model
from sklearn.naive_bayes import GaussianNB
# Define model
model = GaussianNB()
# Define SMOTE-Tomek Links
resample=SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
# Define pipeline
pipeline=Pipeline(steps=[('r', resample), ('m', model)])
# Define evaluation procedure (here we use Repeated Stratified K-Fold CV)
cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# Evaluate model
scoring=['accuracy','precision_macro','recall_macro']
scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)
# Save the performance results
nb_accu = np.mean(scores['test_accuracy'])
nb_prec = np.mean(scores['test_precision_macro'])
nb_reca = np.mean(scores['test_recall_macro'])

# %% [markdown]
# ### 6. Support Vector Machines (with Hyperparameter tuning)

'''
# %%
# Create SVM model
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# Define model
model = SVC()
kernel = ['poly', 'rbf']
C = [10, 1.0, 0.1]
gamma = ['scale']
# define grid search
grid = dict(kernel=kernel,C=C,gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# %% [markdown]
# It took over 10 hours to complete the above SVM hyperparameter tuning
'''

# %%
# Define model with the best parameter values
model = SVC(C=0.1, gamma='scale', kernel='rbf')
# Define SMOTE-Tomek Links
resample=SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
# Define pipeline
pipeline=Pipeline(steps=[('r', resample), ('m', model)])
# Define evaluation procedure (here we use Repeated Stratified K-Fold CV)
cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# Evaluate model
scoring=['accuracy','precision_macro','recall_macro']
scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)
# Save the performance results
svm_accu = np.mean(scores['test_accuracy'])
svm_prec = np.mean(scores['test_precision_macro'])
svm_reca = np.mean(scores['test_recall_macro'])

# %%
# summarize performance
print('Decision Tree       - Mean Accuracy: %.4f' % dt_accu + ' / ' + 'Mean Precision: %.4f' % dt_prec + ' / ' + 'Mean Recall: %.4f' % dt_reca)
print('Logistic Regression - Mean Accuracy: %.4f' % lr_accu + ' / ' + 'Mean Precision: %.4f' % lr_prec + ' / ' + 'Mean Recall: %.4f' % lr_reca)
print('Random Forest       - Mean Accuracy: %.4f' % rf_accu + ' / ' + 'Mean Precision: %.4f' % rf_prec + ' / ' + 'Mean Recall: %.4f' % rf_reca)
print('K-Nearest Neighbors - Mean Accuracy: %.4f' % knn_accu + ' / ' + 'Mean Precision: %.4f' % knn_prec + ' / ' + 'Mean Recall: %.4f' % knn_reca)
print('Naive Bayes         - Mean Accuracy: %.4f' % nb_accu + ' / ' + 'Mean Precision: %.4f' % nb_prec + ' / ' + 'Mean Recall: %.4f' % nb_reca)
print('SVM                 - Mean Accuracy: %.4f' % svm_accu + ' / ' + 'Mean Precision: %.4f' % svm_prec + ' / ' + 'Mean Recall: %.4f' % svm_reca)

# %% [markdown]
# # Predict

# %% [markdown]
# ### Resampling data to fit

# %%
resample = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
X_train_smt, y_train_smt = resample.fit_resample(X_train, y_train)
X_test_smt, y_test_smt = resample.fit_resample(X_test, y_test)

# %% [markdown]
# ### 1. Decision Tree

# %%
# Define model
model = DecisionTreeClassifier()

model.fit(X_train_smt, y_train_smt)
y_pred_dt = model.predict(X_test_smt)
print(y_pred_dt)

# %% [markdown]
# ### 2. Logistic Regression

# %%
# Define model with the best parameter values
model = LogisticRegression(C=100, penalty='l2', solver='newton-cg')

model.fit(X_train_smt, y_train_smt)
y_pred_lr = model.predict(X_test_smt)
print(y_pred_lr)

# %% [markdown]
# ### 3. Random Forest

# %%
# Define model with the best parameter values
model = RandomForestRegressor(max_features='sqrt', n_estimators=10, random_state = 42)

model.fit(X_train_smt, y_train_smt)
y_pred_rf = model.predict(X_test_smt)
print(y_pred_rf)

# %% [markdown]
# ### 4. K-Nearest Neighbors

# %%
# Define model with the best parameter values
model = KNeighborsClassifier(metric='euclidean', n_neighbors=19, weights='uniform')

model.fit(X_train_smt, y_train_smt)
y_pred_knn = model.predict(X_test_smt)
print(y_pred_knn)

# %% [markdown]
# ### 5. Naive Beyes

# %%
# Define model with the best parameter values
model = GaussianNB()

model.fit(X_train_smt, y_train_smt)
y_pred_nb = model.predict(X_test_smt)
print(y_pred_nb)

# %% [markdown]
# ### 6. SVM

# %%
# Define model with the best parameter values
model = SVC(C=0.1, gamma='scale', kernel='rbf')

model.fit(X_train_smt, y_train_smt)
y_pred_svm = model.predict(X_test_smt)
print(y_pred_svm)

# %%
# Comparing results
print("Test score of Decision Tree: {}".format(accuracy_score(y_test_smt, y_pred_dt)))
print("Test score of Logistic Regression: {}".format(accuracy_score(y_test_smt, y_pred_lr)))
print("Test score of K-Nearest Neighbors: {}".format(accuracy_score(y_test_smt, y_pred_knn)))
print("Test score of Naive Beyes: {}".format(accuracy_score(y_test_smt, y_pred_nb)))
print("Test score of SVM: {}".format(accuracy_score(y_test_smt, y_pred_svm)))

# %% [markdown]
# # Conclusion

# %% [markdown]
# <br>1. We could see that the only 'duration' (last contact duration, in seconds) has positive correlation with the term deposit subscription. This means that no matter how old the customer is, where they lives, or what they do, there are more chances for the person who responds to the call longer to subscribe the term deposit.</br>
# <br>2. In terms of model performance, Logistic Regression and SVM showed a similar accuracy but SVM was very slightly better.</br>
# <br>3. I've also checked outlier to remove but the performance got worse when used the detecting outlier method, so I decided not to use it.

# %% [markdown]
# 


