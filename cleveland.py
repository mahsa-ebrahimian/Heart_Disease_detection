#!/usr/bin/env python
# coding: utf-8

# Detection of heart disease

# In[ ]:


##1. import the libraries
import pandas as pd
import numpy as np
import math
from math import sqrt
import seaborn as sns
from sklearn import datasets, linear_model, metrics 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model 
import matplotlib.pyplot as plt 


# In[ ]:


# 2.read the data set file
df = pd.read_csv('cleveland.csv')
df


# In[ ]:


#3.
df.info()


# In[ ]:


#4.
df.describe()


# **Preproccessing Steps**

# In[ ]:


# 5. Check if there is missing value
#df.isnull().sum()
print("There is {} missing values in data frame".format(df.isnull().sum().sum()))


# In[ ]:


# 6. This will select float columns only
float_col = df.select_dtypes(include=['float64']) 
list(float_col.columns.values)
for col in float_col.columns.values:
   df[col] = df[col].astype('int64')
df


# In[ ]:


# 7.Convert categorical features to numerical
df['Ca'] =df['Ca'].astype('category').cat.codes
df['Thal'] =df['Thal'].astype('category').cat.codes
df


# In[ ]:


# 8.Creating the class column for binary classification
df['Class']=df['Num'].apply(lambda x: 1 if x>=1 else 0)
df


# In[ ]:


# 9. removing the "Num" column since we are doing a binary classification in the first step because our label col. is called class
df=df.drop(['Num'], axis=1)
df


# In[ ]:


# view how many 0 and 1 are in the class feature
import seaborn as sns
sns.countplot(x="Class", data=df, palette="bwr")
plt.title('Class Distributions \n 0: No Disease || 1: Disease', fontsize=14)
plt.show()


# **Quantitative assesment**

# In[ ]:


# 10-Correlation Matrix
corr=df.corr()
plt.figure(figsize=(15,12))
sns.heatmap(corr,annot=True,cmap="inferno")


# In[ ]:


## 11. What features have the maximum corrolation

def get_redundant_pairs(x):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, x.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(x, n=14):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(x)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print("---------------------------")
print(get_top_abs_correlations(df,10))


# In[23]:


# xgboost for feature importance on a classification problem
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=303, n_features=13, n_informative=5, n_redundant=5, random_state=1)
# define the model
model = XGBClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[21]:



# random forest for feature importance on a classification problem
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=303, n_features=13, n_informative=5, n_redundant=5, random_state=1)
# define the model
model = RandomForestClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[ ]:


# 12. Calculate The Significant Value of the Features
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

X, y = make_regression(n_samples=303, n_features=13, n_informative=13, random_state=1)
model = LinearRegression()
model.fit(X, y) # fit the model

importance = model.coef_   # get importance
#summarize feature importance
#Feature=['Age',	'Sex',	'CP'	,'Trestbpss',	'Chol',	'Fbs',	'Restecg',	'Thalach'	,'Oldpeak'	,'Exang'	,'Slope'	,'Class']
print("The Significant Value of the Features")
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.3f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[ ]:


df=df.drop(['Sex','Chol','Exang'],axis=1)


# In[ ]:


# 13. Normalization: Scale the dataset
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
min_max_scaler = preprocessing.MinMaxScaler()
df_scale= preprocessing.MinMaxScaler().fit_transform(df)
df_scale=pd.DataFrame(df_scale, columns=['Age','CP','Trestbpss','Fbs','Restecg' ,'Thalach','Oldpeak','Ca','Thal','Slope','Class'])
# Separate the target from the data set
X = df_scale.drop(['Class'], axis=1)
y = df_scale['Class']


# **Split the data set into training and testing sets**

# In[ ]:


# 14.Split the data set into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1)
print('Training instances Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing instances Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)


# In[ ]:


# 15.Build the models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

mlp=MLPClassifier(solver='adam', learning_rate_init = 0.0005, learning_rate = 'adaptive', activation="relu", max_iter=3000, random_state=10)
dec = DecisionTreeClassifier()
ran = RandomForestClassifier(n_estimators=100)
knn = KNeighborsClassifier(n_neighbors=100)
svm = SVC(random_state=1)
naive = GaussianNB()

models = {"DT" : dec,
          "RF" : ran,
          "MLP" :mlp,
          "KNN" : knn,
          "SVM" : svm,
          "NB" : naive}
scores= { }

for key, value in models.items():    
    model = value
    model.fit(X_train,y_train)
    scores[key] = model.score(X_test, y_test)

# 16.Measure the Accuracy
scores_frame = pd.DataFrame(scores, index=["Accuracy Score"]).T
scores_frame.sort_values(by=["Accuracy Score"], axis=0 ,ascending=False, inplace=True)
scores_frame


# In[ ]:


# 17.PLOT THE RESULTS
plt.figure(figsize=(8,5))
sns.barplot(x=scores_frame.index,y=scores_frame["Accuracy Score"])
plt.ylim(0, 1)
plt.ylabel("Accuracy%")
plt.xlabel("Cleveland Dataset")
plt.xticks(rotation=45)


# # **Evaluation Metrics**

# In[ ]:


# Recall, F1 Score, Precision

from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
givenDec = lambda gdVal: float('%.3f' % gdVal) # 1 digit

def evaluate_results(gtestLabels, gTestPred):
    d_accuracy = round(accuracy_score(gtestLabels, gTestPred), 4)
    v_recall = np.round(recall_score(gtestLabels, gTestPred, average = None), 4)
    v_prec = np.round(precision_score(gtestLabels, gTestPred, average = None), 4)
    
    d_recallAvg = np.round(recall_score(gtestLabels, gTestPred, average = 'weighted'), 4)
    d_precAvg = np.round(precision_score(gtestLabels, gTestPred, average = 'weighted'), 4)
    v_summaryReport = classification_report(gtestLabels, gTestPred, digits = 4)

    print('\n')
    print(v_summaryReport)

    #print('\n')
    #print("d_accuracy" + '\t' + str(d_accuracy))
    #print("v_recall" + '\t' + str(v_recall[0]) + '\t' + str(v_recall[1]) + '\t' + str(d_recallAvg))
    #print("v_prec" + '\t' + str(v_prec[0]) + '\t' + str(v_prec[1]) + '\t' + str(d_precAvg))
    
test_predictions = model.predict(X_test)
evaluate_results(y_test, test_predictions)


# In[ ]:


##### ROC ######################
# 1. Random Forest
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from inspect import signature

ran = RandomForestClassifier()
ran.fit(X_train, y_train)
probs = ran.predict_proba(X_test)

malignant_probs = probs[:,1]
fpr, tpr, thresholds = roc_curve(y_test, malignant_probs)
roc_auc = auc(fpr, tpr)

plt.title('Random Forest Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'y', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


#2. KNN

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
probs = knn.predict_proba(X_test)

malignant_probs = probs[:,1]
fpr, tpr, thresholds = roc_curve(y_test, malignant_probs)
roc_auc = auc(fpr, tpr)

plt.title('KNN Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'y', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


#3.Gnaive B 
naive = GaussianNB()
naive.fit(X_train, y_train)
probs = naive.predict_proba(X_test)


malignant_probs = probs[:,1]
fpr, tpr, thresholds = roc_curve(y_test, malignant_probs)
roc_auc = auc(fpr, tpr)

plt.title('GaussianNB Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'y', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


#4.SVM
svm = SVC()
svm.fit(X_train, y_train)

malignant_probs = probs[:,1]
fpr, tpr, thresholds = roc_curve(y_test, malignant_probs)
roc_auc = auc(fpr, tpr)

plt.title('SVM Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'y', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


#5.DT

dec = DecisionTreeClassifier()
dec.fit(X_train, y_train)
probs = dec.predict_proba(X_test)

malignant_probs = probs[:,1]
fpr, tpr, thresholds = roc_curve(y_test, malignant_probs)
roc_auc = auc(fpr, tpr)

plt.title('DT Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'y', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

