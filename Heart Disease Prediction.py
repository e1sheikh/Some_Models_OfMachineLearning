#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 


# In[3]:


#reading data 
data = pd.read_csv ("Heart_Disease_Prediction.csv")
X = data.iloc[ : ,  : -1 ].values 
y = data.iloc[ : , -1  ].values


# In[4]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y = lb.fit_transform(y)
print (y)


# In[5]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=1)
print (X_train.shape)
print (y_train.shape)


# In[6]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[7]:


# Fitting SVC to the Training set
from sklearn.svm import SVC
Cclassifier = SVC(C=0.3,kernel = 'rbf', random_state =1)
Cclassifier.fit(X_train, y_train)


# In[8]:


# Predicting the Test set results
y_pred = Cclassifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
print (accuracy_score(y_test, y_pred))
print (confusion_matrix(y_test, y_pred))


# In[9]:


from sklearn.linear_model import LogisticRegression
log=LogisticRegression(random_state=1)
log.fit(X_train,y_train)


# In[10]:


# Predicting the Test set results
y_pred = log.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
print (accuracy_score(y_test, y_pred))
print (confusion_matrix(y_test, y_pred))


# In[11]:


#fiting rondom forest classification 
from sklearn.ensemble import RandomForestClassifier
rclassifier = RandomForestClassifier(n_estimators = 8, criterion = 'entropy', random_state = 1)
rclassifier.fit(X_train, y_train)


# In[12]:


# Predicting the Test set results
y_pred = rclassifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
print (accuracy_score(y_test, y_pred))
print (confusion_matrix(y_test, y_pred))


# In[13]:


from sklearn.naive_bayes import GaussianNB
nvclassifier = GaussianNB()
nvclassifier.fit(X_train, y_train)


# In[14]:


# Predicting the Test set results
y_pred = nvclassifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
print (accuracy_score(y_test, y_pred))
print (confusion_matrix(y_test, y_pred))


# In[15]:


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors= 11,weights ='uniform',algorithm ='auto')
KNN.fit(X_train, y_train)


# In[16]:


# Predicting the Test set results
y_pred = KNN.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
print (accuracy_score(y_test, y_pred))
print (confusion_matrix(y_test, y_pred))


# In[17]:


from sklearn.cluster import KMeans
KMeansModel = KMeans(n_clusters=1,init='k-means++', #also can be random
                     random_state=1,algorithm= 'auto') # also can be full or elkan
KMeansModel.fit(X_train)


# In[18]:


# Predicting the Test set results
y_pred = KMeansModel.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
print (accuracy_score(y_test, y_pred))
print (confusion_matrix(y_test, y_pred))


# In[19]:


from sklearn.tree import DecisionTreeClassifier
dclassifier = DecisionTreeClassifier(criterion = 'entropy')
dclassifier.fit(X_train, y_train)


# In[20]:


# Predicting the Test set results
y_pred = dclassifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
print (accuracy_score(y_test, y_pred))
print (confusion_matrix(y_test, y_pred))


# In[21]:


from sklearn.linear_model import LinearRegression
lb = LinearRegression ()
lb.fit(X_train, y_train)


# In[22]:


y_pred = dclassifier.predict(X_test)
from sklearn.metrics import mean_squared_error
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

