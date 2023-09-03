#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[3]:


df = pd.read_csv ("creditcard.csv")
df.tail(50)


# In[4]:


plt.figure(figsize = (40,10))
sns.heatmap(df.corr(), annot = True, cmap="tab20c")
plt.show()


# In[5]:


X= df.iloc[ : , : -1].values
y = df.iloc [: , -1].values


# In[6]:


from sklearn.preprocessing import StandardScaler 
SD = StandardScaler ()
X = SD.fit_transform(X)


# In[7]:


from sklearn.model_selection import train_test_split 
x_train , x_test ,y_train,y_test = train_test_split (X,y,test_size=0.25, random_state= 5 )
print (y_train)


# In[8]:


from sklearn.svm import SVC 
classifier = SVC(C=0.3,kernel = 'rbf', random_state =5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)


# In[10]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[9]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
print (accuracy_score(y_test, y_pred))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");


# In[ ]:




