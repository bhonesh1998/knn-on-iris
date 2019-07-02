#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd


# In[74]:


data=pd.read_csv("Iris.csv")


# In[75]:


# 150 observations
# 4 features - sepal length, sepal width, petal length, petal width 
# Response variable is the iris species
# Classification problem since response is categorical 


# In[76]:


data.head(5)


# In[77]:


from sklearn.model_selection import train_test_split


# In[78]:


X=data.drop(['Species'],axis=1)
y=data['Species']


# In[79]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=4)


# In[80]:


#import the KNeighborsClassifier class from sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
k_range = range(1,50) # checking accuracy for a range ok k so that we can choose best model 
scores = {}
scores_list = []
for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        y_pred=knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test,y_pred)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))


# In[81]:


scores


# In[82]:


import matplotlib.pyplot as plt
#plot the relationship between K and the accuracy
plt.plot(k_range,scores_list)
plt.xlabel('Values of K')
plt.ylabel('Accuracy')


# In[83]:


# we can choose value of k from 3 to 27 
# i am choosing k = 5 for the model
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)


# In[84]:


# predict for new data
new_data=[[3,4,5,6],[7,8,9,10],[1,3.4,5.6,7.8],[3,4,5,2],[5,4,2,2],[3, 2, 4, 0.2], [  4.7, 3, 1.3, 0.2 ]]
new_predict=knn.predict(new_data)
new_predict


# In[ ]:




