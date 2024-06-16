#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


# In[2]:


data = pd.read_csv("features.csv")


# In[3]:


data = data.dropna(subset = ['track'])
data = data.fillna(0.00)


# In[4]:


X = data.iloc[1:,:-1]
Y = data.iloc[1:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print(X.head())
Y.head()


# In[ ]:


mlp = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
              beta_1=0.5, beta_2=0.5, early_stopping=True,
              epsilon=1e-08, hidden_layer_sizes=(530,500,450,400,350,300,250,200),
              learning_rate='adaptive', learning_rate_init=0.01,
              max_iter=500, momentum=0.75, n_iter_no_change=50,
              nesterovs_momentum=True, power_t=0.5, random_state=27,
              shuffle=True, solver='adam', tol=0.001,
              validation_fraction=0.1, verbose=True, warm_start=False)

mlp1 = mlp.fit(X_train, y_train)
mlp_pred = mlp1.predict(X_test)
mlp_score = accuracy_score(y_test, mlp_pred)


# In[ ]:


print(mlp_score)


# In[ ]:


svc = SVC(C=0.8, cache_size=200, class_weight=None, coef0=0.5,
    decision_function_shape='ovr', degree=5, gamma='auto', kernel='rbf',
    max_iter=500, probability=False, random_state= 19, shrinking=True,
    tol=0.001, verbose=True)

svc1 = svc.fit(X_train, y_train)
svc_pred = svc1.predict(X_test)
svc_score = accuracy_score(y_test,svc_pred)


# In[ ]:


print(svc_score)


# In[5]:


from sklearn.ensemble import RandomForestClassifier

tree = RandomForestClassifier(max_depth=300, n_estimators = 400, random_state=19)
tree1 = tree.fit(X_train, y_train)
tree_pred = tree1.predict(X_test)
tree_score = accuracy_score(y_test,tree_pred)


# In[6]:


print(tree_score)


# In[ ]:


# gnb = GaussianNB()
# gnb1 = gnb.fit(X_train, y_train)
# gnb_pred = gnb1.predict(X_test)
# gnb_score = accuracy_score(y_test,gnb_pred)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn1 = knn.fit(X_train, y_train)
knn_pred = knn1.predict(X_test)
knn_score = accuracy_score(y_test,knn_pred)


# In[ ]:


print(knn_score)

