#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from  nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


# In[2]:


data = pd.read_csv("sentiment.csv")
data = data.dropna(subset = ['genre'])
data3 = pd.read_csv("features2.csv")
data = data.replace(np.nan, '', regex=True)
data2 = data['track'] + ' ' + data['album']
data3 = data3.dropna(subset = ['track'])
data3 = data3.fillna(0.00)


# In[ ]:


sentiment_list = []
sid_obj = SentimentIntensityAnalyzer() 
for track in data2:
    sentiment_dict = sid_obj.polarity_scores(track)
    sentiment_list.append(sentiment_dict['compound'])
m = interp1d([-1, 1],[0, 2])


# In[ ]:


scaled_sent_list = []
for i, sen in enumerate(sentiment_list):
    scaled_sent_list.append(float(m(sen)))


# In[ ]:


data3["sent_score"] = scaled_sent_list


# In[ ]:


X1 = data3.iloc[:,:-2]
X1["sent_score"] = data3.iloc[:,-1]
Y = data3.iloc[:,-2]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X1, Y, test_size=0.33, random_state=42)

mlp = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
              beta_1=0.5, beta_2=0.5, early_stopping=True,
              epsilon=1e-08, hidden_layer_sizes=(530,500,400,300,200),
              learning_rate='adaptive', learning_rate_init=0.01,
              max_iter=500, momentum=0.75, n_iter_no_change=50,
              nesterovs_momentum=True, power_t=0.5, random_state=27,
              shuffle=True, solver='adam', tol=0.001,
              validation_fraction=0.1, verbose=True, warm_start=False)

mlp1 = mlp.fit(X_train, y_train)
mlp_pred = mlp1.predict(X_test)
mlp_score = accuracy_score(y_test, mlp_pred)

svc = SVC(C=0.8, cache_size=200, class_weight=None, coef0=0.5,
    decision_function_shape='ovr', degree=5, gamma='auto', kernel='rbf',
    max_iter=500, probability=False, random_state= 19, shrinking=True,
    tol=0.001, verbose=True)

svc1 = svc.fit(X_train, y_train)
svc_pred = svc1.predict(X_test)
svc_score = accuracy_score(y_test,svc_pred)



tree = RandomForestClassifier(max_depth=150, random_state=19)
tree1 = tree.fit(X_train, y_train)
tree_pred = tree1.predict(X_test)
tree_score = accuracy_score(y_test,tree_pred)

gnb = GaussianNB()
gnb1 = gnb.fit(X_train, y_train)
gnb_pred = gnb1.predict(X_test)
gnb_score = accuracy_score(y_test,gnb_pred)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn1 = knn.fit(X_train, y_train)
knn_pred = knn1.predict(X_test)
knn_score = accuracy_score(y_test,knn_pred)


print("MLP score = ",mlp_score)
print("Random Forest Score = ",tree_score)
print("Gaussian Naive Bayes Score = ",gnb_score)
print("KNN score = ",knn_score)
print("SVM_score = ", svc_score)

