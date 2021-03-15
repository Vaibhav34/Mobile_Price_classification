#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Importing DataSets

# In[2]:


data_train=pd.read_csv(r'C:\Users\vaibhav\Project\mobile_price\train.csv')


# In[3]:


data_train


# In[4]:


data_test=pd.read_csv(r'C:\Users\vaibhav\Project\mobile_price\test.csv')


# In[5]:


data_test


# In[6]:


data_test=data_test.drop('id',axis=1)


# In[7]:


data_test


# In[ ]:





# # Exploratory Data Analysis

# In[9]:


data_train.info()


# In[10]:


data_train.shape


# In[11]:


data_test.shape


# In[12]:


data_train.describe()


# In[13]:


data_train.plot(x='price_range',y='ram',kind='scatter')


# In[14]:


data_train.plot(x='price_range',y='battery_power',kind='scatter')


# In[15]:


data_train.plot(kind='box',figsize=(20,10))
plt.show()


# # Filtering For train set

# In[16]:


x=data_train.drop('price_range',axis=1)


# In[17]:


x


# In[18]:


y=data_train['price_range']


# In[19]:


y


# # Scaling Data

# In[20]:


from sklearn.preprocessing import StandardScaler
std=StandardScaler()


# In[21]:


x_std=std.fit_transform(x)


# In[22]:


data_test_std=std.transform(data_test)


# In[23]:


data_test_std


# In[156]:


x_std


# # Training Model

# # 1.Decision tree

# In[27]:


from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier()


# In[28]:


dt.fit(x_std,y)


# In[29]:


dt.predict(data_test_std)


# In[30]:


dt.score(x_std,y)


# # 2.KNN

# In[32]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[33]:


knn.fit(x_std,y)


# In[34]:


knn.predict(data_test_std)


# In[35]:


data_test


# # 3.Logistic Regression

# In[37]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[38]:


lr.fit(x_std,y)


# In[39]:


lr.predict(data_test_std)


# In[ ]:





# # Applying Train_Test_Split

# In[94]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=1)


# In[95]:


x_train


# In[96]:


y_train


# In[97]:


x_test


# In[98]:


y_test


# # 1.Decision Tree

# In[99]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[100]:


dt.fit(x_train,y_train)


# In[101]:


y_pred=dt.predict(x_test)


# In[102]:


y_test


# In[ ]:





# In[ ]:





# In[103]:


from sklearn.metrics import accuracy_score
ac_dt=accuracy_score(y_test,y_pred)


# In[104]:


ac_dt


# # 2.KNN

# In[106]:


x_train_std=std.fit_transform(x_train)

x_test_std=std.transform(x_test)


# In[107]:


x_test_std


# In[108]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[109]:


knn.fit(x_train_std,y_train)


# In[110]:


y_pred=knn.predict(x_test_std)


# In[111]:


y_pred


# In[112]:


y_test


# In[113]:


ac_knn=accuracy_score(y_test,y_pred)


# In[114]:


ac_knn


# # 3.Logistic Regression

# In[116]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[117]:


lr.fit(x_train_std,y_train)


# In[118]:


y_pred=lr.predict(x_test_std)


# In[119]:


y_pred


# In[120]:


y_test


# In[121]:


lr_ac=accuracy_score(y_test,y_pred)


# In[122]:


lr_ac


# In[126]:


plt.bar(x=['dt','knn','lr'],height=[ac_dt,ac_knn,lr_ac])
plt.xlabel('Algorithm')
plt.ylabel('Acuracy')
plt.show()


# # 4.SVM

# In[128]:


from sklearn.svm import SVC


# In[129]:


sv=SVC()


# In[130]:


sv.fit(x_train_std,y_train)


# In[137]:


y_pred=sv.predict(x_test_std)


# In[138]:


y_pred.shape


# In[139]:


y_test.shape


# In[140]:


ac_sv=accuracy_score(y_test,y_pred)


# In[141]:


ac_sv


# # 5.Random Forest

# In[142]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()


# In[143]:


rf.fit(x_train_std,y_train)


# In[144]:


y_pred=rf.predict(x_test_std)


# In[145]:


y_pred


# In[146]:


ac_rf=accuracy_score(y_test,y_pred)


# In[147]:


ac_rf


# # Comparing All Models With Their Accuracy Score

# In[155]:


plt.bar(x=['dt','knn','lr','svm','rf'],height=[ac_dt,ac_knn,lr_ac,ac_sv,ac_rf])
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:




