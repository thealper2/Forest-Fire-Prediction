#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("fire_archive.csv")


# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# In[5]:


sns.heatmap(df.corr(), annot=True, cmap="viridis")


# # Preprocess

# In[6]:


df = df.drop(["instrument", "version", "track"], axis=1)


# In[7]:


df["satellite"].value_counts()


# In[8]:


df["daynight"].value_counts()


# In[9]:


df["type"].value_counts()


# In[10]:


le = LabelEncoder().fit(df["satellite"])
le_nm = dict(zip(le.classes_, le.transform(le.classes_)))
df["satellite"] = df["satellite"].apply(lambda x: le_nm[x])


# In[11]:


le_nm


# In[12]:


le2 = LabelEncoder().fit(df["daynight"])
le_nm2 = dict(zip(le2.classes_, le2.transform(le2.classes_)))
df["daynight"] = df["daynight"].apply(lambda x: le_nm2[x])


# In[13]:


le_nm2


# In[14]:


types = pd.get_dummies(df["type"])
types.head()


# In[15]:


types = types.rename(columns={0: "Type0", 2: "Type2", 3: "Type3"})


# In[16]:


df = pd.concat([df, types], axis=1)


# In[17]:


df = df.drop(["type", "Type0"], axis=1)


# In[18]:


df["scan"].value_counts()


# In[19]:


df["scan"] = pd.cut(df["scan"], bins=[0,1,2,3,4,5], labels=[1,2,3,4,5])


# In[20]:


df["acq_date"] = pd.to_datetime(df["acq_date"])
df["day"] = df["acq_date"].dt.day
df["month"] = df["acq_date"].dt.month
df["year"] = df["acq_date"].dt.year


# In[21]:


df = df.drop(["acq_date", "acq_time", "bright_t31"], axis=1)


# In[22]:


X = df.drop(["confidence"], axis=1)
y = df["confidence"]


# In[23]:


sns.heatmap(X.corr(), annot=True, cmap="viridis")


# In[24]:


X.shape


# In[25]:


y.shape


# # Model Training

# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X.iloc[:, :1000], y, test_size=0.2, random_state=4242)


# In[29]:


rf_model = RandomForestRegressor().fit(X_train, y_train)


# In[33]:


rf_model.get_params()


# In[30]:


y_pred = rf_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# In[31]:


acc = round(rf_model.score(X_test, y_test) * 100, 2)
acc


# # Model Tuning

# In[ ]:


rf_params = {'max_depth': list(range(10, 20)), 
             'max_features': [5, 10, 20], 
             'n_estimators': [100, 250, 500]}
rf_model = RandomForestRegressor()
rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1).fit(X_train, y_train)
rf_cv_model.best_params_


# In[35]:


rf_tuned = RandomForestRegressor(max_depth=8, max_features=5, n_estimators=100).fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# In[36]:


cc = round(rf_tuned.score(X_test, y_test) * 100, 2)
acc


# In[37]:


model = pickle.dump(rf_tuned, open("RandomForest.pickle", "wb"))


# In[39]:


y_test.shape, y_pred.shape

