
def MCCR(predicted, expected, C1=-1, C2=1):
    c1_hit, c1_miss, c2_hit, c2_miss = 0,0,0,0
    CCR1,CCR2,MCCR=0,0,0
    for (p, e) in zip(predicted, expected):
        if p == e:
            if e == C1:
                 c1_hit += 1
            else:
                 c2_hit += 1
        else:
             if e == C1:
                 c1_miss += 1
             else:
                 c2_miss += 1
             if(c1_hit+c1_miss!=0): CCR1 = float(c1_hit) / (c1_hit+c1_miss)
             if(c2_hit+c2_miss!=0): CCR2 = float(c2_hit) / (c2_hit+c2_miss)
             MCCR = CCR1 if CCR1 < CCR2 else CCR2
    return CCR1, CCR2, MCCR

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import ensemble
import xgboost as xgb
from sklearn.decomposition import FastICA
from sklearn.linear_model import LogisticRegression 

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
# load data
df_train = pd.read_csv('train.csv')

# remove constant columns
remove = []
for col in df_train.columns:
    if df_train[col].std() == 0:
        remove.append(col)

df_train.drop(remove, axis=1, inplace=True)


# remove duplicated columns
remove = []
c = df_train.columns
for i in range(len(c)-1):
    v = df_train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,df_train[c[j]].values):
            remove.append(c[j])

df_train.drop(remove, axis=1, inplace=True)

y_train = df_train['TARGET'].values
X_train = df_train.drop(['ID','TARGET'], axis=1).values



print "Preproccessing done!"
X_test = X_train[:15000]
y_test = y_train[:15000]
X_train = X_train[15001:]
y_train = y_train[15001:]
print(len(X_train))

dtrain = xgb.DMatrix(X_train,y_train)
dtest = xgb.DMatrix(X_test,y_test)
'''
# classifier
param = {'base_score':0.5,'max_depth': 15,'gamma ':0.8,'booster':'gbtree','max_delta_step':10,'alpha':0.4,'lambda':0.4, 'eta': 0.015, 'silent': 1, 'objective': 'binary:logistic', 'eval_metric':'auc', 'nthread':16 }
watchlist = [(dtrain, 'eval'), (dtrain, 'train')]
num_round = 200
bst = xgb.train(param, dtrain, num_round,watchlist,early_stopping_rounds=25)
preds = bst.predict(dtest)
'''
####################################
# Reduced data for faster performance on other classifiers.

pcaTrain = FastICA(n_components=250, algorithm='parallel').fit_transform(X_train)
pcaTest = FastICA(n_components=250, algorithm='parallel').fit_transform(X_test)


####################################
print("reduction done")

clf_l1_LR = LogisticRegression()

rfpreds = clf_l1_LR.fit(pcaTrain,y_train).predict_proba(pcaTest)



print("Random forest done")
###################################

x,y,thresholds =roc_curve(y_test,rfpreds[:,1],1)
plt.figure()
plt.plot(x,y)
plt.show()

print (auc(x,y))
bestMCCR =0
for threshold in thresholds:
    predicted = rfpreds[:,1] > threshold
    CCR1, CCR2, mCCR = MCCR(predicted,y_test,0,1);
    bestMCCR = max(bestMCCR,mCCR)
print(bestMCCR)
