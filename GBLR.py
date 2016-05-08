'''Gradient Boosted Logistic Regression '''
def MCCR(predicted, expected, C1=-1, C2=1):
    c1_hit, c1_miss, c2_hit, c2_miss = 0,0,0,0
    TP,TN,FP,FN = 0,0,0,0
    for (p, e) in zip(predicted, expected):
        if p == e:
            if e == C1:
                 c1_hit += 1
                 TP += 1
            else:
                 c2_hit += 1
                 TN += 1
        else:
             if e == C1:
                 c1_miss += 1
                 FN += 1
             else:
                 c2_miss += 1
                 FP += 1
             CCR1 = float(c1_hit) / (c1_hit+c1_miss)
             CCR2 = float(c2_hit) / (c2_hit+c2_miss)
             TPR = float(TP)/(TP+FN)
             FPR = float(FP)/(FP+TN)
             MCCR = CCR1 if CCR1 < CCR2 else CCR2
    return CCR1, CCR2, MCCR, TPR, FPR
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor,AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
from sklearn.decomposition import PCA,TruncatedSVD
print "Loading data.."
cnt = 0
unsatisfiedCnt = 0
satisfiedCnt = 0
trainds = [ ]
trainlabels = [ ]
testds = [ ]
testlabels = [ ]
with open('train.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
    	if cnt == 0:
            cnt = cnt+1
            continue
        f = float(row[len(row)-1])
        l = [float(x) for x in row[:len(row)-1]]
    	if cnt <= 15000:
            testds.append(l)
            testlabels.append(f)
    	else:
            trainds.append(l)
            trainlabels.append(f)
        cnt = cnt+1

print "Finished loading data..."
dtrain = xgb.DMatrix(trainds,trainlabels)
dtest = xgb.DMatrix(testds,testlabels)
dtestLabels = dtest.get_label();
param = {'max_depth': 5,'gamma ':0.8,'booster':'gbtree','max_delta_step':10,'alpha':0.4,'lambda':0.4, 'eta': 0.02, 'silent': 1, 'objective': 'binary:logistic', 'eval_metric':'auc'}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 5
#clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242).fit(dtrain,trainlabels)
bst = xgb.train(param, dtrain, num_round,watchlist)

# this is prediction
preds = bst.predict(dtest)
labels = dtest.get_label()
print "error: " + str((sum(1 for i in range(len(preds)) if int(preds[i] >= 0.5) != labels[i])/ float(len(preds))));
x,y,thresholds =roc_curve(testlabels,preds,1)
plt.figure()
plt.plot(x,y)
plt.show()

print (auc(x,y))
predicted = preds > 0.5
CCR1, CCR2, MCCR, TPR, FPR= MCCR(predicted,testlabels,0,1);
print(CCR1)
print CCR2
print "MCCR: "+str(MCCR*100)
