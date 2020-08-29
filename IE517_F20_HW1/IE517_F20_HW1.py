#Our first machine learning model
#uses Treasury Squeeze and SGD classifier \LogisticRegression
import sklearn
print( 'The scikit learn version is {}.'.format(sklearn.__version__))

import pandas as pd
import numpy as np
path= 'C:\\Users\Administrator\\Desktop\\IE517_machine learning\\'
data=pd.read_csv(path+ 'Treasury Squeeze test - DS1.csv',index_col='rowindex')
data.head()


from sklearn.model_selection  import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
y=np.array(data['squeeze'])
X=data.iloc[:,list(range(1,10))]
print( X.shape, y.shape)
# Split the dataset into a training and a testing set
# Test set will be the 25% taken randomly
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=33)
print( X_train.shape, y_train.shape)
# Standardize the features
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import matplotlib.pyplot as plt
squeeze = ['True', 'False']
colors=['red', 'greenyellow']
for i in range(len(squeeze)):
    print(i)
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, c=colors[i])
plt.legend(squeeze)
plt.title('price_crossing vs price_distortion')
plt.xlabel('price_crossing')
plt.ylabel('price_distortion')

###SGDClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
clf = SGDClassifier()
clf.fit(X_train, y_train)
print('clf.coef is:'+ str(clf.coef_))
print( 'clf.intercept is:'+ str(clf.intercept_))

print( clf.predict([[1,1,1,0,1,1,0,1,1]]) )
print( clf.decision_function([[1,1,1,0,1,1,0,1,1]] ))

from sklearn import metrics
y_train_pred = clf.predict(X_train)
print( metrics.accuracy_score(y_train, y_train_pred) )


y_pred = clf.predict(X_test)
print( metrics.accuracy_score(y_test, y_pred) )

print( metrics.classification_report(y_test, y_pred, target_names=['True','False']) )
print( metrics.confusion_matrix(y_test, y_pred) )


###LogisticRegression
clf2 = LogisticRegression(penalty='l2')
clf2.fit(X_train, y_train)


print( clf2.predict([[1,1,1,0,1,1,0,1,1]]) )
print( clf2.decision_function([[1,1,1,0,1,1,0,1,1]] ))

from sklearn import metrics
y_train_pred = clf2.predict(X_train)
print( metrics.accuracy_score(y_train, y_train_pred) )


y_pred = clf2.predict(X_test)
print( metrics.accuracy_score(y_test, y_pred) )

print( metrics.classification_report(y_test, y_pred, target_names=['TRUE','FALSE']) )

accuracy = metrics.accuracy_score(y_test, y_pred)
print ('accuracy: %.2f%%' % (100 * accuracy))


from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
# create a composite estimator made by a pipeline of the standarization and the linear model
clf3 = Pipeline([(
        'scaler', StandardScaler()),
        ('LR_model', LogisticRegression())
])
# create a k-fold cross validation iterator of k=5 folds
cv = KFold( 5, shuffle=False, random_state=33)
# by default the score used is the one returned by score method of the estimator (accuracy)
scores = cross_val_score(clf3, X, y, cv=cv)
print( scores )

from scipy.stats import sem
def mean_score(scores): return ("Mean score: {0:.3f} (+/- {1:.3f})").format(np.mean(scores), sem(scores))
print( mean_score(scores) )


print("My name is Zhiyi Ye")
print("My NetID is: zhiyiye2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
######STOP HERE######################