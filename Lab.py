import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from scipy import stats

data=pd.read_csv('./wineQualityReds.csv')
# print(data.head())
# print(data.describe())
# print(data.isna().sum())

plt.figure(figsize=(30,20))
corr=data.corr()
sns.heatmap(corr,annot=True)
# plt.show()


from scipy import stats
z=np.abs(stats.zscore(data))
# print(z)
# print(np.where(z>3))
dataset=data[(z<3).all(axis=1)]
# print(dataset.shape)


from sklearn.model_selection import train_test_split
x=dataset.drop(columns='quality')
y=dataset['quality']
# print(x.head())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
# print(y_pred)

from sklearn import metrics
print('Accuracy score:',metrics.accuracy_score(y_test,y_pred))

from sklearn import tree
plt.figure(figsize=(20,20))
tree.plot_tree(rf.estimators_[0],filled=True)
# for i in range(len(rf.estimators_)):
#     tree.plot_tree(rf.estimators_[i],filled=True)