import pandas as pd
import graphviz
from sklearn.ensemble import RandomForestClassifier #losowe drzewo
from sklearn.metrics import confusion_matrix

train = pd.read_csv('train.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')

train_X = pd.DataFrame(train, columns=train.columns[:-1])
train_Y = train['Class']

test_X = pd.DataFrame(test, columns=test.columns[:-1])
test_Y = test['Class']

clf = RandomForestClassifier()
clf = clf.fit(train_X, train_Y)
print('TEST SET')
#print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))
print('TRAIN SET')
#print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))


clf = RandomForestClassifier(max_depth=4)
clf = clf.fit(train_X, train_Y)

print('TEST SET MAX DEPTH 4')
#print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))
print('TRAIN SET MAX DEPTH 4')
#print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))

clf = RandomForestClassifier(n_estimators=30,oob_score=True)
clf = clf.fit(train_X, train_Y)

print('TEST SET - MY OWN')
#print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))
print('TRAIN SET - MY OWN')
#print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
