#!/user/bin/python3

import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plot
from sklearn import linear_model


report=pd.read_csv('/home/students/s441405/Desktop/myFork/umz-template/zajecia1/zadanie3/train/train.tsv', sep='\t', names=['price', 'isNew', 'rooms', 'floor', 'location', 'sqrMeters'])
reg=linear_model.LinearRegression()
reg.fit(pd.DataFrame(report, columns=['rooms','floor', 'sqrMeters']), report['price'])
print(reg.coef_)
print(reg.intercept_)

report2=pd.read_csv('/home/students/s441405/Desktop/myFork/umz-template/zajecia1/zadanie3/dev-0/in.tsv', sep='\t', names=['isNew', 'rooms', 'floor', 'location', 'sqrMeters'])
x_dev=pd.DataFrame(report2, columns=['rooms','floor', 'sqrMeters'])
y_dev_predict=reg.predict(x_dev)
pd.DataFrame(y_dev_predict).to_csv('/home/students/s441405/Desktop/myFork/umz-template/zajecia1/zadanie3/dev-0/out.tsv', sep='\t', index=False, header=False)

report3=pd.read_csv('/home/students/s441405/Desktop/myFork/umz-template/zajecia1/zadanie3/test-A/in.tsv', sep='\t', names=['isNew', 'rooms', 'floor', 'location','sqrMeters']) 
x_test=pd.DataFrame(report3, columns=['rooms', 'floor', 'sqrMeters'])
y_test_predict=reg.predict(x_test)
pd.DataFrame(y_test_predict).to_csv('/home/students/s441405/Desktop/myFork/umz-template/zajecia1/zadanie3/test-A/out.tsv', sep='\t', index=False, header=False)

sb.regplot(y=report["floor"], x=report["price"]); plot.show()
sb.regplot(y=report["price"], x=report["rooms"]); plot.show()
