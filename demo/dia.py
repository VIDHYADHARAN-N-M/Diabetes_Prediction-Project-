import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")



diabetes_df =pd.read_csv('C:/Users/VIDHYADHARAN N M/Desktop/DIABET/diabetes.csv')

X=diabetes_df.drop('Outcome',axis=1)
Y=diabetes_df['Outcome']

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test =train_test_split(X ,Y ,test_size=0.20)
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

RF=RandomForestClassifier()

RF.fit(X_train ,Y_train)

rfc_pred = RF.predict(X_train)

pickle.dump(RF,open('dia.pkl','wb'))

