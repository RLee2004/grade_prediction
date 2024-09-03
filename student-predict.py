import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("student-mat.csv", sep=";")

print(data.head())

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'famrel', 'goout', 'health', 'absences']]
print(data.head())

predict = 'G3'

x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# target_acc = 0.96
# acc = 0
# while(acc < target_acc):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

#     linear = linear_model.LinearRegression()

#     linear.fit(x_train,y_train)
#     acc = linear.score(x_test,y_test)

# with open("studentmodel.pickle", "wb") as f:
#     pickle.dump(linear,f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

acc = linear.score(x_test,y_test)
print("Accuracy: ", acc)
print("Co: ", linear.coef_)
print("Intercept: ", linear.intercept_)


prediction = linear.predict(x_test)

# for i in range(len(prediction)):
#     print(prediction[i], x_test[i], y_test[i])

for n in data.columns:
	pyplot.figure()
	pyplot.scatter(data[n], data["G3"])
	pyplot.xlabel(n)
	pyplot.ylabel("Final Grade")
	
pyplot.show()
