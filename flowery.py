import matplotlib.pyplot as plt

import pandas
from pandas.plotting import scatter_matrix

from sklearn import model_selection

from sklearn.naive_bayes import GaussianNB

url="C:/Users/ekmuriuki/Documents/iris.csv"
names= ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url,names=names)

print(dataset.shape)
print(dataset.head(5))
print(dataset)
print(dataset.describe())
print(dataset.groupby('class').size())

#visualization
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

dataset.hist()
plt.show()

scatter_matrix(dataset)
plt.show()

#Split-out validation dataset
array = dataset.values
x = array[:,0:4]
y = array[:,4]
validation_size = 0.20
seed = 7
x_train, x_test, y_train, y_test = model_selection.train_test_split (x,y, test_size=validation_size, random_state=seed)

print("x_train",x_train)
print("x_test",x_test)
print("Y_train",y_train)
print("y_test",y_test)

url="C:/User/ekmuriuki/Documents/iris_test.csv"
names = ['sepal-length','sepal-width','petal-length','petal-width','class']
