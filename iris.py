from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
iris=load_iris()
#print(iris.keys())
#print(iris)
features=iris.data.T #taking transpose of all the data i.e. sepal length width petal length, width it will seperate all the four fields
sepal_length=features[0]
sepal_width=features[1]
petal_length=features[2]
petal_width=features[3]
#print(iris.feature_names)
sepal_length_label=iris.feature_names[0]
sepal_width_label=iris.feature_names[1]
petal_length_label=iris.feature_names[2]
petal_width_label=iris.feature_names[3]
colors=['red','yellow','purple']
# plt.scatter(sepal_length, sepal_width, c=iris.target) ## iris.target is expected class of given sepal length and sepal width
# plt.xlabel(sepal_length_label)
# plt.ylabel(sepal_width_label)
# plt.show()
#-----------------------------
# iris meri dictionary hai uski key data hai jisme saare features hai same as target
#X is data and y is targets i.e. classes of flower
X_train,X_test,y_train,y_test=train_test_split(iris['data'],iris['target'],random_state=0,test_size=0.25)
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train) ## dono training set hai
x_new=np.array(X_test)
#x_new=np.array([[1.2,2.9,8.9,2.0]])
prediction=knn.predict(x_new)
print(prediction)
score=knn.score(X_test,y_test) # this will check that how much the values returned from X_test is similar to y_test
# x_test data se jo predict kr k and. aaye hai vo y_test k kitne similar hai
print(score)