#Importing libraries
import pandas as pd #For accessing dataset
import numpy as np
import matplotlib.pyplot as plt #For Graph

dataset = pd.read_csv('SindhDistrict.csv') #Reading from folder where csv file is
print(dataset.shape) #Tells how many rows and columns are in the dataset.

#Plotting data points on a graph
#Manual checking if we can find relationship between the data.
#dataset.plot(x='Year', y='Population', style='o')
#plt.title('Pakistan Population')
#plt.xlabel('Year', 'City')
#plt.ylabel('Population')
#plt.plot(dataset.Year,dataset.City, dataset.Population, color='red', marker='+')
#plt.show()#The grap shows, there is a linear relation between Year and Population.

#Graph with seaborn
import seaborn as sns
#sns.regplot(x="Year", y="Population", data=dataset);


#Preparing Data
X = dataset.iloc[:, :2] #Year and District
y = dataset.iloc[:, -1] #Population


#Splitting the dataset, 20% for test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#Training the algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Multivariable Linear regression, has to find the most optimal coefficients for all the attributes
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

#Making Predictions 
#The y_pred is a numpy array containing 
#predicted values for the input values in the X_test set.
y_pred = regressor.predict(X_test)

#Comparing actual output values for X_test with the predicted values for y_pred
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)


#Evaluation of the Algorithm 
from sklearn.metrics import r2_score
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R Square Score:',r2_score(y_test, y_pred))


print(regressor.predict([[2020, 110]]))


import pickle
pickle.dump(regressor, open('model.pkl','wb'))


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2020,110]]))
