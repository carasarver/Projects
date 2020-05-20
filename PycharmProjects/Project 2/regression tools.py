import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Generate X values and Y values for f(x) = 2x + 3 + e, where e represents random error

data = []
def pop_regression(n):
    for i in range(n):
        epsilon = np.random.normal(0,10)
        d = {'Xvalue': i, 'Yvalue': 2*i + 3 + epsilon}
        data.append(d)
    df = pd.DataFrame(data)
    return df

sim_100 = pop_regression(100)           # Generate dataset for X = [0,100]

print('\n A look at the data: \n')
print(sim_100.sample(5), '\n')

X = sim_100['Xvalue'].values.reshape(-1,1)                  # Split data into attributes and labels
Y = sim_100['Yvalue'].values.reshape(-1,1)

regressor = LinearRegression()                                 # fit the data to a linear regresssion model
regressor.fit(X,Y)
print('Regression model results:')
print('The intercept is', regressor.intercept_)
print('The regression coefficient is', regressor.coef_, '\n')

m, b = np.polyfit(sim_100['Xvalue'], sim_100['Yvalue'], 1)      # simple alternative to regression fit
print('Numpy polyfit results:')
print('The intercept is', b)
print('The slope is', m)

plt.scatter(sim_100['Xvalue'], sim_100['Yvalue'], facecolors='none', edgecolors='k')      # scatter plot of X vs Y data points
plt.plot(sim_100['Xvalue'], m*sim_100['Xvalue'] + b, 'k')               # plot line of best fit
plt.show()


