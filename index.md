# Lab 01

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
data = pd.read_csv("../../Data310/Sample_Data/L1Data.csv")

**Q1**
Since rain amount is determined in volume this can take on any unit. From 1 L to 20 mL etc. This makes it interval. Additionally there is no true 0. 

**Q2**
The Data is represented as 21 as shown below. Examples include Sophmore 12 and Sophmore 20.

imputing_configuration = SimpleImputer(missing_values = np.nan, strategy = 'median')
imp = imputing_configuration.fit(data[["Age"]])
data[["Age"]] = imp.transform(data[["Age"]]).ravel()

data.loc[12]

**Q3**
Bayesian inference on the "likelihood" is defined in Lecture 2 as shown below:

- <font color='deepskyblue'>*Likelihood*</font>: It represents how probable is the evidence given that our hypothesis is true.

**Q4**
This is on Monte Carlo sims. Covered in 4-6. This evidence is from lecture 4.

**The main goal is to solve problems of data science by approximating probability values via carefully designed simulations.**

**Q5**
What is the probability of having a symptom given that the disease holds?

Notes on Bayes Theorem

P(A|B) = P(A and B)/P(B) 

P(B|A) = P(B and A)/P(A)

P(A and B) = P(B and A)

Therefore

P(A|B) = (P(B|A) * P(A)) / P(B)

P(Getting Infection) = 15%
P(General Respiratory Symptoms) = 35%
P(Asymptomatic Infected) = 30%

((.15)*(.35))/.3

**Q6**

The examples done in class have iterations that go from 2000 to 3000 as shown on the graphs.

**Q7**

This sounds true since in classs we have done examples were the closer the probabilitiees hugged a line, meaning that this is where the results tend to lay probability wise.

**Q8**

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

doors = ["goat","goat","goat","car"]

switch_win_probability = []
stick_win_probability = []

plt.axhline(y=0.75, color='red', linestyle='--')
plt.axhline(y=0.25, color='green', linestyle='--')

def monte_carlo(n):

  switch_wins = 0
  stick_wins = 0

  for i in range(n):
     random.shuffle(doors)

     k = random.randrange(4)

     if doors[k] != 'car':
       switch_wins +=1
    
     else:
       stick_wins +=1
    
     switch_win_probability.append(switch_wins/(i+1))
     stick_win_probability.append(stick_wins/(i+1))
    
  plt.plot(switch_win_probability,label='Switch')
  plt.plot(stick_win_probability,label='Stick')
  plt.tick_params(axis='x', colors='navy')
  plt.tick_params(axis='y', colors='navy')
  plt.xlabel('Iterations',fontsize=14,color='DeepSkyBlue')
  plt.ylabel('Probability of Winning',fontsize=14,color='green')
  plt.legend()
  print('Winning probability if you always switch:', switch_win_probability[-1])
  print('Winning probability if you always stick to your original choice:', stick_win_probability[-1])

monte_carlo(3000)


**Q9**

Determined from lecture 5-6 analysis.

**Q10**

A permutation is a way, especially one of several possible variations, in which a set or number of things can be ordered or arranged.

The awnser selected is most logical.

# Lab 02

**Q1**

An "ordinary least squares" (or OLS) model seeks to minimize the differences between your true and estimated dependent variable. **Lecture 9**

To determine the line of best fit the goal is to minimize the sum of squared residuals: 

<font color='navy'>
$$\large
\min_{m,n} \sum\limits_{i=1}^{n}(y_i-mx_i-n)^2
$$
</font>

In statistical models, a residual is the difference between the observed value and the mean value that the model predicts for that observation.

**Therefore this is true**

**Q2**

Do you agree or disagree with the following statement: In a linear regression model, all feature must correlate with the noise in order to obtain a good fit.

Noise tends to be bad, so you do not want to correlate with it. Though this is just baised from previous stats experience. **Therefore I disagree with the statement**.

**Q3**

Additional notes:

Look at lecture 10 for this problemfrom yellowbrick.regressor import ResidualsPlot

df = pd.read_csv('../../Data310/Sample_Data/L3Data.csv')
y = df['Grade'].values
#df.loc[(df['age'] == 21) & df['favorite_color'].isin(array)]
#X = df.loc[ : , df.columns != 'Grade' & df.columns != 'questions'].values
           #' & df.columns != 'questions'].values
X = df[['days online','views','contributions','answers']].values

df.head()

y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

model = LinearRegression()
model.fit(X_train, y_train)
y_poly_pred = model.predict(X_test)
MSE = mean_squared_error(y_test,y_poly_pred)
np.sqrt(MSE)
**8.324478857196398**

**Q4**
In practice we determine the weights for linear regression with the "X_test" data. **Lecture 9**

Training means running an optimization algorithm and determining the values of the weights that minimize an objective function.

The weights should be determined by the training data, or from real data. **Therefore false**

**Q5**
The goal of polynomial regression is to model a non-linear relationship between the independent and dependent variables. **Therefore true**

**Q6**
Linear regression, multiple linear regression, and polynomial regression can be all fit using LinearRegression() from the sklearn.linear_model module in Python.

**Polynomial regression**
polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(x.reshape((-1,1)))

**the model created is linear in weights**
model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)


**Multiple linear regression**
polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(x.reshape((-1,1)))

**the model created is linear in weights**
model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

**Q7**
df = pd.read_csv('../../Data310/Sample_Data/L3Data.csv')
y = df['Grade'].values
#df.loc[(df['age'] == 21) & df['favorite_color'].isin(array)]
#X = df.loc[ : , df.columns != 'Grade' & df.columns != 'questions'].values
           #' & df.columns != 'questions'].values
X = df[['days online','views','contributions','answers']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)
len(X_train)
**23**

**Q8**
The gradient descent method does not need any hyperparameters.

From lecture 9,

#hyperparamters established before the learning process begins
learning_rate = 0.05
initial_n = 20
initial_m = -2
num_iterations = 500

def gradient_descent(data, starting_b, starting_m, learning_rate, num_iterations)

Therefore **False**

**Q9**
Code from lecture 10:

x_range = np.arange(np.min(x)-2,np.max(x)+3)
yhat = lm.predict(x_range.reshape(-1,1))
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x,y,color='cyan',edgecolors='k',s=60)
ax.plot(x_range, yhat, '-',color='red',lw=2)
ax.set_xlim(-2, 15)
ax.set_ylim(-2, 20)
ax.set_xlabel('Input Feature',fontsize=20,color='navy')
ax.set_ylabel('Output',fontsize=20,color='navy')
ax.grid(b=True,which='major', color ='grey', linestyle='-', alpha=0.8)
ax.grid(b=True,which='minor', color ='grey', linestyle='--', alpha=0.2)
plt.tick_params(axis='x', colors='black',labelsize=18)
plt.tick_params(axis='y', colors='black',labelsize=18)
ax.minorticks_on()
plt.show()

1. import matplotlib.pyplot as plt

2. fig, ax = plt.subplots(figsize=(12,8))

3. ax.scatter(x,y,color='cyan',edgecolors='k',s=60)
    ax.plot(x_range, yhat, '-',color='red',lw=2)
    ax.set_xlim(-2, 15)
    ax.set_ylim(-2, 20)
    ax.set_xlabel('Input Feature',fontsize=20,color='navy')
    ax.set_ylabel('Output',fontsize=20,color='navy')

4. ax.grid(b=True,which='major', color ='grey', linestyle='-', alpha=0.8)
    ax.grid(b=True,which='minor', color ='grey', linestyle='--', alpha=0.2)
    plt.tick_params(axis='x', colors='black',labelsize=18)
    plt.tick_params(axis='y', colors='black',labelsize=18)
    ax.minorticks_on()
    
**Q10**

Which of the following forms is not  linear in the weights? From lecture 10:

<font color='navy'>Linear vs Non-linear models</font>

This is a linear model in terms of the weights $\beta$: 


$$\large
\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2 - \beta_3x_3
$$

An example for what linear in weights means
<font color='green'>
$$\large
\hat{y}(2\beta+3\alpha) = 2\hat{y}(\beta)+3\hat{y}(\alpha)
$$</font>

The following is a non-linear model in terms of the weights:


$$\large
\hat{y} = \beta_0 + \beta_1^3x_1 + \frac{1}{\beta_2+\beta_3}x_2 - e^{\beta_3}x_3
$$

<font color='magenta'>
$$\large
\hat{y}(2\beta+3\alpha) \neq 2\hat{y}(\beta)+3\hat{y}(\alpha)
$$</font>

The main point of linear regression is to assume that predictions can ben made by using a linear combination of the features. 

For example, if the data from each feature is $\begin{bmatrix}
           x_{11} \\
           x_{21} \\
           \vdots \\
           x_{n1}
         \end{bmatrix}$, $\begin{bmatrix}
           x_{12} \\
           x_{22} \\
           \vdots \\
           x_{n2}
         \end{bmatrix}$, $...
         \begin{bmatrix}
           x_{1p} \\
           x_{2p} \\
           \vdots \\
           x_{np}
         \end{bmatrix}$ then we assume that the depdendent variable is predicted by a linear combination of these columns populated with features' data. Each column represents a feature and each row an independent observation.

The predicted value is denoted by $\hat{y}$ and 


$$
\hat{y} = \beta_1\begin{bmatrix}
           x_{11} \\
           x_{21} \\
           \vdots \\
           x_{n1}
         \end{bmatrix}
         +
         \beta_2\begin{bmatrix}
           x_{11} \\
           x_{21} \\
           \vdots \\
           x_{n1}
         \end{bmatrix}
                  + ...
         \beta_p\begin{bmatrix}
           x_{1p} \\
           x_{2p} \\
           \vdots \\
           x_{np}
         \end{bmatrix}
$$

**Therefore it is the second answer.**

# Lab 03 (missed due to schedule)

# Lab 04

**Q1**

- A solution for rank defficient Multiple Linear Regression: Regularization

- Main Idea: minimize the sum of the square residuals plus a constraint on the vector of weights

**Therefore first one**

**Q2**

- This is an example of rank deficency.

- The more features there are the lower the mse for the OLS is.

- Whereas for L2, this is solved for by adding the weights. 

**Therefore true**

**Q3** 

L1 is a Euclidean norm. Therefore we need to compare the differences of these two.

The sum of the adjacent plus the opisite ends of a triangle is never less then the hypoteneus of a triangle. Another way to think of L1 is as the taxicab distance.

**Therefore false**

**Q4**

- The first option uses L1 which is commonly used for Lasso regularization, so not typical. Therefore this is not the answer.

- The second option is about elastic net. Therefore this is not the answer.

- I don't remember division being in any of the regularization formulas. Therefore this is not the answer.

- From lecture, the main idea of regularization is to minimize the sum of the square residuals plus a constraint on the vector of weights (aka coefficients) **This is my choice**

**Q5**

- Not polynomial regression since this only plots out the line of a polynomial through the data. Nothing should be estimated as 0.

- Not OLS since this is the sum of square residuals. Under some conditions C is 0 but only sometimes. Therefore this is not the answer.

- It is lasso due to the existance of the absolute value sign making it very likely for it to be set to 0. **This is my choice**

- It is not ridge since the coefficent is squared. Therefore this is not the answer.

**Q6**

data = datasets.load_boston()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
y = data.target
x = data.data

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
Xs_train = scale.fit_transform(x)

model = LinearRegression()
model.fit(Xs_train, y)
y_poly_pred = model.predict(Xs_train)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
print(rmse)

**Output: 4.679191295697281**

**Q7**

Q7data = datasets.load_boston()
Q7df = pd.DataFrame(data=Q7data.data, columns=Q7data.feature_names)
Q7y = Q7data.target

Q7kf = KFold(n_splits=10, random_state=1234,shuffle=True)

i = 0
PE = []
PE_train = []
model = Lasso(alpha=0.03)
for train_index, test_index in Q7kf.split(Q7df):
    X_train = Q7df.values[train_index]
    y_train = Q7y[train_index]
    X_test = Q7df.values[test_index]
    y_test = Q7y[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    PE_train.append(MSE(y_train,y_pred_train))
    PE.append(MSE(y_test, y_pred))

print('The k-fold crossvalidated error rate on the train sets is: ' + str(np.mean(PE_train)))
print('The k-fold crossvalidated error rate on the test sets is: ' + str(np.mean(PE)))

**The k-fold crossvalidated error rate on the test sets is: 24.742480813910557**

**Q8**

Q8data = datasets.load_boston()
Q8df = pd.DataFrame(data=Q8data.data, columns=Q8data.feature_names)
Q8y = Q8data.target

Q8kf = KFold(n_splits=10, random_state=1234,shuffle=True)

i = 0
PE = []
PE_train = []
model = model = ElasticNet(alpha=0.05,l1_ratio=0.9)
for train_index, test_index in Q8kf.split(Q8df):
    X_train = Q8df.values[train_index]
    y_train = Q8y[train_index]
    X_test = Q8df.values[test_index]
    y_test = Q8y[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    PE_train.append(MSE(y_train,y_pred_train))
    PE.append(MSE(y_test, y_pred))

print('The k-fold crossvalidated error rate on the train sets is: ' + str(np.mean(PE_train)))
print('The k-fold crossvalidated error rate on the test sets is: ' + str(np.mean(PE)))

**The k-fold crossvalidated error rate on the train sets is: 22.68329398863849**

**Q9**

Q9data = datasets.load_boston()
Q9df = pd.DataFrame(data=Q9data.data, columns=Q9data.feature_names)
Q9y = Q9data.target
Q9x = Q9data.data

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
Q9Xs_train = scale.fit_transform(Q9x)

Q9polynomial_features = PolynomialFeatures(degree=2)
Q9x_poly = Q9polynomial_features.fit_transform(Q9Xs_train)

model = LinearRegression()
model.fit(Q9x_poly, Q9y)
Q9y_poly_pred = model.predict(Q9x_poly)

Q9rmse = np.sqrt(mean_squared_error(Q9y,Q9y_poly_pred))
Q9r2 = r2_score(Q9y,Q9y_poly_pred)
print(Q9rmse)

**Output: 2.449087064744557**

**Q10**

data = datasets.load_boston()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
y = data.target
x = data.data

polynomial_features = PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x)

model = Ridge(alpha=0.1)
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

residuals = (y - y_poly_pred)

ax1 = sns.distplot(residuals,
                                        
                  bins=11,
                  kde=False,
                  color='deepskyblue',
                  hist_kws={"color":'lightpink'},
                  fit=stats.norm,
                  fit_kws={"color":'deepskyblue'})
ax1.set(xlabel='Residuals', ylabel='Frequency')

![Screenshot 2021-03-14 222042](https://user-images.githubusercontent.com/78627324/111095313-8a886700-8513-11eb-86ab-927dc37639ea.png)

stat, p = stats.shapiro(residuals)
print('The p-value is: '+str(p))

The p-value is: 5.911846966827339e-12

# Midterm Lab

**Q1**

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
data = pd.read_csv("../../Data310/Sample_Data/weatherHistory.csv")

from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import KFold # import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

**There are 12 observations**

**Q2**

Types of Data

- Data could be classified based on the levels of measurement, such as nominal, ordinal, interval and ratio.
- Nominal data: unique identifiers but order does not matter for example, peoples' names, names of flowers, colors etc.
- Ordinal data: unique indentifiers and order matters, such as in letter grades, military or administration ranks, etc.
- Interval level of measurement: numerical values where differences make sense in the context of using them but ratios or proportions could be meaningless, such as temperatures, heights of people, days in the calendar.
- Ratio level of measurement: numerical values, that are at the interval level of measurement and also the ratio and proportions make sense in the context of using them, such as salaries, weights, distances, forces, etc. 

**There are 3 nominal variables: Summary Precip Type and daily summary**

**Q3**

y3 = data['Temperature (C)'].values
x3 = data['Humidity'].values

model = LinearRegression()
model.fit(x3.reshape(-1,1), y3)

y_pred = model.predict(x3.reshape(-1,1))

np.sqrt(MSE(y3,y_pred))

**Q4**

data4 = data[['Humidity']]
y4 = data['Temperature (C)'].values
x4 = data['Humidity'].values
kf4 = KFold(n_splits=20, random_state=2020,shuffle=True)
scale = StandardScaler()
data4s = scale.fit_transform(data4)

i = 0
PE = []
PE_train = []
model4 = Ridge(alpha=0.1)
for train_index, test_index in kf4.split(data4):
    X_train = data4s[train_index]
    y_train = y4[train_index]
    X_test = data4s[test_index]
    y_test = y4[test_index].reshape(-1, 1)
    model4.fit(X_train.reshape(-1, 1), y_train)
    y_pred = model4.predict(X_test.reshape(-1, 1))
    y_pred_train = model4.predict(X_train.reshape(-1, 1))
    PE_train.append(np.sqrt(MSE(y_train,y_pred_train)))
    PE.append(np.sqrt(MSE(y_test, y_pred)))

print('The k-fold crossvalidated error rate on the train sets is: ' + str(np.mean(PE_train)))
print('The k-fold crossvalidated error rate on the test sets is: ' + str(np.mean(PE)))

**7.4000070622435405**

**Q5**

data5 = data[['Temperature (C)', 'Humidity']]
y5 = data['Temperature (C)']
x5 = data['Humidity']
kf5 = KFold(n_splits=10, random_state=1693,shuffle=True)
RF_reg = RandomForestRegressor(n_estimators=100,max_depth=50)

i = 0
PE = []
PE_train = []
model5 = RF_reg
for train_index, test_index in kf5.split(data5):
    X_train = x5.values[train_index]
    y_train = y5[train_index]
    X_test = x5.values[test_index]
    y_test = y5[test_index]
    model5.fit(X_train.reshape(-1, 1), y_train)
    y_pred = model5.predict(X_test.reshape(-1, 1))
    y_pred_train = model5.predict(X_train.reshape(-1, 1))
    PE_train.append(np.sqrt(MSE(y_train,y_pred_train)))
    PE.append(np.sqrt(MSE(y_test, y_pred)))

print('The k-fold crossvalidated error rate on the train sets is: ' + str(np.mean(PE_train)))
print('The k-fold crossvalidated error rate on the test sets is: ' + str(np.mean(PE)))

**Q6**
data6 = data[['Temperature (C)', 'Humidity']]
y6 = data['Temperature (C)'].values
x6 = data['Humidity'].values
kf6 = KFold(n_splits=10, random_state=1693,shuffle=True)
scale = StandardScaler()
Q6Xs_train = scale.fit_transform(x6.reshape(-1, 1))
Q6polynomial_features = PolynomialFeatures(degree=6)
data6_poly = Q6polynomial_features.fit_transform(data6)

i = 0
MSE_train = []
MSE_test = []
model6 = LinearRegression()
for train_index, test_index in kf6.split(data6):
    X_train = data6.values[train_index]
    y_train = y6[train_index]
    X_test = data6.values[test_index]
    y_test = y6[test_index]
    x_poly_train = Q6polynomial_features.fit_transform(np.array(X_train))
    x_poly_test = Q6polynomial_features.fit_transform(np.array(X_test))
    model6.fit(x_poly_train, y_train)
    yhat_train = model6.predict(x_poly_train)
    yhat_test = model6.predict(x_poly_test)
    MSE_train.append(np.sqrt(mean_squared_error(y_train,yhat_train)))
    MSE_test.append(np.sqrt(mean_squared_error(y_test,yhat_test)))
    
print('The k-fold crossvalidated error rate on the train sets is: ' + str(np.mean(MSE_train)))
print('The k-fold crossvalidated error rate on the test sets is: ' + str(np.mean(MSE_test)))

**~7.40**

**Q7**
data7 = data[['Humidity']]
y7 = data['Temperature (C)']
x7 = data['Humidity']
kf7 = KFold(n_splits=10, random_state=1234,shuffle=True)
scale = StandardScaler()
data7s = scale.fit_transform(data7)

i = 0
PE = []
PE_train = []
model7 = Ridge(alpha=0.1)
for train_index, test_index in kf7.split(data7s):
    X_train = data7.values[train_index]
    y_train = y7.values[train_index].reshape(-1, 1)
    X_test = data7.values[test_index]
    y_test = y7.values[test_index]
    model7.fit(X_train.reshape(-1, 1), y_train)
    y_pred = model7.predict(X_test.reshape(-1, 1))
    y_pred_train = model7.predict(X_train.reshape(-1, 1))
    PE_train.append(np.sqrt(MSE(y_train,y_pred_train)))
    PE.append(np.sqrt(MSE(y_test, y_pred)))

print('The k-fold crossvalidated error rate on the train sets is: ' + str(np.mean(PE_train)))
print('The k-fold crossvalidated error rate on the test sets is: ' + str(np.mean(PE)))

**7.4001**

**Q8**

data8 = data[['Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)', 'Wind Bearing (degrees)']]
y8 = data['Temperature (C)']
x8 = data[['Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)', 'Wind Bearing (degrees)']].values
kf8 = KFold(n_splits=10, random_state=1234,shuffle=True)
Q8polynomial_features = PolynomialFeatures(degree=6)
data8_poly = Q8polynomial_features.fit_transform(data8)

i = 0
PE = []
PE_train = []
model8 = Ridge(alpha=0.1)
for train_index, test_index in kf8.split(data8_poly):
    X_train = Q8polynomial_features.fit_transform(x8[train_index])
    y_train = y8.values[train_index]
    X_test = Q8polynomial_features.fit_transform(x8[test_index])
    y_test = y8.values[test_index]
    model8.fit(X_train, y_train)
    y_pred = model8.predict(X_test)
    y_pred_train = model8.predict(X_train)
    PE_train.append(np.sqrt(MSE(y_train,y_pred_train)))
    PE.append(np.sqrt(MSE(y_test, y_pred)))

print('The k-fold crossvalidated error rate on the train sets is: ' + str(np.mean(PE_train)))
print('The k-fold crossvalidated error rate on the test sets is: ' + str(np.mean(PE)))

**Q9**
data9 = data[['Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)', 'Wind Bearing (degrees)']]
y9 = data['Temperature (C)']
x9 = data[['Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)', 'Wind Bearing (degrees)']]
kf9 = KFold(n_splits=10, random_state=1234,shuffle=True)

RF_reg = RandomForestRegressor(n_estimators=100,max_depth=50)
i = 0
PE = []
PE_train = []
model9 = RF_reg
for train_index, test_index in kf9.split(data9):
    X_train = x9.values[train_index]
    y_train = y9[train_index]
    X_test = x9.values[test_index]
    y_test = y9[test_index]
    model9.fit(X_train, y_train)
    y_pred = model9.predict(X_test)
    y_pred_train = model9.predict(X_train)
    PE_train.append(np.sqrt(MSE(y_train,y_pred_train)))
    PE.append(np.sqrt(MSE(y_test, y_pred)))

print('The k-fold crossvalidated error rate on the train sets is: ' + str(np.mean(PE_train)))
print('The k-fold crossvalidated error rate on the test sets is: ' + str(np.mean(PE)))

**Q10**
y10 = data['Temperature (C)']
x10 = data['Humidity']
fig, ax = plt.subplots(figsize=(10,8))
ax.scatter(y10,x10)
plt.show()

![Screenshot 2021-03-23 173559](https://user-images.githubusercontent.com/78627324/112221798-41fd3780-8bfe-11eb-8084-2f54ce02b782.png)

# Lab 05

**Q1**

SVM classification, one or more landmark points.

Evidence? 

<figcaption>SVM with Radial Basis Function Kernel</figcaption></center>
</figure>

For this we would need at least one landmark point $x_0$. The following is also called a "Gaussian" kernel

$$\Large
(x,y) \rightarrow \left(x,y,z:=e^{-\gamma[(x-x_0)^2+(y-y_0)^2]}\right)
$$

**Q2**

Therefore it is false, hardmargins can be overly sensitive to noise. This makes logical sense.

**Q3**

This is true, for k-nearest neighbors the number of neighbors makes a circle. For estimating a point, it is always estimated within the cirlce. The classification depends on the majority. Distance can become involved too.

**Q4**

dat = load_breast_cancer()
df = pd.DataFrame(data=dat.data, columns=dat.feature_names)
X = df[['mean radius', 'mean texture']]

y = dat.target
dat.target_names

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1693)

len(x_train)
len(y_train)

**426**

**Q5**

False since the next questions involve independent varaibles that are two. I think the minimum required is two for the hyperplane.

**Q6**

scale = StandardScaler()

xscaled = scale.fit_transform(X)
xscaledtrain = scale.fit_transform(x_train)
datax = [[15.78,17.89]]
datay = y.transpose()

svm = SVC(kernel='rbf')
svm.fit(x_train, y_train)
svm.predict([[16.78,17.89]])
#0 is malignant
**array([0])**

**Q7**

model = LogisticRegression(solver='lbfgs')
model.fit(xscaledtrain, y_train)
prediction = model.predict(X)
accuracy = accuracy_score(y, prediction)
print(accuracy)

**0.37258347978910367**

**Q8**

n_neighbors = 5
#clf = neighbors.KNeighborsClassifier(n_neighbors)
#clf.predict([[16.78,17.89]])
knn = KNN(n_neighbors=5, weights='uniform')
knn.fit(xscaledtrain, y_train)
knn.predict([[17.18,8.65]])
# 0 is malignant

**array([0])**

**Q9**

model = RandomForestClassifier(random_state=1234, max_depth=5, n_estimators = 100)
model.fit(X, y);
predicted_classes = model.predict(X)
accuracy = accuracy_score(y,predicted_classes)
print(accuracy)

from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold

def validation(X,y,k,model):
  PA_IV = []
  PA_EV = []
  pipe = Pipeline([('scale',scale),('Classifier',model)])
  kf = KFold(n_splits=k,shuffle=True,random_state=1234)
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test = X[idxtest,:]
    y_test = y[idxtest]
    pipe.fit(X_train,y_train)
    PA_IV.append(accuracy_score(y_train,pipe.predict(X_train)))
    PA_EV.append(accuracy_score(y_test,pipe.predict(X_test)))
  return np.mean(PA_IV), np.mean(PA_EV)
  
  cv = StratifiedKFold(n_splits=10)
classifier = RandomForestClassifier(random_state=1234, max_depth=5, n_estimators = 100)


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for i, (train, test) in enumerate(cv.split(X, y)):
    #classifier.fit(X[train], y[train])
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

**Q10**

Problem #1: Predicted value is continuous, not probabilistic

In a binary classification problem, what we are interested in is the probability of an outcome occurring. Probability is ranged between 0 and 1, where the probability of something certain to happen is 1, and 0 is something unlikely to happen. But in linear regression, we are predicting an absolute number, which can range outside 0 and 1.

**Q11**

Sometimes the data is not about the distances, instead it is about the majority. So in this case KNN would not have better results.

# Lab 06

**Q1**

The maximum depth of the tree if none, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

**True**

**Q2**

This is correct due to the creation of more complex decision boundries or ones that can be nonlinear.

**True**

**Q3**

n_estimators are the number of trees in the forest. 

**Q4**

dat = load_breast_cancer()
df = pd.DataFrame(data=dat.data, columns=dat.feature_names)
X = df[['mean radius', 'mean texture']].values
y = dat.target

y = dat.target
dat.target_names
Xtrain,Xtest,ytrain,ytest= tts(X,y,test_size=0.25,random_state=1693)
scaler=StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)
spc = dat.target_names

**BAYESIAN TREE DOES NOT EXIST**

dt_class = RandomForestClassifier(random_state=1693, max_depth=5, n_estimators = 1000)
dt_class.fit(Xtrain,ytrain)
y_pred = dt_class.predict(Xtest)
dt_cm = confusion_matrix(ytest, y_pred)
pd.DataFrame(dt_cm, columns=spc, index=spc)

dt_class = GaussianNB()
dt_class.fit(Xtrain,ytrain)
y_pred = dt_class.predict(Xtest)
dt_cm = confusion_matrix(ytest, y_pred)
pd.DataFrame(dt_cm, columns=spc, index=spc)

dt_class = DecisionTreeClassifier(random_state=1693)
dt_class.fit(Xtrain,ytrain)
y_pred = dt_class.predict(Xtest)
dt_cm = confusion_matrix(ytest, y_pred)
pd.DataFrame(dt_cm, columns=spc, index=spc)

**CLASSIFICATION TREE HAD HIGHEST WITH 7**

**Q5**

This is choosen since testing positive means you have cancer. So that means it is malignant when in reality it is benign. Therefore the option I choose should be correct

**Q6** 

Naive Bayes classifiers are probabilistic classifiers due to the use of Bayes Theorem.

**True**

**Q7**

This is the unsplitted top of the tree. Therefore it contains all data points.

A root node contains all data

**Q8**

This form of classification is about the probability of a membership. Additionally, it can be assigned to more than one class.

Soft classification is about probablility. 

**Q9**

According to the definitions from lecture 3, it is:

The probability you are solving for, for each class.

**Q10**

That is what confusion matrixes do!

The number of true pos, false pos, true neg and false neg

**Q11**

Axon is about outputs, specifically in biology it involves action potentials that eventually reach another neuron. This is all about the output!

The outputs from each neuron along the synapse.

**Q12**

Neuron the main agent of action for the network

Hidden layer a set of neurons that shares weight information

Cost function this is the error compared to real data.

Back propagation is infromation that reduces error and increases accuracy.

Gradient descent is the method that minimizes the cost function.

**Q13**

*Didn't type anything here or it got deleted by accident*

**Q14**

dat = load_breast_cancer()
df = pd.DataFrame(data=dat.data, columns=dat.feature_names)
X = df[['mean radius', 'mean texture']].values
y = dat.target

kf = KFold(n_splits=10,shuffle=True,random_state=1693)

model = GaussianNB()

AC = []
for idxtrain, idxtest in kf.split(X):
  Xtrain = X[idxtrain,:]
  Xtest = X[idxtest,:]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  model.fit(Xtrain,ytrain)
  AC.append(model.score(Xtest,ytest))
  
np.mean(AC)
**0.8805137844611528**

**Q15**
dat = load_breast_cancer()
df = pd.DataFrame(data=dat.data, columns=dat.feature_names)
X = df[['mean radius', 'mean texture']].values
y = dat.target

kf = KFold(n_splits=10,shuffle=True,random_state=1693)

model = RandomForestClassifier(random_state=1693, max_depth=7, n_estimators = 100)

AC = []
for idxtrain, idxtest in kf.split(X):
  Xtrain = X[idxtrain,:]
  Xtest = X[idxtest,:]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  model.fit(Xtrain,ytrain)
  AC.append(model.score(Xtest,ytest))
  
np.mean(AC)
**0.8804824561403508**

**Q16**

dat = load_breast_cancer()
df = pd.DataFrame(data=dat.data, columns=dat.feature_names)
X = df[['mean radius', 'mean texture']].values
y = dat.target

scale = StandardScaler()

model = Sequential()
model.add(Dense(16,kernel_initializer='random_normal', input_dim=2, activation='relu'))
model.add(Dense(8,kernel_initializer='random_normal', activation='relu'))
model.add(Dense(4,kernel_initializer='random_normal', activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

kf = KFold(n_splits=10,shuffle=True,random_state=1693)

AC = []
for idxtrain, idxtest in kf.split(X):
  Xtrain = X[idxtrain,:]
  Xtest  = X[idxtest,:]
  ytrain = y[idxtrain]
  ytest  = y[idxtest]
  Xstrain = scale.fit_transform(Xtrain)
  Xstest  = scale.transform(Xtest)
  model.fit(Xstrain, ytrain, epochs=150, verbose=0,validation_split=0.25,batch_size=10,shuffle=False)
  AC.append(acc(ytest,model.predict_classes(Xstest)))
  print(acc(ytest,model.predict_classes(Xstest)))

np.mean(AC)
**0.8893170426065163**
