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

import random
import matplotlib.pyplot as plt

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

import pandas as pd
import operator
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
#from yellowbrick.regressor import ResidualsPlot
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

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

# Lab 03

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

import pandas as pd
import operator
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
#from yellowbrick.regressor import ResidualsPlot
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn import datasets
from sklearn.model_selection import KFold # import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
import seaborn as sns
from scipy import stats
from scipy.stats import norm

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

