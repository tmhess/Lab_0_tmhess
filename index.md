Lab01

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
# The following replaces the NaNs in the "Funds" column with the mean value
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

# approximated results
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
