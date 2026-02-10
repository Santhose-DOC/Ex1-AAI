<H3> Name : SANTHOSE AROCKIARAJ J</H3>
<H3>Register No. : 212224230248</H3>
<H3> Experiment 1</H3>
<H3>DATE : 10/02/2026 </H3>
<H1 ALIGN=CENTER> Implementation of Bayesian Networks</H1>

## Aim :

To create a bayesian Network for the given dataset in Python

## Algorithm:

### Step 1
Import the required libraries such as `pandas`, `networkx`, `matplotlib.pyplot`, and Bayesian Network modules (`DiscreteBayesianNetwork`, `MaximumLikelihoodEstimator`, `VariableElimination`) from `pgmpy`.

### Step 2
Set pandas display options (optional) to allow better visibility of dataset columns during analysis.

### Step 3
Load the weather dataset from a CSV file using the pandas `read_csv()` function.

### Step 4
Remove all records where the target variable **RainTomorrow** contains missing values to ensure valid training data.

### Step 5
Identify numeric columns in the dataset and replace missing values with their respective column mean values.

### Step 6
Discretize continuous variables into categorical bands:
- Convert `WindGustSpeed` into **Low**, **Medium**, and **High**
- Convert `Humidity9am` into **Low** and **High**
- Convert `Humidity3pm` into **Low** and **High**

### Step 7
Create a new dataset containing only the selected categorical variables:
`WindGustSpeedCat`, `Humidity9amCat`, `Humidity3pmCat`, and `RainTomorrow`.
Convert all these variables into string format.

### Step 8
Define the Bayesian Network structure using `DiscreteBayesianNetwork` by specifying the dependency relationships:
- `Humidity9amCat → Humidity3pmCat`
- `Humidity3pmCat → RainTomorrow`
- `WindGustSpeedCat → RainTomorrow`

### Step 9
Train the Bayesian Network using **Maximum Likelihood Estimation (MLE)** to learn the Conditional Probability Tables (CPTs).

### Step 10
Construct a directed graph using NetworkX based on the Bayesian Network structure.

### Step 11
Visualize the Bayesian Network graph by setting node positions, colors, and labels using Matplotlib.

### Step 12
Initialize the **Variable Elimination** inference method for probabilistic reasoning on the trained model.

### Step 13
Perform inference to compute the probability of **RainTomorrow** given evidence on:
- `Humidity3pmCat`
- `WindGustSpeedCat`

### Step 14
Display the inference results and the Bayesian Network diagram.

---


## Program:
```python
#----------------------------------
# Imports
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

#----------------------------------
# Load dataset
df = pd.read_csv("/content/weatherAUSnew.csv")

# Drop missing target
df = df[df['RainTomorrow'].notna()]

# Fill numeric missing values with mean
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

#----------------------------------
# Discretize variables
df['WindGustSpeedCat'] = df['WindGustSpeed'].apply(
    lambda x: 'Low' if x <= 40 else 'Medium' if x <= 50 else 'High'
)

df['Humidity9amCat'] = df['Humidity9am'].apply(
    lambda x: 'Low' if x <= 60 else 'High'
)

df['Humidity3pmCat'] = df['Humidity3pm'].apply(
    lambda x: 'Low' if x <= 60 else 'High'
)

# Final DataFrame
bn_df = df[['WindGustSpeedCat','Humidity9amCat','Humidity3pmCat','RainTomorrow']]
for col in bn_df.columns:
    bn_df[col] = bn_df[col].astype(str)

#----------------------------------
# Define Bayesian Network structure
model = DiscreteBayesianNetwork([
    ('Humidity9amCat','Humidity3pmCat'),
    ('Humidity3pmCat','RainTomorrow'),
    ('WindGustSpeedCat','RainTomorrow')
])

#----------------------------------
# Train model
model.fit(bn_df, estimator=MaximumLikelihoodEstimator)

#Conditional probability
print("Conditional Probability Tables")
for cpd in model.get_cpds():
    print(cpd)
    print("-" * 60)

#----------------------------------
# Plot simple BN diagram
G = nx.DiGraph(model.edges())

plt.figure(figsize=(6,4))
nx.draw(G, with_labels=True, node_size=3000, node_color='lightblue')
plt.title("Bayesian Network")
plt.show()

#----------------------------------
# Inference example
inference = VariableElimination(model)
print("Inference example\n ***If 'Humidity3pmCat'='High' & 'WindGustSpeedCat'='High'***")
result = inference.query(
    variables=['RainTomorrow'],
    evidence={'Humidity3pmCat':'High','WindGustSpeedCat':'High'}
)

print(result)


```
## Output:

![alt text](image.png)

![alt text](image-1.png)

## Result:
   Thus a Bayesian Network is generated using Python

