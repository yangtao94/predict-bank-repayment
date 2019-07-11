# predict-bank-repayment
Decision Tree model to predict if bank loans are being repaid.

Data is first being cleaned and saved to another .csv files

Then the model is trained using data from the clean .csv file. 

Adaboost is used during learning.


# Predict Bank Loan Repayment

## Table of Contents
+ [About](#about)
+ [Getting Started](#getting_started)
+ [Usage](#usage)
+ [Contributing](../CONTRIBUTING.md)

## About <a name = "about"></a>
A machine learning model to predict if bank loans are being repaid (Binary Classification). Utilized Adaboost with decision tree as the weak classifier. Top 20 placing in <a href = " https://imgur.com/CINwpWB">Kaggle Competition</a>

## Getting Started <a name = "getting_started"></a>
Clone the repository and navigate to the working folder!

Loan_Predict.csv -- the data to predict

Loan_Training.csv -- training data

run both training.py and testing.py to get a cleaned version of the data.

### Prerequisites

Anaconda; Python, Jupyter Notebook

Install <a href= "https://www.anaconda.com/distribution/">Anaconda</a> and the rest are all installed together

```
jupyter notebook
```
in the Anaconda terminal to open up the jupyter notebook
### Installing


## Usage <a name = "usage"></a>
After running training.py and testing.py, two new files will be created. Run kaggle.py to output a file containing the predicted results

