#import matplotlib
import matplotlib.pyplot as plt
#import sklearn
import sklearn.datasets
import pandas as pd
import numpy as np
import os

# Display plots inline and change default figure size
%matplotlib inline


""" Loading the moons dataset

"""
def load_moons_dataset():

    # Use pandas to read the CSV file as a dataframe
    df = pd.read_csv("dataset" + os.path.sep + "moons400.csv")

    return df
    
    
""" Splitting a dataset into training and testing.

"""

def split_dataset(df, K = 0.7):
    
    # Reshuffle the dataset first randomly.
    df = df.sample(frac = 1)


    # Split the dataset
    L = len(df)

    n = int(K * L)

    df_train = df[0: n]
    df_test = df[n: L]

    return df_train, df_test




# We load the moons dataset into df
df = load_moons_dataset()


#print("len(df)=", len(df))

# We split the dataset by 70:30
df_train, df_test = split_dataset(df, 0.7)  

#
# We ontain the X and y for training and testing respectively.
#
y_train = df_train['Class'].values
del df_train['Class']
X_train = df_train.values

y_test = df_test['Class'].values
del df_test['Class']
X_test = df_test.values


n_input = len(X_train[0])


# We create a logistic regression here
mylr = MyFancyLogisticRegression(   learning_rate = 0.01,
                                    max_iters = 300,
                                    n_input = n_input
                                )

# Training our logistic regression
mylr.fit(X_train, y_train)

# Predicting on testing dataset
y_pred = mylr.predict(X_test)

# Computing the accuracy on testing dataset
accuracy = mylr.accuracy(y_pred, y_test)

#we manually verified the correctness of our algorithm!
#print(list(y_pred))
#print(list(y_test))

print("\n")
print("Accuracy on testing dataset: ", accuracy)
print("\n")


# Save our model to a file.
mylr.save("my_lr.model")

# We re-load the saved model and test it on testing dataset!
mylr2 = MyFancyLogisticRegression(modelfile = "my_lr.model")
y_pred = mylr2.predict(X_test)
accuracy = mylr2.accuracy(y_pred, y_test)

print("\n")
print("Loaded model, the measured accuracy on testing dataset is ", accuracy)
print("\n")

#
# We draw our loss in the following code.
#

hist_loss = mylr.hist_loss()
fig = plt.figure()

plt.title("The loss value with respect to iterations")
plt.plot(hist_loss, color = 'coral', label = 'Loss value')

plt.xlabel("Iterations", fontsize = 12) 
plt.ylabel("Loss Value", fontsize = 12) 

plt.xlim(0, len(hist_loss) - 1)
plt.ylim(0, max(hist_loss))

plt.legend( loc = 'upper center',
            fontsize = 12, 
            ncol = 1,
            frameon = False)  
