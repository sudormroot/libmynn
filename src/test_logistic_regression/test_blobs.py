
""" Loading the blobs dataset

"""
def load_blobs_dataset():

    # Use pandas to read the CSV file as a dataframe
    df = pd.read_csv("dataset" + os.path.sep + "blobs250.csv")

    return df
    
    




# We load the blobs dataset into df
df = load_blobs_dataset()

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
                                    max_iters = 200,
                                    n_input = n_input
                                )

# Training our logistic regression
mylr.fit(X_train, y_train)

# Predicting on testing dataset
y_pred = mylr.predict(X_test)

# Computing the accuracy on testing dataset
accuracy = mylr.accuracy(y_pred, y_test)


print("\n")
print("Accuracy on testing dataset: ", accuracy)
print("\n")


#
# We draw our loss in the following code.
#

hist_loss = mylr.hist_loss()
fig = plt.figure()
plt.plot(hist_loss, color = 'red', label = 'Loss value')

plt.xlabel("Iterations", fontsize = 12) 
plt.ylabel("Loss Value", fontsize = 12) 

plt.xlim(0, len(hist_loss) - 1)
plt.ylim(0, max(hist_loss))

plt.legend( loc = 'upper center',
            fontsize = 12, 
            ncol = 1,
            frameon = False) 

