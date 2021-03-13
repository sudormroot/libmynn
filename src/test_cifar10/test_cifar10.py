# This function taken from the CIFAR website
#import pickle

def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        
    return dict

# Loaded in this way, each of the batch files contains a dictionary with the following elements:
#   data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. 
#           The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. 
#           The image is stored in row-major order, so that the first 32 entries of the array are the red channel values 
#           of the first row of the image.
#   labels -- a list of 10000 numbers in the range 0-9. 
#             The number at index i indicates the label of the ith image in the array data.


def loadbatch(batchname):
    folder = 'dataset/cifar-10-batches-py'
    batch = unpickle(folder + "/" + batchname)
    return batch

def loadlabelnames():
    folder = 'dataset/cifar-10-batches-py'
    meta = unpickle(folder+"/"+'batches.meta')
    return meta[b'label_names']

def visualise(data, index):
    # MM Jan 2019: Given a CIFAR data nparray and the index of an image, display the image.
    # Note that the images will be quite fuzzy looking, because they are low res (32x32).

    picture = data[index]
    # Initially, the data is a 1D array of 3072 pixels; reshape it to a 3D array of 3x32x32 pixels
    # Note: after reshaping like this, you could select one colour channel or average them.
    picture.shape = (3,32,32) 
    
    # Plot.imshow requires the RGB to be the third dimension, not the first, so need to rearrange
    picture = picture.transpose([1, 2, 0])
    plt.imshow(picture)
    plt.show()
    

batch1 = loadbatch('data_batch_1')
#print("Number of items in the batch is", len(batch1))

# Display all keys, so we can see the ones we want
#print('All keys in the batch:', batch1.keys())

data = batch1[b'data']
labels = batch1[b'labels']
print ("size of data in this batch:", len(data), ", size of labels:", len(labels))
print (type(data))
print(data.shape)

names = loadlabelnames()


# Display a few images from the batch
#for i in range (100,105):
#   visualise(data, i)
#   print("Image", i,": Class is ", names[labels[i]])

# print(len(data))


#print(names[labels[0]])



""" RGB to grayscale

     Gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

"""

def rgb2gray(img):

    # Initially, the data is a 1D array of 3072 pixels; reshape it to a 3D array of 3x32x32 pixels
    # Note: after reshaping like this, you could select one colour channel or average them.

    img.shape = (3,32,32)

    # Plot.imshow requires the RGB to be the third dimension, not the first, so need to rearrange
    img = img.transpose([1, 2, 0])

    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

    gray = gray.reshape(1, -1).flatten()

    return gray




""" We normalise each image by using max-min scheme.

"""

def normalise_image(gray):

    xmax = np.max(gray)
    xmin = np.min(gray)


    return (gray - xmin) / (xmax - xmin)



N = len(data)

# print("Dataset size: ", N)

# We randomly sample samples for training and testing.
N_TRAIN = 3000
N_TEST = 100


""" We choose dog and cat as the two classes.

"""

CLASS_A = "deer"
CLASS_B = "dog"


""" We collect the indices of all the images having the label cat or dog.
    We then will sample on the indices rather on data direclty.

"""

indices = []

for i, image in enumerate(data):
    label = names[labels[i]].decode("utf-8")

    if label == CLASS_A:
        indices.append(i)
    elif label == CLASS_B:
        indices.append(i)

""" We sample samples from the indices.

"""

# reshuffle the data points
np.random.shuffle(indices)

indices_train = np.random.choice(indices, N_TRAIN)
indices_test = np.random.choice(indices, N_TEST)

# print(indices_test)



""" We build training dataset here

"""

X_train = []
y_train = []

for i in indices_train:
    image = data[i]
    label = names[labels[i]]
    #print(label.decode("utf-8") )
    gray = rgb2gray(image)
    grap = normalise_image(gray)
    X_train.append(gray)
    y_train.append(label)

X_train = np.array(X_train)




""" We build testing dataset here

"""

X_test = []
y_test = []

for i in indices_test:
    image = data[i]
    label = names[labels[i]]
    gray = rgb2gray(image)
    grap = normalise_image(gray)
    X_test.append(gray)
    y_test.append(label)


X_test = np.array(X_test)



""" We create our classifier.

"""

n_input = len(X_train[0])

n_output = len(set(y_train))

print("n_input=", n_input)
print("n_output=", n_output)

myclf = MyMLPClassifier( n_input = n_input,
                         n_output = n_output,
                         hidden_sizes = (32,), #define hidden layers
                         n_validation = 50,
                         learning_rate = 0.001,
                         n_epochs = 300,
                         batch_size = 1,
                         #alpha = 0.0001,
                         activation = 'relu',
                         print_per_epoch = 10,
                         debug = True)


myclf.fit(X_train, y_train)
y_pred = myclf.predict(X_test)

accuracy = myclf.accuracy(y_pred, y_test)

print("\n")
print("Accuracy on testing dataset: ", accuracy)
print("\n")


#
# We draw our loss in the following code.
#

hist_loss = myclf.loss_history()

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


