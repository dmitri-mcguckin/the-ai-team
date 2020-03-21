import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt

#Preprocessing. 60,000 X 784pixels training images  paired with 60,000 image cfs.
images,labels = loadlocal_mnist(
        images_path='data/train-images-idx3-ubyte',
        labels_path='data/train-labels-idx1-ubyte') #load ubyte training data and corresponding labels.

test_images,test_labels = loadlocal_mnist(
        images_path='data/t10k-labels-idx1-ubyte',
        labels_path='data/t10k-images-idx3-ubyte') #load ubyte training data and corresponding labels.


images = images / 255 #normalize pixel values in matrix such that for all pixels, pixel value <= 1.
learning_param = 0.0001 #This is the learning param of the network.

#These two variables configure the left strip of the image.
left_start = int(6 * 28)
left_end =  int(7 * 28)
#These two variables configure the right strip of the image.
right_start = 16 * 28
right_end =  21 * 28
#These two variables configure the center strip of the image.
start = 12 * 28
end   = 15 * 28

perceptron_network =  np.random.uniform(-0.05,0.05,size=(end-start + left_end - left_start + right_end - right_start + 1,10)) #785 weights X 10 perceptrons
confusion_matrix =  np.zeros([10,10]) #confusion matrix for training.
test_confusion_matrix = np.zeros([10,10]) #confusion matrix for testing.
epoch_accuracy = [] # this python array will be used to store (epoch,accuracy tuples) for training data
test_epoch_accuracy = [] # this python array will be used to store (epoch,accuracy tuples) for test data





#This function trains our network
def train():
    accuracy = 0.0 #This value tracks the accuracy of the perceptron network for a certain epoch.
    accuracyIncrementor = 1/len(images) #Whenever a data element is correctly classified, accuracy is incremented 1/60000 in a given epoch.

    test_accuracy = 0.0
    test_accuracyIncrementor = 1/len(test_images)

    i = 0
    epoch = 0

    while epoch  <20:
        epoch_accuracy.append(accuracy)
        test_epoch_accuracy.append(test_accuracy)
        accuracy = 0.0
        test_accuracy = 0.0

        i= 0 #i iterates over test labels
        k = 0 #k iterates over training labels
        print(epoch_accuracy)

        for training_data in images:
            training_data_left = training_data.transpose()[left_start:left_end]
            training_data_right =  training_data.transpose()[right_start:right_end]
            training_data =  training_data.transpose()[start:end]
            training_data = np.append(training_data_left,training_data)
            training_data = np.append(training_data_right,training_data)
            biased_data = np.append([1], training_data) # Add bias to training data
            perceptrons_output =  np.dot( biased_data,perceptron_network) # debug
            percp_class = np.argmax(perceptrons_output) #This number contains the highest firing perceptron index which corresponds to the classified #.
            confusion_matrix[labels[i]][percp_class] += 1

            if  percp_class != labels[i]: #if incorrect classification
                j = 0
                for neuron in perceptron_network.T:
                    y =  1 if perceptrons_output[j] > 0 else 0 # set y of perceptron to 1 if w.x  > 0
                    t = int(j == labels[i])                     # If neuron j corresponds to desired class, set t to 1 else set t to 0
                    neuron += biased_data * learning_param * (t - y)  # add delta w to neurons previous weight vector
                    j = j + 1                                   # go onto next neuron
            else: #classification correct increment acc!
                accuracy += accuracyIncrementor


            i = i + 1 #increment i.


        for test_data in test_images:
            left_test_data = test_data.transpose()[left_start:left_end]
            right_test_data = test_data.transpose()[right_start:right_end]
            test_data = test_data.transpose()[start:end]
            test_data = np.append(left_test_data,test_data)
            test_data = np.append(right_test_data, test_data) # Add bias to test data
            biased_data = np.append([1],test_data)
            percp_class = np.argmax(np.dot( biased_data,perceptron_network)) #This number contains the highest firing perceptron index which corresponds to the classified #.
            test_confusion_matrix[test_labels[k]][percp_class] += 1

            if(percp_class == test_labels[k]):
                test_accuracy += test_accuracyIncrementor


            k+=1


        epoch += 1 #all data has been processed which means one epoch has elapsed.








train()
plt.title("Dim reduction test with 60000/60000 images used to train" )
plt.plot(test_epoch_accuracy)
plt.plot(epoch_accuracy)
plt.show()


print(confusion_matrix)
print(test_confusion_matrix)
