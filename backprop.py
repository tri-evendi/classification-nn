from random import randrange
from random import random
from csv import reader
from math import exp
import numpy as np
from sklearn.metrics import confusion_matrix,precision_score,recall_score,accuracy_score
import pickle
from tqdm import tqdm


# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	print("Lookup : ", lookup)
	return lookup

# mencari nilai terkecil dan terbesar dari data set
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

# Normalisasi Data 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def normalize_data(data, minmax):
		for i in range(len(data)-1):
			data[i] = (data[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Pengujian K-Fold
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0



# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# fungsi aktifasi output sigmoid
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# propagasi maju input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# propragasi mundur untuk menghitung error 
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network bobot dengan error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network 
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in tqdm(range(n_epoch), desc='Training Progress'):
        sum_error = 0
        y_pred = list()
        y_actual = list()
        for row in train:
            # outputs = forward_propagate(network, row)
            outputs = forward_propagate(network, row[:-1]) # Exclude the label from inputs
            y_pred.append(outputs.index(max(outputs)))
            expected = [0 for i in range(n_outputs)]
            # expected[row[-1]] = 1
            expected[int(row[-1])] = 1  # Ensure the class label is an integer
            y_actual.append(expected.index(max(expected)))
            backward_propagate_error(network, expected)
            # update_weights(network, row, l_rate)
            update_weights(network, row[:-1], l_rate)  # Exclude the label from inputs
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            # print ('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

	# Evaluate the model using confusion matrix, recall, precision, and accuracy
    cm = confusion_matrix(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred, average=None)
    prec = precision_score(y_actual, y_pred, average=None)
    acc = accuracy_score(y_actual,y_pred)
    print("Confusion Matrix: \n",cm)
    print("Recall Confusion Matrix: ",recall)
    print("Precission Confusion Matrix: ",prec)
    print("Accuracy: ",round(acc*100,1))
    
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# Backpropagation Algorithm for k-fold
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)

# Load a network from a file
def save_network(network, filename):
    try:
        network_copy = []
        for layer in network:
            # Assuming each neuron in the layer is a dictionary with 'weights' and 'output'
            layer_copy = [{'weights': neuron['weights'], 'output': neuron['output']} for neuron in layer]
            network_copy.append(layer_copy)
        
        # Convert to a numpy array to ensure uniformity
        network_array = np.array(network_copy, dtype=object)
        print("Network array:", network_array)
        
        # Save the array
        np.save(filename, network_array, allow_pickle=True, fix_imports=True)
        print("Network saved successfully to", filename)
    except Exception as e:
        print("Failed to save the network:", e)


# Load data
filename = 'dataset.csv'
dataset = load_csv(filename)

# Skip the header
processed_dataset = dataset[1:]

# Convert string columns to float for all rows in the dataset (excluding the header)
for i in range(len(processed_dataset[0])-1):
    str_column_to_float(processed_dataset, i)

# Convert class column (last column) to integers
str_column_to_int(processed_dataset, len(processed_dataset[0])-1)

# Normalize input variables
minmax = dataset_minmax(processed_dataset)
normalize_dataset(processed_dataset, minmax)

# Define inputs and outputs for the network
n_inputs = len(processed_dataset[0]) - 1  # Number of features
n_outputs = len(set(row[-1] for row in processed_dataset))  # Number of unique classes

print ("input ", n_inputs)
print ("output ", n_outputs)
print ("data 0:", dataset[0])
print ("data processed 0:", processed_dataset[0])

# Initialize the network
hiddenlayer = 12
network = initialize_network(n_inputs, hiddenlayer, n_outputs)
# Train the model using backpropagation
train_network(network, processed_dataset, 0.05, 3800, n_outputs)
print ("After training ", network)
# saving the model using npy format
save_network(network, 'network-train.npy')
# np.save("network-train.npy", network, allow_pickle=True, fix_imports=True)
