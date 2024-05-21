from random import random
from math import exp
import numpy as np
from csv import reader
import pickle

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

# Find the minimum and maximum values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

# Normalize the data input
def normalize_data(data, minmax):
    # example of minmax : [[35.6771, 293.8562], [0.1825, 0.911], [0.0446, 0.8539], [0.8319, 0.9859], [-0.5271, -0.002], [0, 1]]
	for i in range(len(data)):
		data[i] = (data[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            
# Calculate neuron activation for an input
def activate(weights, inputs):
	"""Calculate neuron activation for an input using the dot product plus bias."""
	activation = weights[-1] # Bias term
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	"""Sigmoid activation function."""
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    print ("Result of forward propagation : ", inputs)
    return inputs

def predict(network, row):
    outputs = forward_propagate(network, row)
    print("Result of predict:", outputs)
    
    # Check for NaN values in the output
    if np.any(np.isnan(outputs)):
        return -1

    # Find the index with the maximum probability
    predicted_class_index = np.argmax(outputs)
    print("Predicted class index:", predicted_class_index)

    # Define a threshold to identify uncertain predictions as 'Unknown'
    threshold = 0.5  # You can adjust this threshold based on your requirements
    unknowed = 0.0001
    
    # Check if any of the output probabilities are less than 0.0001
    if any(prob < unknowed for prob in outputs):
        return -1

	# Check if the output probabilities are less than 0.5
    if all(prob < threshold for prob in outputs):
        return -1
    else:
        # Map the index to the class label
        if predicted_class_index == 0:
            return 0
        elif predicted_class_index == 1:
            return 1
        else:
            return -1  # Assuming the 3rd class is for 'Unknown'

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
print ("Minmax : ", minmax)

# Load model
netw = np.load("network-train.npy", allow_pickle=True, fix_imports=True, encoding='ASCII')

def klasifikasiBP(data):
    print ("Data awal citra :", data, " dengan panjang ", len(data))
    # Ensure `data` only contains feature values (no class label)
    if len(data) > len(minmax):
        data = data[:len(minmax)]  # Adjust if `data` includes a class label
    print("Dataset minmax:", minmax)
    # Normalize the features
    normalize_data(data, minmax)
    print("Data setelah dinormalisasi:", data)
    # Make a prediction using the neural network
    prediction = predict(netw.tolist(), data)  # Ensure `netw` is in the correct format
    print("Hasil klasifikasi:", prediction)
    return prediction