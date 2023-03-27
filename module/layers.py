import numpy as np

class Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)


        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * \
                             self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                            self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    # Retrieve layer parameters
    def get_parameters(self):
        return self.weights, self.biases

    # Set weights and biases in a layer instance
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


# LSTM
class LSTM(Dense):
    def __init__(self, input_size, hidden_size):
        # Call the super constructor to initialize the weights and biases
        super().__init__(input_size + hidden_size, 4 * hidden_size)
        self.hidden_size = hidden_size

    def forward(self, inputs, state):
        # Concatenate the input and previous hidden state
        concatenated = np.concatenate([inputs, state], axis=1)
        # Call the forward method of the super class with the concatenated input
        super().forward(concatenated, training=True)
        # Split the output into four parts for the input gate, forget gate,
        # output gate, and cell state update
        self.ig, self.fg, self.og, self.cu = np.split(self.output, 4, axis=1)
        # Apply sigmoid and tanh activations to the parts
        self.ig = 1 / (1 + np.exp(-self.ig))
        self.fg = 1 / (1 + np.exp(-self.fg))
        self.og = 1 / (1 + np.exp(-self.og))
        self.cu = np.tanh(self.cu)
        # Compute the new cell state and output
        self.c = self.fg * state + self.ig * self.cu
        self.h = self.og * np.tanh(self.c)
        # Return the output and new cell state as a tuple
        return self.h, self.c

    def backward(self, dh, dc):
        # Compute the gradients of the output and cell state with respect to the loss
        dho = dh * np.tanh(self.c)
        dc = dc + dh * self.og * (1 - np.tanh(self.c) ** 2)
        # Compute the gradients of the input gate, forget gate, output gate, and cell state update
        dcu = dc * self.ig * (1 - self.cu ** 2)
        dig = dc * self.cu * self.ig * (1 - self.ig)
        dfg = dc * self.fg * (1 - self.fg)
        dog = dho * np.tanh(self.c) * self.og * (1 - self.og)
        # Concatenate the gradients and backpropagate through the super class
        gradients = np.concatenate([dig, dfg, dog, dcu], axis=1)
        super().backward(gradients)
        # Return the gradients with respect to the input and previous hidden state
        dconcat = self.dinputs
        dstate = dconcat[:, self.input_size:]
        dinputs = dconcat[:, :self.input_size]
        return dinputs, dstate

    def get_parameters(self):
        return self.weights, self.biases





# Conv1D
class Conv1D(Dense):
    def __init__(self, n_inputs, n_filters, filter_size, 
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(filter_size, n_inputs, n_filters)
        self.biases = np.zeros((1, 1, n_filters))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        self.filter_size = filter_size

    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        batch_size, n_inputs, sequence_length = inputs.shape

        # Pad the inputs for valid convolution
        padding = self.filter_size - 1
        inputs_padded = np.pad(inputs, ((0,0), (padding,0), (0,0)), mode='constant')

        # Perform convolution
        output_shape = (batch_size, sequence_length, self.weights.shape[2])
        output = np.zeros(output_shape)
        for i in range(sequence_length):
            input_window = inputs_padded[:, i:i+self.filter_size, :]
            output[:, i, :] = np.dot(input_window, self.weights) + self.biases

        # Store output values
        self.output = output

    def backward(self, dvalues):
        # Gradients on parameters
        batch_size, sequence_length, n_filters = dvalues.shape
        self.dbiases = np.sum(dvalues, axis=(0, 1), keepdims=True)
        self.dweights = np.zeros_like(self.weights)
        inputs_padded = np.pad(self.inputs, ((0,0), (self.filter_size-1,0), (0,0)), mode='constant')
        for i in range(sequence_length):
            input_window = inputs_padded[:, i:i+self.filter_size, :]
            self.dweights += np.dot(input_window.T, dvalues[:, i, :])
        self.dinputs = np.zeros_like(self.inputs)
        for i in range(sequence_length):
            input_window = inputs_padded[:, i:i+self.filter_size, :]
            self.dinputs[:, :, i:i+self.filter_size] += np.dot(dvalues[:, i, :], self.weights.T)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * \
                             self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                            self.biases

    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


# Dropout
class Dropout:

    # Init
    def __init__(self, rate):
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    # Forward pass
    def forward(self, inputs, training):
        # Save input values
        self.inputs = inputs


        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return

        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate,
                           size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask


# Input "layer"
class Input:

    # Forward pass
    def forward(self, inputs, training):
        self.output = inputs