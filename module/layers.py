import numpy as np
import scipy.signal

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

# Flatten

class Flatten():
    def __init__(self):
        pass
    def forward(self, inputs, training):
        self.inputs = inputs
        batch_size = inputs.shape[0]
        self.output = inputs.reshape(batch_size, -1)

    def backward(self, dvalues):
        self.dinputs = dvalues.reshape(self.inputs.shape)

    def get_parameters(self):
        return None  # Flatten layer has no parameters, so return None

    def set_parameters(self, weights, biases):
        pass  # Flatten layer has no parameters, so do nothing



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
    def __init__(self, n_inputs, n_filters, filter_size, stride=1,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Calculate the number of output neurons
        n_neurons = n_filters
        self.n_filters = n_filters

        # Initialize weights and biases using the Dense layer
        super().__init__(n_inputs, n_neurons, weight_regularizer_l1, weight_regularizer_l2,
                         bias_regularizer_l1, bias_regularizer_l2)

        # Reshape the weights to match the convolutional filter shape
        self.weights = 0.01 * np.random.randn(n_filters, filter_size)

        # Set other attributes
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs

        # Calculate output shape
        batch_size, n_inputs, sequence_length = inputs.shape
        output_length = (n_inputs - self.filter_size) // self.stride + 1

        self.output = np.zeros((batch_size, output_length, self.n_filters))


        for i in range(batch_size):
            for j in range(self.n_filters):
                self.output[i, :, j] = scipy.signal.convolve(self.inputs[i, :, 0], self.weights[j, :], mode='valid')

        # Add biases and reshape the output
        self.output += np.reshape(self.biases, (1, 1, self.n_filters))

        return self.output



    def backward(self, dvalues):
        # Calculate gradients on parameters
        batch_size, output_length, n_filters = dvalues.shape

        self.dbiases = np.sum(dvalues, axis=(0, 1), keepdims=True)

        dvalues_reshaped = np.transpose(dvalues, axes=(0, 2, 1))

        self.dweights = np.zeros_like(self.weights)
        for i in range(self.filter_size):
            self.dweights[:, i] = np.sum(self.inputs[:, i:i + output_length, 0] * dvalues_reshaped, axis=(0, 2))

        # Calculate gradients on input values
        self.dinputs = np.zeros_like(self.inputs, dtype=np.float64)  # Explicitly set dtype to float64
        for i in range(self.filter_size):
            weight_reshaped = np.expand_dims(self.weights[:, i], axis=1)
            self.dinputs[:, i:i + output_length, 0] += np.sum(
                dvalues_reshaped * np.expand_dims(weight_reshaped, axis=2), axis=1
            )
        self.dinputs = self.dinputs

    def get_parameters(self):
        # Reshape the weights before returning
        return self.weights.reshape((-1, self.weights.shape[2])), self.biases

    def set_parameters(self, weights, biases):
        # Reshape the weights before setting
        self.weights = weights.reshape((self.filter_size, self.weights.shape[1], self.weights.shape[2]))
        self.biases = biases


# MaxPool1D
class MaxPool1D:
    def __init__(self, pool_size, strides):
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size
        self.trainable = False

    def forward(self, inputs, training):
        self.inputs = inputs

        batch_size, num_features, sequence_length = inputs.shape
        downsampled_num_features = (num_features - self.pool_size) // self.strides + 1
        output_length = sequence_length

        self.output = np.zeros((batch_size, downsampled_num_features, output_length))

        for i in range(downsampled_num_features):
            start = i * self.strides
            end = start + self.pool_size
            self.output[:, i, :] = np.max(inputs[:, start:end, :], axis=1)

        return self.output

    def backward(self, dvalues):
        self.dinputs = np.zeros_like(self.inputs)

        batch_size, downsampled_num_features, output_length = dvalues.shape
        # num_features = downsampled_num_features * self.pool_size

        for i in range(downsampled_num_features):
            start = i * self.strides
            end = start + self.pool_size

            pool_slice = self.inputs[:, start:end, :]
            pool_slice_reshaped = pool_slice.reshape(batch_size, self.pool_size, output_length)

            max_values = np.max(pool_slice_reshaped, axis=1, keepdims=True)
            mask = (pool_slice_reshaped == max_values)

            dvalues_reshaped = np.expand_dims(dvalues[:, i, :], axis=1)
            self.dinputs[:, start:end, :] += mask * dvalues_reshaped

    def get_parameters(self):
        return None  # Flatten layer has no parameters, so return None

    def set_parameters(self, weights, biases):
        pass  # Flatten layer has no parameters, so do nothing
           



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