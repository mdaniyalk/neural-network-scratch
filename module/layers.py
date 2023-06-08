import numpy as np
import scipy.signal

class Dense:
    """
    Dense (fully connected) layer implementation.

    This class represents a dense (fully connected) layer in a neural network. It initializes the layer with weights and biases, and provides methods for forward and backward passes, retrieving and setting parameters, and applying regularization.

    Parameters:
    - n_inputs (int): Number of input features.
    - n_neurons (int): Number of neurons in the layer.
    - weight_regularizer_l1 (float): L1 regularization strength for weights.
    - weight_regularizer_l2 (float): L2 regularization strength for weights.
    - bias_regularizer_l1 (float): L1 regularization strength for biases.
    - bias_regularizer_l2 (float): L2 regularization strength for biases.

    Methods:
    - forward(inputs, training): Performs the forward pass of the layer.
    - backward(dvalues): Performs the backward pass of the layer.
    - get_parameters(): Retrieves the weights and biases of the layer.
    - set_parameters(weights, biases): Sets the weights and biases of the layer.

    Example usage:
    layer = Dense(n_inputs=100, n_neurons=50)
    layer.forward(inputs, training=True)
    layer.backward(dvalues)
    weights, biases = layer.get_parameters()
    layer.set_parameters(new_weights, new_biases)
    """

    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        """
        Initializes the dense layer.

        The dense layer is initialized with random weights and zero biases. The regularization strengths for weights and biases can be specified.

        Args:
        - n_inputs (int): Number of input features.
        - n_neurons (int): Number of neurons in the layer.
        - weight_regularizer_l1 (float): L1 regularization strength for weights.
        - weight_regularizer_l2 (float): L2 regularization strength for weights.
        - bias_regularizer_l1 (float): L1 regularization strength for biases.
        - bias_regularizer_l2 (float): L2 regularization strength for biases.
        """

        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training):
        """
        Performs the forward pass of the dense layer.

        The forward pass calculates the output values of the layer from the inputs, weights, and biases. The input values are stored for use in the backward pass.

        Args:
        - inputs (ndarray): Input values.
        - training (bool): Indicates whether the network is in training mode.

        Returns:
        None
        """

        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        """
        Performs the backward pass of the dense layer.

        The backward pass calculates the gradients on parameters and values, taking into account the regularization terms.

        Args:
        - dvalues (ndarray): Gradients of the loss with respect to the output values.

        Returns:
        None
        """

        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_parameters(self):
        """
        Retrieves the weights and biases of the dense layer.

        Returns:
        - weights (ndarray): Weights of the layer.
        - biases (ndarray): Biases of the layer.
        """

        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        """
        Sets the weights and biases of the dense layer.

        Args:
        - weights (ndarray): New weights for the layer.
        - biases (ndarray): New biases for the layer.

        Returns:
        None
        """

        self.weights = weights
        self.biases = biases


# Flatten
class Flatten():
    """
    Flattens the input data.

    This class represents a layer that flattens the input data by reshaping it into a two-dimensional array. It is commonly used as a preprocessing step in neural networks to convert multidimensional input into a flat representation.

    Example usage:
    flatten_layer = Flatten()
    flatten_layer.forward(inputs, training=True)
    flattened_data = flatten_layer.output
    """

    def __init__(self):
        pass

    def forward(self, inputs, training):
        """
        Performs forward propagation for the flatten layer.

        Parameters:
        - inputs: The input data.
        - training: A flag indicating whether the network is in training mode.

        This method takes the input data and reshapes it into a two-dimensional array. The reshaped data is stored in the `output` attribute of the layer.

        Note: The `training` parameter is unused in this layer.

        Example usage:
        flatten_layer.forward(inputs, training=True)
        """

        self.inputs = inputs
        batch_size = inputs.shape[0]
        self.output = inputs.reshape(batch_size, -1)

    def backward(self, dvalues):
        """
        Performs backward propagation for the flatten layer.

        Parameters:
        - dvalues: The gradient of the loss with respect to the layer's outputs.

        This method reshapes the gradient `dvalues` to match the shape of the original input and stores it in the `dinputs` attribute of the layer.

        Example usage:
        flatten_layer.backward(dvalues)
        """

        self.dinputs = dvalues.reshape(self.inputs.shape)

    def get_parameters(self):
        """
        Returns the parameters of the flatten layer.

        Since the flatten layer does not have any learnable parameters, this method returns `None`.

        Example usage:
        params = flatten_layer.get_parameters()
        """

        return None

    def set_parameters(self, weights, biases):
        """
        Sets the parameters of the flatten layer.

        Since the flatten layer does not have any learnable parameters, this method does nothing.

        Example usage:
        flatten_layer.set_parameters(weights, biases)
        """

        pass




# LSTM
class LSTM(Dense):
    """
    Long Short-Term Memory (LSTM) layer implementation.

    This class inherits from the `Dense` class and extends it to implement the LSTM layer. The LSTM layer is a type of recurrent neural network (RNN) layer that is capable of capturing long-term dependencies in sequential data.

    Parameters:
    - input_size: The size of the input to the LSTM layer.
    - hidden_size: The number of hidden units in the LSTM layer.

    The LSTM layer is initialized by calling the constructor of the `Dense` class with modified input and output sizes. The `hidden_size` parameter determines the number of hidden units in the LSTM layer.

    The forward method performs the forward pass of the LSTM layer. It takes the inputs and previous hidden state as arguments and computes the new hidden state and output. The method applies the sigmoid and tanh activations to different parts of the concatenated input and updates the cell state and output based on these activations.

    The backward method performs the backward pass of the LSTM layer. It takes the gradients of the output and cell state with respect to the loss and computes the gradients of the input gate, forget gate, output gate, and cell state update. The method then backpropagates these gradients through the `Dense` class to update the weights and biases.

    The get_parameters method returns the weights and biases of the LSTM layer.

    Example usage:
    lstm = LSTM(input_size=10, hidden_size=32)
    lstm.forward(inputs, state)
    lstm.backward(dh, dc)
    weights, biases = lstm.get_parameters()
    """

    def __init__(self, input_size, hidden_size):
        """
        Initialize the LSTM layer.

        Args:
        - input_size: The size of the input to the LSTM layer.
        - hidden_size: The number of hidden units in the LSTM layer.

        The LSTM layer is initialized by calling the constructor of the `Dense` class with modified input and output sizes. The `hidden_size` parameter determines the number of hidden units in the LSTM layer.
        """
        super().__init__(input_size + hidden_size, 4 * hidden_size)
        self.hidden_size = hidden_size

    def forward(self, inputs, state):
        """
        Perform the forward pass of the LSTM layer.

        Args:
        - inputs: The input to the LSTM layer.
        - state: The previous hidden state of the LSTM layer.

        Returns:
        - output: The output of the LSTM layer.
        - new_state: The new hidden state of the LSTM layer.

        The forward method performs the forward pass of the LSTM layer. It takes the inputs and previous hidden state as arguments and computes the new hidden state and output. The method applies the sigmoid and tanh activations to different parts of the concatenated input and updates the cell state and output based on these activations.
        """
        concatenated = np.concatenate([inputs, state], axis=1)
        super().forward(concatenated, training=True)
        self.ig, self.fg, self.og, self.cu = np.split(self.output, 4, axis=1)
        self.ig = 1 / (1 + np.exp(-self.ig))
        self.fg = 1 / (1 + np.exp(-self.fg))
        self.og = 1 / (1 + np.exp(-self.og))
        self.cu = np.tanh(self.cu)
        self.c = self.fg * state + self.ig * self.cu
        self.h = self.og * np.tanh(self.c)
        return self.h, self.c

    def backward(self, dh, dc):
        """
        Perform the backward pass of the LSTM layer.

        Args:
        - dh: The gradients of the output with respect to the loss.
        - dc: The gradients of the cell state with respect to the loss.

        Returns:
        - dinputs: The gradients with respect to the input to the LSTM layer.
        - dstate: The gradients with respect to the previous hidden state of the LSTM layer.

        The backward method performs the backward pass of the LSTM layer. It takes the gradients of the output and cell state with respect to the loss and computes the gradients of the input gate, forget gate, output gate, and cell state update. The method then backpropagates these gradients through the `Dense` class to update the weights and biases.
        """
        dho = dh * np.tanh(self.c)
        dc = dc + dh * self.og * (1 - np.tanh(self.c) ** 2)
        dcu = dc * self.ig * (1 - self.cu ** 2)
        dig = dc * self.cu * self.ig * (1 - self.ig)
        dfg = dc * self.fg * (1 - self.fg)
        dog = dho * np.tanh(self.c) * self.og * (1 - self.og)
        gradients = np.concatenate([dig, dfg, dog, dcu], axis=1)
        super().backward(gradients)
        dconcat = self.dinputs
        dstate = dconcat[:, self.input_size:]
        dinputs = dconcat[:, :self.input_size]
        return dinputs, dstate

    def get_parameters(self):
        """
        Get the parameters of the LSTM layer.

        Returns:
        - weights: The weights of the LSTM layer.
        - biases: The biases of the LSTM layer.

        The get_parameters method returns the weights and biases of the LSTM layer.
        """
        return self.weights, self.biases






# Conv1D
class Conv1D(Dense):
    """
    1D Convolutional Layer.

    This class represents a 1D convolutional layer, which performs convolutional operations on the input data. It is a subclass of the `Dense` class, inheriting its functionalities.

    Parameters:
    - n_inputs: The number of input neurons.
    - n_filters: The number of filters.
    - filter_size: The size of the convolutional filters.
    - stride: The stride value for the convolutional operation. Default is 1.
    - weight_regularizer_l1: L1 regularization factor for the weights. Default is 0.
    - weight_regularizer_l2: L2 regularization factor for the weights. Default is 0.
    - bias_regularizer_l1: L1 regularization factor for the biases. Default is 0.
    - bias_regularizer_l2: L2 regularization factor for the biases. Default is 0.

    The `Conv1D` class extends the `Dense` class to support 1D convolutional operations. It initializes the weights and biases using the `Dense` class's initialization method. The weights are reshaped to match the convolutional filter shape. The class also stores other attributes such as the filter size and stride.

    The `forward` method performs the forward pass of the convolutional layer. It takes the inputs and training flag as parameters. The method calculates the output shape based on the inputs and filter size. It performs convolutional operations on each batch and filter, using the `scipy.signal.convolve` function. Biases are added to the output, and the output is reshaped accordingly.

    The `backward` method performs the backward pass of the convolutional layer. It takes the gradient values (`dvalues`) as a parameter. The method calculates gradients on the parameters (weights and biases) based on the input and output shapes. It also calculates gradients on the input values.

    The `get_parameters` method returns the reshaped weights and biases of the convolutional layer.

    The `set_parameters` method sets the weights and biases of the convolutional layer after reshaping them.

    Example usage:
    conv_layer = Conv1D(n_inputs=32, n_filters=64, filter_size=3)
    output = conv_layer.forward(inputs, training=True)
    conv_layer.backward(dvalues)
    weights, biases = conv_layer.get_parameters()
    conv_layer.set_parameters(new_weights, new_biases)
    """

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
        """
        Perform forward pass for the convolutional layer.

        Parameters:
        - inputs: The input data.
        - training: A flag indicating whether the network is in training mode.

        Returns:
        - output: The output of the convolutional layer.

        This method performs the forward pass for the convolutional layer. It takes the inputs and training flag as parameters. The method calculates the output shape based on the inputs and filter size. It performs convolutional operations on each batch and filter, using the `scipy.signal.convolve` function. Biases are added to the output, and the output is reshaped accordingly.

        Example usage:
        output = conv_layer.forward(inputs, training=True)
        """

        if len(inputs.shape) == 2:
            inputs = np.expand_dims(inputs, axis=-1)
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
        """
        Perform backward pass for the convolutional layer.

        Parameters:
        - dvalues: The gradient values from the next layer.

        This method performs the backward pass for the convolutional layer. It takes the gradient values (`dvalues`) as a parameter. The method calculates gradients on the parameters (weights and biases) based on the input and output shapes. It also calculates gradients on the input values.

        Example usage:
        conv_layer.backward(dvalues)
        """

        # Calculate gradients on parameters
        batch_size, output_length, n_filters = dvalues.shape

        self.dbiases = np.sum(dvalues, axis=(0, 1), keepdims=True)

        dvalues_reshaped = np.transpose(dvalues, axes=(0, 2, 1))

        self.dweights = np.zeros_like(self.weights)
        for i in range(self.filter_size):
            try:
                self.dweights[:, i] = np.sum(self.inputs[:, i:i + output_length, 0] * dvalues_reshaped, axis=(0, 2))
            except:
                self.dweights[:, i] = np.sum(np.transpose(np.expand_dims(self.inputs[:, i:i + output_length, 0], axis=-1), (0, 2, 1)) * dvalues_reshaped, axis=(0,2))
            else:
                self.dweights[:, i] = np.sum(self.inputs[:, i:i + output_length, 0] * dvalues_reshaped, axis=(0, 2))

        # Calculate gradients on input values
        self.dinputs = np.zeros_like(self.inputs, dtype=np.float64)  # Explicitly set dtype to float64
        for i in range(self.filter_size):
            weight_reshaped = np.expand_dims(self.weights[:, i], axis=1)
            try:
                self.dinputs[:, i:i + output_length, 0] += np.sum(
                    dvalues_reshaped * np.expand_dims(weight_reshaped, axis=2), axis=1
                )

            except:
                self.dinputs[:, i:i + output_length, 0] += np.sum(dvalues_reshaped * weight_reshaped, axis=1)

            else:
                self.dinputs[:, i:i + output_length, 0] += np.sum(
                    dvalues_reshaped * np.expand_dims(weight_reshaped, axis=2), axis=1
                )
        self.dinputs = self.dinputs

    def get_parameters(self):
        """
        Get the parameters of the convolutional layer.

        Returns:
        - weights: The reshaped weights of the convolutional layer.
        - biases: The biases of the convolutional layer.

        This method returns the reshaped weights and biases of the convolutional layer.

        Example usage:
        weights, biases = conv_layer.get_parameters()
        """

        # Reshape the weights before returning
        return self.weights.reshape((-1, self.weights.shape[2])), self.biases

    def set_parameters(self, weights, biases):
        """
        Set the parameters of the convolutional layer.

        Parameters:
        - weights: The reshaped weights of the convolutional layer.
        - biases: The biases of the convolutional layer.

        This method sets the weights and biases of the convolutional layer after reshaping them.

        Example usage:
        conv_layer.set_parameters(new_weights, new_biases)
        """

        # Reshape the weights before setting
        self.weights = weights.reshape((self.filter_size, self.weights.shape[1], self.weights.shape[2]))
        self.biases = biases



# MaxPool1D
class MaxPool1D:
    def __init__(self, pool_size, strides):
        """
        Initializes the MaxPool1D layer.

        Parameters:
        - pool_size (int): The size of the pooling window.
        - strides (int): The stride length for pooling.

        This class represents a MaxPool1D layer, which performs 1D max pooling on the input data.

        The `pool_size` parameter specifies the size of the pooling window, and the `strides` parameter determines the stride length for pooling. If `strides` is not provided, it defaults to `pool_size`.

        The `trainable` attribute is set to `False` by default, as the MaxPool1D layer does not have any trainable parameters.

        Example usage:
        max_pool = MaxPool1D(pool_size=2, strides=1)
        """
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size
        self.trainable = False

    def forward(self, inputs, training):
        """
        Performs forward pass computation of the MaxPool1D layer.

        Parameters:
        - inputs (ndarray): The input data.
        - training (bool): Indicates whether the model is in training mode.

        Returns:
        - output (ndarray): The output of the MaxPool1D layer.

        This method performs the forward pass computation of the MaxPool1D layer on the input data. It downsamples the input by applying max pooling with the specified `pool_size` and `strides`.

        The method calculates the output shape based on the input shape, pool size, and strides. It initializes an output array and iterates over the downsampled features, applying max pooling to each feature.

        The method returns the computed output of the MaxPool1D layer.

        Note: This method assumes the input data has a shape of (batch_size, num_features, sequence_length).

        Example usage:
        output = max_pool.forward(inputs, training=True)
        """
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
        """
        Performs backward pass computation of the MaxPool1D layer.

        Parameters:
        - dvalues (ndarray): The gradients of the loss with respect to the layer's output.

        This method performs the backward pass computation of the MaxPool1D layer. It calculates the gradients of the loss with respect to the layer's inputs (`dinputs`) using the chain rule and the gradients of the loss with respect to the layer's output (`dvalues`).

        The method iterates over the downsampled features and applies the mask-based gradient calculation to assign the gradients to the corresponding positions in the input data.

        Note: This method assumes the input data has a shape of (batch_size, num_features, sequence_length).

        Example usage:
        max_pool.backward(dvalues)
        """
        self.dinputs = np.zeros_like(self.inputs)

        batch_size, downsampled_num_features, output_length = dvalues.shape

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
        """
        Returns the parameters of the MaxPool1D layer.

        Returns:
        - None

        This method returns None, as the MaxPool1D layer does not have any trainable parameters.

        Example usage:
        params = max_pool.get_parameters()
        """
        return None

    def set_parameters(self, weights, biases):
        """
        Sets the parameters of the MaxPool1D layer.

        Parameters:
        - weights: Not applicable.
        - biases: Not applicable.

        This method does not set any parameters for the MaxPool1D layer.

        Example usage:
        max_pool.set_parameters(weights, biases)
        """
        pass

           



# Dropout
class Dropout:
    """
    Dropout layer for neural networks.

    Args:
    - rate (float): The dropout rate, indicating the fraction of input units to drop during training.

    This class implements a dropout layer, which is a regularization technique commonly used in neural networks. During training, dropout randomly sets a fraction of the input units to 0 at each update, which helps prevent overfitting by reducing the interdependencies between neurons.

    The class has two main methods: `forward()` and `backward()`. The `forward()` method takes the input values and applies dropout based on the specified rate. The `backward()` method calculates the gradients on the input values based on the dropout mask.

    Example usage:
    dropout = Dropout(rate=0.2)
    dropout.forward(inputs, training=True)
    dropout.backward(dvalues)
    """

    def __init__(self, rate):
        """
        Initialize the Dropout layer.

        Args:
        - rate (float): The dropout rate, indicating the fraction of input units to drop during training.

        The rate is stored as `self.rate`, and it is inverted to represent the success rate (`1 - rate`). This inversion is necessary because for dropout rate of `0.1`, we need a success rate of `0.9` (keeping `90%` of the input units).
        """
        self.rate = 1 - rate

    def forward(self, inputs, training):
        """
        Apply dropout during the forward pass.

        Args:
        - inputs (ndarray): The input values to the dropout layer.
        - training (bool): Indicates whether the model is in training mode.

        During the forward pass, this method saves the input values in `self.inputs`. If the model is not in training mode, it returns the input values without applying dropout.

        If the model is in training mode, the method generates a binary mask of the same shape as the inputs using `np.random.binomial()`. The mask is scaled by dividing it by the success rate (`self.rate`). Then, the method applies the mask to the input values to create the output.

        The output values are saved in `self.output`.
        """
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        """
        Calculate the gradients on the input values during the backward pass.

        Args:
        - dvalues (ndarray): The gradient of the loss with respect to the output values.

        During the backward pass, this method calculates the gradients on the input values (`dinputs`) based on the dropout mask (`self.binary_mask`). The gradients are calculated as the element-wise product of the input gradients (`dvalues`) and the binary mask.

        The calculated gradients on the input values are saved in `self.dinputs`.
        """
        self.dinputs = dvalues * self.binary_mask



# Input "layer"
class Input:
    """
    A class representing an input layer in a neural network.

    The `Input` class is used to define an input layer in a neural network. It serves as the starting point of the network, taking input data and passing it through the network during the forward pass.

    Example usage:
    input_layer = Input()
    output = input_layer.forward(inputs, training)
    """

    def forward(self, inputs, training):
        """
        Forward pass of the input layer.

        Parameters:
        - inputs: The input data to be passed through the layer.
        - training: A flag indicating whether the model is in training mode or not.

        Returns:
        - output: The output of the input layer, which is the same as the input data.

        This method performs the forward pass of the input layer. It takes the input data and assigns it as the output of the layer. During training, this layer does not perform any computations or transformations on the input data.

        The method takes two parameters: `inputs`, which represents the input data, and `training`, which is a boolean flag indicating whether the model is in training mode or not. This flag can be used for certain operations that are only applied during training.

        The output of the input layer is the same as the input data, and it is stored in the `output` attribute of the instance.

        Example usage:
        input_layer = Input()
        output = input_layer.forward(inputs, training)
        """

        self.output = inputs
        return self.output
