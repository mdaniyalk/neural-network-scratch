import numpy as np



class ReLU:
    """
    Rectified Linear Unit (ReLU) activation function.
    """

    def forward(self, inputs, training):
        """
        Forward pass of ReLU activation function.

        Args:
        inputs (ndarray): A numpy array of input values.
        training: A boolean flag indicating whether the network is in training mode.

        Returns:
        None
        """
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        """
        Backward pass of ReLU activation function.

        Args:
        dvalues (ndarray): A numpy array of gradient values.

        Returns:
        None
        """
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        """
        Returns predictions for outputs.

        Args:
        outputs (ndarray): A numpy array of output values.

        Returns:
        outputs (ndarray): A numpy array of output values.
        """
        return outputs



class Softmax:
    """
    Softmax activation function.
    """
    def forward(self, inputs, training):
        """
        Calculates the forward pass of the Softmax function.

        Args:
        inputs (ndarray): A numpy array of input values to the Softmax function.
        training: A boolean flag indicating if the network is training or testing.

        Returns:
        None
        """
        self.inputs = inputs

        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))

        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        """
        Calculates the backward pass of the Softmax function.

        Args:
        dvalues (ndarray): A numpy array of derivative values.

        Returns:
        None
        """
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)

    def predictions(self, outputs):
        """
        Calculates the predictions for the outputs of the Softmax function.

        Args:
        outputs (ndarray): A numpy array of outputs from the Softmax function.

        Returns:
        outputs (ndarray): A numpy array of output values.
        """
        return np.argmax(outputs, axis=1)



class Sigmoid:
    """
    Sigmoid activation function class.
    """
    def forward(self, inputs, training):
        """
        Calculates the forward pass of the Sigmoid function.

        Args:
        inputs (ndarray): A numpy array of input values to the Sigmoid function.
        training: A boolean flag indicating if the network is training or testing.

        Returns:
        None
        """
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        """
        Calculates the backward pass of the Sigmoid function.

        Args:
        dvalues (ndarray): A numpy array of derivative values.

        Returns:
        None
        """
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        """
        Calculates the predictions for the outputs of the Sigmoid function.

        Args:
        outputs (ndarray): A numpy array of outputs from the Sigmoid function.

        Returns:
        outputs (ndarray): A numpy array of output values.
        """
        return (outputs > 0.5) * 1


# Linear activation
class Linear:
    """
    Linear activation function class.
    """
    def forward(self, inputs, training):
        """
        Calculates the forward pass of the Linear function.

        Args:
        inputs (ndarray): A numpy array of input values to the Linear function.
        training: A boolean flag indicating if the network is training or testing.

        Returns:
        None
        """
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        """
        Calculates the backward pass of the Linear function.

        Args:
        dvalues (ndarray): A numpy array of derivative values.

        Returns:
        None
        """
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        """
        Calculates the predictions for the outputs of the Linear function.

        Args:
        outputs (ndarray): A numpy array of outputs from the Linear function.

        Returns:
        outputs (ndarray): A numpy array of output values.
        """
        return outputs


