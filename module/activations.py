import numpy as np



class ReLU:
    """
    Rectified Linear Unit (ReLU) activation function.

    The ReLU class implements the ReLU activation function, which applies an element-wise operation on the inputs, setting negative values to zero and leaving positive values unchanged.

    Example usage:
    relu = ReLU()
    output = relu.forward(inputs, training=True)
    relu.backward(dvalues)
    predictions = relu.predictions(outputs)
    """


    def forward(self, inputs, training):
        """
        Forward pass of the ReLU activation function.

        Args:
        - inputs (ndarray): A numpy array of input values.
        - training: A boolean flag indicating whether the network is in training mode.

        Returns:
        None

        This method performs the forward pass of the ReLU activation function on the given inputs. The ReLU activation function applies an element-wise maximum operation, setting all negative values to zero.

        The method stores the input values in `self.inputs` and computes the output values by applying `np.maximum(0, inputs)`.

        Example usage:
        relu = ReLU()
        relu.forward(inputs, training=True)
        """

        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        """
        Backward pass of the ReLU activation function.

        Args:
        - dvalues (ndarray): A numpy array of gradient values.

        Returns:
        None

        This method performs the backward pass of the ReLU activation function. It computes the gradient of the loss with respect to the inputs by copying the `dvalues` array and setting the gradients to zero where the corresponding input values were less than or equal to zero.

        The method stores the computed gradients in `self.dinputs`.

        Example usage:
        relu = ReLU()
        relu.backward(dvalues)
        """

        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        """
        Returns predictions for the given outputs.

        Args:
        - outputs (ndarray): A numpy array of output values.

        Returns:
        - outputs (ndarray): A numpy array of output values.

        This method returns the predictions for the given outputs. In the case of ReLU activation, the predictions are the same as the outputs.

        Example usage:
        relu = ReLU()
        predictions = relu.predictions(outputs)
        """

        return outputs




class Softmax:
    """
    Softmax activation function and its derivatives.

    This class implements the forward and backward pass for the Softmax activation function, as well as a method for making predictions based on the output values.

    Methods:
    - forward(inputs, training): Performs the forward pass of the Softmax activation function.
    - backward(dvalues): Performs the backward pass of the Softmax activation function.
    - predictions(outputs): Returns the predicted classes based on the output values.

    Example usage:
    softmax = Softmax()
    softmax.forward(inputs, training=True)
    softmax.backward(dvalues)
    softmax_predictions = softmax.predictions(outputs)
    """

    def forward(self, inputs, training):
        """
        Performs the forward pass of the Softmax activation function.

        Parameters:
        - inputs (ndarray): The input values.
        - training (bool): Specifies if the forward pass is performed during training.

        This method takes the input values and applies the Softmax activation function to produce the output probabilities. The Softmax function is defined as the exponential of each input value divided by the sum of exponentials over all input values.

        The method stores the input values and the computed output probabilities in `self.inputs` and `self.output`, respectively.

        Example usage:
        softmax.forward(inputs, training=True)
        """
        self.inputs = inputs

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        """
        Performs the backward pass of the Softmax activation function.

        Parameters:
        - dvalues (ndarray): The gradients of the loss with respect to the output values.

        This method calculates the gradients of the loss with respect to the input values using the derivative of the Softmax function. The derivative of Softmax with respect to each input value is given by the Jacobian matrix, which is the diagonal matrix formed by the output probabilities minus the outer product of the probabilities with itself.

        The method stores the computed gradients of the loss with respect to the input values in `self.dinputs`.

        Example usage:
        softmax.backward(dvalues)
        """
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs):
        """
        Returns the predicted classes based on the output values.

        Parameters:
        - outputs (ndarray): The output values.

        Returns:
        - predictions (ndarray): The predicted classes.

        This method returns the predicted classes based on the output values. It selects the class with the highest probability for each input sample.

        Example usage:
        softmax_predictions = softmax.predictions(outputs)
        """
        return np.argmax(outputs, axis=1)




class Sigmoid:
    """
    Sigmoid activation function.

    The Sigmoid class represents the sigmoid activation function, which maps the input values to a range between 0 and 1. It is commonly used in neural networks to introduce non-linearity.

    The class provides methods to perform the forward pass, backward pass, and generate predictions using the sigmoid activation function.

    Example usage:
    sigmoid = Sigmoid()
    sigmoid.forward(inputs, training=True)
    sigmoid.backward(dvalues)
    sigmoid_predictions = sigmoid.predictions(outputs)
    """

    def forward(self, inputs, training=True):
        """
        Performs the forward pass of the sigmoid activation function.

        Parameters:
        - inputs (ndarray): The input values.
        - training (bool): A flag indicating if the forward pass is performed during training.

        This method performs the forward pass of the sigmoid activation function. It takes the input values and computes the sigmoid activation using the formula 1 / (1 + exp(-inputs)).

        The method stores the inputs and the computed output in the object's attributes for later use in the backward pass.

        Example usage:
        sigmoid = Sigmoid()
        sigmoid.forward(inputs, training=True)
        """

        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        """
        Performs the backward pass of the sigmoid activation function.

        Parameters:
        - dvalues (ndarray): The gradient of the loss with respect to the output values.

        This method performs the backward pass of the sigmoid activation function. It takes the gradient of the loss with respect to the output values and computes the gradient of the loss with respect to the inputs using the formula dvalues * (1 - output) * output.

        The method stores the computed gradient of the loss with respect to the inputs in the object's attribute for later use in backpropagation.

        Example usage:
        sigmoid = Sigmoid()
        sigmoid.backward(dvalues)
        """

        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        """
        Generates predictions based on the sigmoid output.

        Parameters:
        - outputs (ndarray): The output values.

        Returns:
        - predictions (ndarray): The binary predictions.

        This method generates predictions based on the sigmoid output values. It takes the output values and applies a threshold of 0.5 to classify the predictions as 0 or 1.

        The method returns the binary predictions as an ndarray.

        Example usage:
        sigmoid = Sigmoid()
        sigmoid_predictions = sigmoid.predictions(outputs)
        """

        return (outputs > 0.5) * 1



# Linear activation
class Linear:
    """
    Linear layer implementation.

    This class represents a linear layer in a neural network. It performs the forward pass, backward pass, and predictions for the layer.

    Methods:
    - forward(inputs, training): Performs the forward pass of the linear layer.
    - backward(dvalues): Performs the backward pass of the linear layer.
    - predictions(outputs): Returns the predictions given the outputs.

    Example usage:
    linear_layer = Linear()
    linear_layer.forward(inputs, training=True)
    linear_layer.backward(dvalues)
    preds = linear_layer.predictions(outputs)
    """

    def forward(self, inputs, training):
        """
        Performs the forward pass of the linear layer.

        Parameters:
        - inputs: The input data.
        - training: A boolean flag indicating whether the model is in training mode.

        This method sets the inputs and output attributes of the linear layer to the provided inputs. In the case of a linear layer, the output is the same as the inputs.

        Example usage:
        linear_layer.forward(inputs, training=True)
        """

        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        """
        Performs the backward pass of the linear layer.

        Parameters:
        - dvalues: The gradients of the loss with respect to the layer's output.

        This method sets the dinputs attribute of the linear layer to the provided dvalues.

        Example usage:
        linear_layer.backward(dvalues)
        """

        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        """
        Returns the predictions given the outputs.

        Parameters:
        - outputs: The output values of the linear layer.

        Returns:
        - predictions: The predictions based on the outputs.

        This method returns the outputs as the predictions. In the case of a linear layer, there is no activation function applied, so the outputs are considered as predictions.

        Example usage:
        preds = linear_layer.predictions(outputs)
        """

        return outputs



