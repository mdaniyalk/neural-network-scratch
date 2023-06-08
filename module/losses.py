import numpy as np


# Common loss class
class Loss:
    """
    Loss class for calculating and managing data and regularization losses.

    This class provides methods to calculate data and regularization losses, as well as methods to manage accumulated losses during training.

    Attributes:
    - trainable_layers: A list of trainable layers in the model.
    - accumulated_sum: The accumulated sum of losses.
    - accumulated_count: The accumulated count of samples.

    Methods:
    - regularization_loss(): Calculates the regularization loss based on the weights and biases of the trainable layers.
    - remember_trainable_layers(trainable_layers): Sets the list of trainable layers for the loss calculation.
    - calculate(output, y, include_regularization=False): Calculates the data and regularization losses given the model output and ground truth values.
    - calculate_accumulated(include_regularization=False): Calculates the accumulated loss.
    - new_pass(): Resets the variables for accumulated loss.

    Example usage:
    loss = Loss()
    loss.remember_trainable_layers(model.trainable_layers)
    loss.calculate(output, y, include_regularization=True)
    """

    def regularization_loss(self):
        """
        Calculates the regularization loss based on the weights and biases of the trainable layers.

        Returns:
        - regularization_loss: The calculated regularization loss.

        This method iterates over all the trainable layers and calculates the regularization loss. It considers L1 and L2 regularization for both weights and biases.

        The regularization loss is computed by multiplying the regularization factors with the corresponding weights or biases and summing them up.

        Example usage:
        regularization_loss = loss.regularization_loss()
        """

        regularization_loss = 0

        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                                       np.sum(np.abs(layer.weights))

            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                                       np.sum(layer.weights * layer.weights)

            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                                       np.sum(np.abs(layer.biases))

            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                                       np.sum(layer.biases * layer.biases)

        return regularization_loss

    def remember_trainable_layers(self, trainable_layers):
        """
        Sets the list of trainable layers for the loss calculation.

        Args:
        - trainable_layers: A list of trainable layers.

        This method is used to set or remember the list of trainable layers in the model. The list is later used for calculating the regularization loss.

        Example usage:
        loss.remember_trainable_layers(model.trainable_layers)
        """

        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization=False):
        """
        Calculates the data and regularization losses given the model output and ground truth values.

        Args:
        - output: The model output.
        - y: The ground truth values.
        - include_regularization: Whether to include regularization loss in the calculation (default: False).

        Returns:
        - data_loss: The calculated data loss.
        - regularization_loss: The calculated regularization loss, if include_regularization is True.

        This method calculates the data loss based on the model output and ground truth values. It also optionally calculates the regularization loss.

        The sample losses are calculated using the forward method (not shown), and the mean loss is computed. The accumulated sum of losses and sample count are updated for monitoring purposes.

        If include_regularization is False, only the data loss is returned. Otherwise, both the data loss and regularization loss are returned.

        Example usage:
        data_loss = loss.calculate(output, y, include_regularization=True)
        """

        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()

    def calculate_accumulated(self, *, include_regularization=False):
        """
        Calculates the accumulated loss.

        Args:
        - include_regularization: Whether to include regularization loss in the calculation (default: False).

        Returns:
        - data_loss: The calculated accumulated loss.
        - regularization_loss: The calculated regularization loss, if include_regularization is True.

        This method calculates the accumulated loss based on the accumulated sum of losses and sample count. It optionally calculates the regularization loss.

        If include_regularization is False, only the accumulated data loss is returned. Otherwise, both the accumulated data loss and regularization loss are returned.

        Example usage:
        data_loss = loss.calculate_accumulated(include_regularization=True)
        """

        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()

    def new_pass(self):
        """
        Resets the variables for accumulated loss.

        This method is called at the start of a new pass during training to reset the variables for accumulated loss (accumulated sum and accumulated count).

        Example usage:
        loss.new_pass()
        """

        self.accumulated_sum = 0
        self.accumulated_count = 0



# Cross-entropy loss
class CategoricalCrossentropy(Loss):
    """
    Categorical cross-entropy loss function for multi-class classification.

    This class provides the implementation of the categorical cross-entropy loss function for multi-class classification problems. It inherits from the `Loss` base class.

    The categorical cross-entropy loss is commonly used when the targets are one-hot encoded. It measures the dissimilarity between the predicted probabilities and the true target probabilities. The loss is defined as the negative log likelihood of the correct class probabilities.

    The class defines two methods: `forward` for the forward pass and `backward` for the backward pass.

    Example usage:
    loss_func = CategoricalCrossentropy()
    loss = loss_func.forward(y_pred, y_true)
    loss_func.backward(dvalues, y_true)
    """

    def forward(self, y_pred, y_true):
        """
        Computes the forward pass of the categorical cross-entropy loss.

        Parameters:
        - y_pred (ndarray): The predicted probabilities.
        - y_true (ndarray): The true target probabilities or one-hot encoded labels.

        Returns:
        - negative_log_likelihoods (ndarray): The negative log likelihoods.

        This method computes the forward pass of the categorical cross-entropy loss. It takes the predicted probabilities `y_pred` and the true target probabilities or one-hot encoded labels `y_true` as inputs.

        The method first clips the predicted probabilities to prevent division by zero. It then calculates the correct confidences by either selecting the predicted probabilities corresponding to the true labels (for categorical labels) or by taking the dot product between the predicted probabilities and the one-hot encoded labels (for one-hot encoded labels).

        The negative log likelihoods are calculated as the negative logarithm of the correct confidences. Finally, the negative log likelihoods are returned.

        Note: This method assumes the use of NumPy for array operations.

        Example usage:
        loss = forward(y_pred, y_true)
        """

        samples = len(y_pred)  # Number of samples in a batch

        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            # Probabilities for target values (categorical labels)
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            # Mask values (one-hot encoded labels)
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        """
        Computes the backward pass of the categorical cross-entropy loss.

        Parameters:
        - dvalues (ndarray): The gradients of the loss with respect to the predictions.
        - y_true (ndarray): The true target probabilities or one-hot encoded labels.

        This method computes the backward pass of the categorical cross-entropy loss. It takes the gradients `dvalues` and the true target probabilities or one-hot encoded labels `y_true` as inputs.

        The method calculates the number of samples and the number of labels. If the labels are sparse, it converts them into a one-hot encoded vector. The gradient is then computed by dividing the one-hot encoded labels by the gradients. Finally, the gradient is normalized by dividing it by the number of samples.

        Example usage:
        backward(dvalues, y_true)
        """

        samples = len(dvalues)  # Number of samples
        labels = len(dvalues[0])  # Number of labels in every sample

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]  # Convert labels to one-hot vector

        self.dinputs = -y_true / dvalues  # Calculate gradient
        self.dinputs = self.dinputs / samples  # Normalize gradient



# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Softmax_Loss_CategoricalCrossentropy():
    """
    Softmax Loss Categorical Crossentropy.

    This class represents the Softmax Loss Categorical Crossentropy. It is commonly used as the loss function in multi-class classification problems when the output is passed through a softmax activation function.

    The backward pass of this class calculates the gradient of the loss with respect to the inputs (dvalues).

    Example usage:
    loss = Softmax_Loss_CategoricalCrossentropy()
    loss.backward(dvalues, y_true)
    """

    def backward(self, dvalues, y_true):
        """
        Performs the backward pass to calculate the gradient of the loss with respect to the inputs (dvalues).

        Parameters:
        - dvalues: The gradient of the loss with respect to the outputs.
        - y_true: The true labels/targets.

        This method performs the backward pass to calculate the gradient of the loss with respect to the inputs (dvalues). It supports both one-hot encoded labels and discrete labels.

        If the labels are one-hot encoded, they are converted into discrete values by taking the argmax along the axis 1. The method then calculates the gradient by subtracting 1 from the dvalues corresponding to the true labels. Finally, the gradient is normalized by dividing it by the number of samples.

        Example usage:
        loss = Softmax_Loss_CategoricalCrossentropy()
        loss.backward(dvalues, y_true)
        """

        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples



# Binary cross-entropy loss
class BinaryCrossentropy(Loss):
    """
    Binary cross-entropy loss function.

    This class implements the binary cross-entropy loss, which is commonly used in binary classification tasks. The loss measures the dissimilarity between predicted and true binary values.

    The class inherits from the base `Loss` class.

    Example usage:
    loss = BinaryCrossentropy()
    y_pred = np.array([0.2, 0.8, 0.4])
    y_true = np.array([0, 1, 1])
    loss_value = loss.forward(y_pred, y_true)
    loss.backward()
    """

    def forward(self, y_pred, y_true):
        """
        Performs the forward pass of the binary cross-entropy loss.

        Parameters:
        - y_pred (ndarray): The predicted binary values.
        - y_true (ndarray): The true binary values.

        Returns:
        - sample_losses (ndarray): The sample-wise losses.

        This method calculates the forward pass of the binary cross-entropy loss. It takes the predicted binary values `y_pred` and the true binary values `y_true`. The predicted values are clipped to prevent division by zero and to avoid dragging the mean towards any value. The sample-wise loss is then calculated as the negative log likelihood of the predicted values for the true labels. Finally, the mean of the sample losses is returned.

        Note: This implementation assumes the use of NumPy.

        Example usage:
        loss = BinaryCrossentropy()
        y_pred = np.array([0.2, 0.8, 0.4])
        y_true = np.array([0, 1, 1])
        sample_losses = loss.forward(y_pred, y_true)
        """

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        # Return losses
        return sample_losses

    def backward(self, dvalues, y_true):
        """
        Performs the backward pass of the binary cross-entropy loss.

        Parameters:
        - dvalues (ndarray): The gradient of the loss with respect to the predicted values.
        - y_true (ndarray): The true binary values.

        This method calculates the backward pass of the binary cross-entropy loss. It takes the gradient `dvalues` of the loss with respect to the predicted values and the true binary values `y_true`. The number of samples and the number of outputs are extracted from the input arrays. The gradient is calculated as the derivative of the loss function with respect to the predicted values, and it is normalized by the number of outputs. Finally, the gradient is divided by the number of samples.

        Example usage:
        loss = BinaryCrossentropy()
        dvalues = np.array([0.1, 0.5, -0.3])
        y_true = np.array([0, 1, 1])
        loss.backward(dvalues, y_true)
        """

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues -
                         (1 - y_true) / (1 - clipped_dvalues)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples



# Mean Squared Error loss
class MeanSquaredError(Loss):
    """
    Mean Squared Error (MSE) loss function.

    This class represents the Mean Squared Error loss, also known as L2 loss. It is a commonly used loss function for regression problems. The MSE loss measures the average squared difference between the predicted and true values.

    The class inherits from the `Loss` base class, which provides the structure and methods required for a loss function.

    Methods:
    - forward(y_pred, y_true): Performs the forward pass and calculates the MSE loss.
    - backward(dvalues, y_true): Performs the backward pass and calculates the gradients.

    Example usage:
    loss = MeanSquaredError()
    mse = loss.forward(y_pred, y_true)
    loss.backward(dvalues, y_true)
    """

    def forward(self, y_pred, y_true):
        """
        Performs the forward pass and calculates the Mean Squared Error loss.

        Parameters:
        - y_pred: The predicted values.
        - y_true: The true values.

        Returns:
        - sample_losses: The losses for each sample.

        This method calculates the MSE loss by computing the average squared difference between the predicted values `y_pred` and the true values `y_true`. It uses the `np.mean()` function to compute the mean of the squared differences along the last axis, representing the outputs.

        The method returns the losses for each sample as `sample_losses`.
        """

        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        """
        Performs the backward pass and calculates the gradients.

        Parameters:
        - dvalues: The gradient of the loss with respect to the outputs of the layer.
        - y_true: The true values.

        This method calculates the gradients of the MSE loss with respect to the inputs of the layer, given the gradient `dvalues` and the true values `y_true`. It first computes the number of samples and the number of outputs in each sample. Then, it computes the gradient on values using the formula `-2 * (y_true - dvalues) / outputs`. Finally, it normalizes the gradient by dividing it by the number of samples.

        The method stores the computed gradients in `self.dinputs`.
        """

        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples



# Mean Absolute Error loss
class MeanAbsoluteError(Loss):
    """
    Mean Absolute Error (L1 loss) loss function.

    This class represents the Mean Absolute Error (MAE) loss function, also known as L1 loss. It measures the average absolute difference between the predicted values and the true values.

    The forward pass of the MAE loss calculates the loss by computing the mean absolute difference between the predicted values (`y_pred`) and the true values (`y_true`) along the last axis.

    The backward pass of the MAE loss calculates the gradient of the loss with respect to the predicted values (`y_pred`) and the true values (`y_true`). It uses the sign of the difference between the true values and the predicted values to determine the direction of the gradient. The gradient is then divided by the number of outputs in each sample and the number of samples for normalization.

    Example usage:
    loss = MeanAbsoluteError()
    loss_value = loss.forward(y_pred, y_true)
    gradients = loss.backward(dvalues, y_true)
    """

    def forward(self, y_pred, y_true):
        """
        Calculates the forward pass of the Mean Absolute Error (MAE) loss function.

        Parameters:
        - y_pred: The predicted values.
        - y_true: The true values.

        Returns:
        - sample_losses: The calculated loss for each sample.

        This method computes the mean absolute difference between the predicted values (`y_pred`) and the true values (`y_true`) along the last axis. It returns the calculated loss for each sample.

        Example usage:
        loss = MeanAbsoluteError()
        loss_value = loss.forward(y_pred, y_true)
        """

        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        # Return losses
        return sample_losses


    def backward(self, dvalues, y_true):
        """
        Calculates the backward pass of the Mean Absolute Error (MAE) loss function.

        Parameters:
        - dvalues: The gradient of the loss with respect to the predicted values.
        - y_true: The true values.

        Returns:
        - dinputs: The gradient of the loss with respect to the inputs (predicted values).

        This method calculates the gradient of the loss with respect to the predicted values (`dvalues`) and the true values (`y_true`). It uses the sign of the difference between the true values and the predicted values to determine the direction of the gradient. The gradient is then divided by the number of outputs in each sample and the number of samples for normalization. The method returns the gradient of the loss with respect to the inputs (predicted values).

        Example usage:
        loss = MeanAbsoluteError()
        gradients = loss.backward(dvalues, y_true)
        """

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        return self.dinputs


