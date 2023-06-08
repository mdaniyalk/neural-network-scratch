import numpy as np


# SGD optimizer
class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.

    This class represents the SGD optimizer, which is commonly used for updating the parameters (weights and biases) of a neural network during training. The optimizer performs parameter updates based on the calculated gradients and specified learning rate.

    The SGD optimizer supports learning rate decay and momentum. Learning rate decay gradually reduces the learning rate over time to allow the optimizer to converge to a better solution. Momentum helps accelerate the optimization process by considering the previous parameter updates.

    Example usage:
    optimizer = SGD(learning_rate=0.1, decay=0.001, momentum=0.9)
    optimizer.pre_update_params()
    optimizer.update_params(layer)
    optimizer.post_update_params()
    """

    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        """
        Initialize the SGD optimizer.

        Parameters:
        - learning_rate (float): The learning rate for parameter updates. Default is 1.0.
        - decay (float): The learning rate decay. Default is 0.0.
        - momentum (float): The momentum for parameter updates. Default is 0.0.

        This method initializes the SGD optimizer with the specified learning rate, learning rate decay, and momentum. The current learning rate is set to the initial learning rate. The iterations count is set to 0.

        Example usage:
        optimizer = SGD(learning_rate=0.1, decay=0.001, momentum=0.9)
        """

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        """
        Prepare for parameter updates.

        This method is called once before any parameter updates. If learning rate decay is enabled, it calculates the current learning rate based on the decay and iteration count.

        Example usage:
        optimizer.pre_update_params()
        """

        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """
        Update parameters for a layer.

        Parameters:
        - layer (Layer): The layer to update its parameters.

        This method updates the parameters (weights and biases) for the specified layer based on the specified layer's gradients and the optimizer's settings.

        If momentum is enabled, it checks if the layer contains momentum arrays. If not, it creates momentum arrays filled with zeros. It then calculates the weight and bias updates using the momentum-based formula. The momentum arrays are updated with the new weight and bias updates.

        If momentum is not enabled, it calculates the weight and bias updates using the vanilla SGD formula.

        Finally, it updates the layer's weights and biases with the calculated updates.

        Example usage:
        optimizer.update_params(layer)
        """

        if self.momentum:

            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        """
        Complete parameter updates.

        This method is called once after any parameter updates. It increments the iteration count.

        Example usage:
        optimizer.post_update_params()
        """

        self.iterations += 1



# Adagrad optimizer
class Adagrad:
    """
    Adagrad optimizer.

    Args:
    - learning_rate (float): The learning rate for parameter updates.
    - decay (float): The decay rate for learning rate decay.
    - epsilon (float): A small value added for numerical stability.

    This class implements the Adagrad optimizer. Adagrad is an adaptive learning rate optimization algorithm that adapts the learning rate for each parameter based on the historical gradients. It is well-suited for dealing with sparse data or data with large variations in gradients.

    The class initializes the optimizer with the specified `learning_rate`, `decay`, and `epsilon` values. It also tracks the number of iterations.

    The optimizer provides methods for pre-update, update, and post-update operations on the parameters of a layer. The `pre_update_params` method adjusts the current learning rate based on the decay rate. The `update_params` method updates the parameters of a layer using the Adagrad update rule. The `post_update_params` method increments the number of iterations.

    Example usage:
    optimizer = Adagrad(learning_rate=0.01, decay=0.001, epsilon=1e-7)
    optimizer.pre_update_params()
    optimizer.update_params(layer)
    optimizer.post_update_params()
    """

    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        """
        Initialize the optimizer with specified settings.

        Args:
        - learning_rate (float): The learning rate for parameter updates.
        - decay (float): The decay rate for learning rate decay.
        - epsilon (float): A small value added for numerical stability.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        """
        Perform pre-update operations on parameters.

        This method adjusts the current learning rate based on the decay rate if provided.
        Call this method once before any parameter updates.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """
        Update the parameters of a layer.

        Args:
        - layer: The layer to update the parameters.

        This method updates the parameters of the specified layer using the Adagrad update rule.
        It also initializes the cache arrays for the layer if they don't exist.
        The update rule includes the normalization with the square rooted cache.
        """
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        """
        Perform post-update operations.

        This method increments the number of iterations.
        Call this method once after any parameter updates.
        """
        self.iterations += 1



# RMSprop optimizer
class RMSprop:
    """
    RMSprop optimizer.

    Attributes:
    - learning_rate (float): The learning rate.
    - current_learning_rate (float): The current learning rate.
    - decay (float): The learning rate decay.
    - iterations (int): The number of iterations/updates.
    - epsilon (float): A small value added for numerical stability.
    - rho (float): The decay rate for computing the moving average of squared gradients.

    This class represents the RMSprop optimizer, which is an optimization algorithm commonly used in neural networks. RMSprop adapts the learning rate for each parameter based on the average of recent gradients.

    The optimizer is initialized with the learning rate, decay, epsilon, and rho. The learning rate can be decayed over time by specifying a nonzero decay rate. The current_learning_rate attribute is used to track the dynamically adjusted learning rate during parameter updates.

    The optimizer provides methods for pre-updating parameters, updating parameters, and post-updating parameters. These methods are called before any parameter updates, during parameter updates, and after parameter updates, respectively.

    During parameter updates, the optimizer updates the cache arrays for the weights and biases of each layer. The cache arrays store the squared gradients. The optimizer then performs the parameter update using the RMSprop equation, which involves dividing the gradients by the square root of the corresponding cache value.

    Example usage:
    optimizer = RMSprop(learning_rate=0.001, decay=0.001, epsilon=1e-7, rho=0.9)
    optimizer.pre_update_params()
    optimizer.update_params(layer)
    optimizer.post_update_params()
    """

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        """
        Initialize the RMSprop optimizer.

        Parameters:
        - learning_rate (float): The learning rate.
        - decay (float): The learning rate decay.
        - epsilon (float): A small value added for numerical stability.
        - rho (float): The decay rate for computing the moving average of squared gradients.

        This constructor initializes the RMSprop optimizer with the provided learning rate, decay, epsilon, and rho.

        If decay is nonzero, the current_learning_rate attribute is initialized to the learning_rate divided by (1 + decay * iterations), where iterations is the number of iterations/updates.

        The iterations attribute is set to 0, and the epsilon and rho attributes are set to the provided values.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        """
        Prepare for parameter updates.

        This method is called once before any parameter updates. If the decay attribute is nonzero, it updates the current_learning_rate attribute by decaying the learning rate based on the decay rate and the number of iterations.

        Example usage:
        optimizer.pre_update_params()
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """
        Update the parameters of a layer.

        Parameters:
        - layer: The layer to update.

        This method updates the parameters of the given layer using the RMSprop algorithm. It calculates the squared gradients by updating the cache arrays for the weights and biases of the layer. The cache arrays are initialized if they do not exist.

        The method then performs the parameter update using the RMSprop equation. It updates the weights and biases by subtracting the current_learning_rate multiplied by the gradients divided by the square root of the corresponding cache value.

        Example usage:
        optimizer.update_params(layer)
        """
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        """
        Complete parameter updates.

        This method is called once after any parameter updates. It increments the iterations attribute by 1.

        Example usage:
        optimizer.post_update_params()
        """
        self.iterations += 1



# Adam optimizer
class Adam:
    """
    Adam optimizer class for updating model parameters.

    Parameters:
    - learning_rate (float): The learning rate for parameter updates.
    - decay (float): The learning rate decay rate.
    - epsilon (float): A small value added to avoid division by zero.
    - beta_1 (float): The exponential decay rate for the first moment estimates.
    - beta_2 (float): The exponential decay rate for the second moment estimates.

    This class implements the Adam optimizer, which is an adaptive learning rate optimization algorithm. It is widely used for training deep learning models.

    The optimizer is initialized with the specified settings. The `learning_rate` is the initial learning rate for parameter updates. The `decay` is the learning rate decay rate, which reduces the learning rate over time. The `epsilon` is a small value added to the denominator to avoid division by zero. The `beta_1` and `beta_2` are the exponential decay rates for the first and second moment estimates, respectively.

    The optimizer provides three methods: `pre_update_params()`, `update_params()`, and `post_update_params()`. The `pre_update_params()` method is called before any parameter updates and adjusts the current learning rate based on the decay rate. The `update_params()` method performs the actual parameter updates using the Adam optimization algorithm. The `post_update_params()` method is called after parameter updates to update the iteration count.

    Example usage:
    optimizer = Adam(learning_rate=0.001, decay=0.01, beta_1=0.9, beta_2=0.999)
    optimizer.pre_update_params()
    optimizer.update_params(layer)
    optimizer.post_update_params()
    """

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        """
        Initializes the Adam optimizer with the specified parameters.

        Args:
        - learning_rate (float): The learning rate for parameter updates.
        - decay (float): The learning rate decay rate.
        - epsilon (float): A small value added to avoid division by zero.
        - beta_1 (float): The exponential decay rate for the first moment estimates.
        - beta_2 (float): The exponential decay rate for the second moment estimates.

        This constructor initializes the Adam optimizer with the specified settings. The `learning_rate` is the initial learning rate for parameter updates. The `decay` is the learning rate decay rate, which reduces the learning rate over time. The `epsilon` is a small value added to the denominator to avoid division by zero. The `beta_1` and `beta_2` are the exponential decay rates for the first and second moment estimates, respectively.
        """

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2


    def pre_update_params(self):
        """
        Adjusts the current learning rate based on the decay rate.

        This method should be called once before any parameter updates. It adjusts the current learning rate by multiplying it with the decay factor `(1. / (1. + self.decay * self.iterations))`.

        Example usage:
        optimizer.pre_update_params()
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """
        Performs parameter updates using the Adam optimization algorithm.

        Parameters:
        - layer: The layer object to update its parameters.

        This method should be called to update the parameters of a layer using the Adam optimization algorithm. It performs the following steps:

        1. Checks if the layer contains cache arrays. If not, it creates them and initializes them with zeros.
        2. Updates the momentum with the current gradients using the beta_1 coefficient.
        3. Calculates the corrected momentum by dividing the momentum by `(1 - self.beta_1 ** (self.iterations + 1))`.
        4. Updates the cache with the squared current gradients using the beta_2 coefficient.
        5. Calculates the corrected cache by dividing the cache by `(1 - self.beta_2 ** (self.iterations + 1))`.
        6. Updates the weights and biases of the layer using the Adam update rule, normalized by the square root of the corrected cache.

        Example usage:
        optimizer.update_params(layer)
        """
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * \
                                 layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
                               layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2

        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) +
                             self.epsilon)
        layer.biases += -self.current_learning_rate * \
                bias_momentums_corrected.reshape(layer.biases.shape) / \
                (np.sqrt(bias_cache_corrected.reshape(layer.biases.shape)) + self.epsilon)

    def post_update_params(self):
        """
        Updates the iteration count after parameter updates.

        This method should be called once after any parameter updates. It increments the iteration count by 1.

        Example usage:
        optimizer.post_update_params()
        """
        self.iterations += 1



