import numpy as np



class Accuracy:
    """
    Calculates the accuracy of model predictions compared to ground truth values.

    This class provides methods to calculate the accuracy of model predictions compared to ground truth values. It allows for calculating the accuracy for individual predictions as well as accumulating the accuracy over multiple passes through the data.

    Example usage:
    accuracy_calculator = Accuracy()
    accuracy = accuracy_calculator.calculate(predictions, y)
    print("Accuracy:", accuracy)
    """

    def __init__(self):
        """
        Initializes an Accuracy object with variables for accumulated sum of matching values and sample count.
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
    def calculate(self, predictions, y):
        """
        Calculates the accuracy of model predictions compared to ground truth values.
        
        Args:
        - predictions (ndarray): A numpy array of model predictions.
        - y (ndarray): A numpy array of ground truth values.
        
        Returns:
        - accuracy (float): A float representing the accuracy of the model predictions.

        This method calculates the accuracy of the model predictions by comparing them to the ground truth values. It calculates the mean of the comparison results to obtain the accuracy.

        The method also updates the accumulated sum of matching values and sample count, which can be used to calculate the accumulated accuracy over multiple passes through the data.

        Example usage:
        accuracy = accuracy_calculator.calculate(predictions, y)
        """

        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate the accuracy
        accuracy = np.mean(comparisons)

        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        # Return the accuracy
        return accuracy

    def calculate_accumulated(self):
        """
        Calculates the accumulated accuracy over multiple passes through the data.
        
        Returns:
        - accuracy (float): A float representing the accumulated accuracy of the model predictions.

        This method calculates the accumulated accuracy over multiple passes through the data by dividing the accumulated sum of matching values by the accumulated sample count.

        Example usage:
        accumulated_accuracy = accuracy_calculator.calculate_accumulated()
        """

        # Calculate the accumulated accuracy
        accuracy = self.accumulated_sum / self.accumulated_count

        # Return the accumulated accuracy
        return accuracy

    def new_pass(self):
        """
        Resets the accumulated sum and count for a new pass through the data.

        This method resets the accumulated sum and count variables to start a new pass through the data. It is typically called before starting a new training epoch or evaluation pass.

        Example usage:
        accuracy_calculator.new_pass()
        """

        self.accumulated_sum = 0
        self.accumulated_count = 0




class Categorical_Accuracy(Accuracy):
    """
    Calculates the categorical accuracy metric.

    This class represents a categorical accuracy metric for evaluating classification models. It inherits from the `Accuracy` base class.

    The `Categorical_Accuracy` class supports both binary and multiclass classification. By default, it is set to multiclass classification (`binary=False`), where the true labels `y` are expected to be one-hot encoded. If the `binary` parameter is set to `True`, the class assumes binary classification and does not require one-hot encoding.

    Example usage:
    accuracy = Categorical_Accuracy(binary=False)
    accuracy.update_state(y_true, y_pred)
    result = accuracy.result()
    """

    def __init__(self, *, binary=False):
        """
        Initializes the Categorical_Accuracy instance.

        Parameters:
        - binary (bool): Indicates whether the classification is binary or multiclass. Default is False (multiclass).

        The `binary` parameter specifies whether the classification is binary or multiclass. If set to `True`, the classification is considered binary and the true labels `y` are not required to be one-hot encoded. If set to `False` (default), the classification is assumed to be multiclass and the true labels `y` should be one-hot encoded.
        """
        super().__init__()
        self.binary = binary

    def init(self, y):
        """
        Initializes the metric.

        Parameters:
        - y (ndarray): The true labels.

        This method initializes the metric and performs any necessary initialization steps. It does not require any specific initialization for the `Categorical_Accuracy` metric.
        """
        pass

    def compare(self, predictions, y):
        """
        Compares the predictions to the true labels.

        Parameters:
        - predictions (ndarray): The predicted labels.
        - y (ndarray): The true labels.

        Returns:
        - accuracy (ndarray): The accuracy values.

        This method compares the predictions to the true labels and returns the accuracy values. For multiclass classification (when `binary=False`), the true labels `y` are expected to be one-hot encoded. If the predictions match the true labels, the accuracy is considered correct.

        For binary classification (when `binary=True`), the true labels `y` are not required to be one-hot encoded. The predictions are compared directly to the true labels, and the accuracy is calculated based on the exact match.

        The method returns the accuracy values, indicating the correctness of the predictions.
        """
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y




class Regression_Accuracy(Accuracy):
    """
    Accuracy class for regression tasks.

    This class extends the `Accuracy` class and provides functionality specifically designed for regression tasks. It compares predictions with ground truth values and determines if they are within a specified precision.

    The `Regression_Accuracy` class inherits from the `Accuracy` class and overrides its methods to implement the regression-specific logic.

    Attributes:
    - precision: The precision value used to determine if predictions are accurate.

    Methods:
    - init: Initializes the precision value based on the standard deviation of the ground truth data.
    - compare: Compares predictions with ground truth values and determines if they are within the precision.

    Example usage:
    accuracy = Regression_Accuracy()
    accuracy.init(y_train)
    is_accurate = accuracy.compare(predictions, y_test)
    """

    def __init__(self):
        """
        Initialize the Regression_Accuracy object.
        """
        super().__init__()
        self.precision = None

    def init(self, y, reinit=False):
        """
        Initialize the precision value based on the standard deviation of the ground truth data.

        Args:
        - y (ndarray): The ground truth values.
        - reinit (bool): Whether to reinitialize the precision value if it already exists. Default is False.
        """
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        """
        Compare predictions with ground truth values and determine if they are within the precision.

        Args:
        - predictions (ndarray): The predicted values.
        - y (ndarray): The ground truth values.

        Returns:
        - is_accurate (ndarray): A boolean array indicating if predictions are accurate.

        This method compares the predictions with the ground truth values and determines if they are within the specified precision. It calculates the absolute difference between predictions and ground truth values and checks if the difference is less than the precision value. The method returns a boolean array indicating if the predictions are accurate for each corresponding data point.

        Example usage:
        accuracy = Regression_Accuracy()
        accuracy.init(y_train)
        is_accurate = accuracy.compare(predictions, y_test)
        """
        return np.absolute(predictions - y) < self.precision


