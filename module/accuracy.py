import numpy as np



class Accuracy:
    """
    Calculates the accuracy of model predictions compared to ground truth values.
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
        predictions (ndarray): A numpy array of model predictions.
        y (ndarray): A numpy array of ground truth values.
        
        Returns:
        accuracy (float): A float representing the accuracy of the model predictions.
        """
        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate an accuracy
        accuracy = np.mean(comparisons)

        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        # Return accuracy
        return accuracy

    def calculate_accumulated(self):
        """
        Calculates the accumulated accuracy over multiple passes through the data.
        
        Returns:
        accuracy (float): A float representing the accumulated accuracy of the model predictions.
        """
        # Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count

        # Return the data and regularization losses
        return accuracy

    def new_pass(self):
        """
        Resets the accumulated sum and count for a new pass through the data.
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0



class Categorical_Accuracy(Accuracy):
    """
    Calculates the categorical accuracy of model predictions compared to ground truth values.
    Inherits from the Accuracy class.
    """
    def __init__(self, *, binary=False):
        """
        Initializes a Categorical_Accuracy object with a binary mode flag.

        Args:
        binary: A boolean flag indicating whether to use binary mode. Default is False.
        """
        super().__init__()
        self.binary = binary

    def init(self, y):
        """
        No initialization is needed for categorical accuracy calculation.

        Args:
        y: A numpy array of ground truth values.
        """
        pass

    def compare(self, predictions, y):
        """
        Compares model predictions to the ground truth values.

        Args:
        predictions (ndarray): A numpy array of model predictions.
        y (ndarray): A numpy array of ground truth values.

        Returns:
        comparisons (ndarray): A boolean numpy array representing the comparison results.
        """
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y



class Regression_Accuracy(Accuracy):
    """
    Calculates the regression accuracy of model predictions compared to ground truth values.
    Inherits from the Accuracy class.
    """
    def __init__(self):
        """
        Initializes a Regression_Accuracy object with a precision property.
        """
        super().__init__()
        self.precision = None

    def init(self, y, reinit=False):
        """
        Calculates the precision value based on the passed-in ground truth values.

        Args:
        y (ndarray): A numpy array of ground truth values.
        reinit (bool): A boolean flag indicating whether to reinitialize the precision value. Default is False.
        """
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        """
        Compares model predictions to the ground truth values.

        Args:
        predictions (ndarray): A numpy array of model predictions.
        y (ndarray): A numpy array of ground truth values.

        Returns:
        comparisons (ndarray): A boolean numpy array representing the comparison results.
        """
        return np.absolute(predictions - y) < self.precision

