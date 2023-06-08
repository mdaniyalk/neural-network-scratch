import numpy as np
import joblib

def one_hot_encoding(data, categories=None):
    """
    One hot encode the data.

    Args:
    - data (ndarray): The non-encoded data.
    - categories (ndarray): The categories or unique values of the given data.

    Returns:
    - one_hot (ndarray): One-hot encoded data.

    This function performs one-hot encoding on the provided data. One-hot encoding is a technique used to represent categorical data as binary vectors. It creates a binary vector of length equal to the number of categories and sets the appropriate index to 1 to indicate the presence of that category.

    The function accepts the non-encoded data as `data` and an optional `categories` array containing the categories or unique values of the data. If `categories` is not provided, it will be inferred from the unique values in the data.

    The function first creates a mapping (`cat_to_label`) between each category and its corresponding label index. It then assigns a label to each category in the data using this mapping. Next, it creates a one-hot encoded representation of the labels using `np.eye()` function, which creates a diagonal matrix with ones on the diagonal corresponding to the labels. Finally, it returns the one-hot encoded data.

    Example usage:
    data = ['apple', 'banana', 'apple', 'orange', 'banana']
    one_hot_encoded = one_hot_encoding(data)
    """

    if categories is None:
        categories = np.unique(data)
    cat_to_label = {cat: i for i, cat in enumerate(categories)}
    labels = np.array([cat_to_label[cat] for cat in data])
    one_hot = np.eye(len(categories))[labels]
    return one_hot



def train_test_split(X, y, test_size=0.25, train_size=None, shuffle=True, random_state=None):
    """
    Split the data into training and testing sets.

    Args:
    - X (ndarray): The input features.
    - y (ndarray): The target variable.
    - test_size (float or int): The proportion of the data to use for testing (default is 0.25).
    - train_size (float or int): The proportion of the data to use for training (default is None).
    - shuffle (bool): Whether or not to shuffle the data before splitting (default is True).
    - random_state (int): The seed used by the random number generator (default is None).

    Returns:
    - X_train (ndarray): The training features.
    - y_train (ndarray): The training target variable.
    - X_test (ndarray): The testing features.
    - y_test (ndarray): The testing target variable.

    This function splits the given data into training and testing sets. It allows specifying the proportion of the data to be used for testing (`test_size`) or training (`train_size`). Alternatively, the sizes can be specified as integers indicating the number of samples.

    The function shuffles the data before splitting if the `shuffle` parameter is set to True (default). The shuffling can be controlled by providing a `random_state` value as the seed for the random number generator.

    The function returns the training features (`X_train`), training target variable (`y_train`), testing features (`X_test`), and testing target variable (`y_test`) as separate arrays.

    Example usage:
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)
    """
    
    if train_size is not None:
        if train_size > 1:
            test_size = int(X.shape[0]-train_size)
        else:
            test_size = int(X.shape[0]*train_size)
            train_size = int(X.shape[0]-test_size)
    else:
        if test_size > 1:
            train_size = int(X.shape[0]-test_size)
        else:
            train_size = int(X.shape[0]*test_size)
            test_size = int(X.shape[0]-train_size)
    
    if random_state:
        np.random.seed(random_state)

    if shuffle:
        keys = np.array(range(X.shape[0]))
        np.random.shuffle(keys)
        X = X[keys]
        y = y[keys]
    
    X_test = X[:test_size]
    y_test = y[:test_size]
    X_train = X[test_size:]
    y_train = y[test_size:]

    return X_train, y_train, X_test, y_test


    
