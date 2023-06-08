import numpy as np
import joblib

def one_hot_encoding(data, categories = None):
    """
    One hot encode the data.

    Args:
    data (ndarray): The non-encoded data.
    categories (ndarray): The categories or unique value of given data.

    Returns:
    one_hot(ndarray): One-hot encoded data.
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
    X (ndarray): The input features.
    y (ndarray): The target variable.
    test_size (float or int): The proportion of the data to use for testing (default is 0.25).
    train_size (float or int): The proportion of the data to use for training (default is None).
    shuffle (bool): Whether or not to shuffle the data before splitting. (default is True).
    random_state (int): The seed used by the random number generator (default is None).

    Returns:
    X_train (ndarray): The training features.
    y_train (ndarray): The training target variable.
    X_test (ndarray): The testing features.
    y_test (ndarray): The testing target variable.
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


    
