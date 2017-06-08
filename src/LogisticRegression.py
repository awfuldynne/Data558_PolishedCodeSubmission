import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """ Compute the sigmoid of x
    :param x: Numeric
    """
    return 1 / (1 + np.exp(-x))


def computegrad(beta, x, y, l=0.1):
    """ Computes the gradient of the logistic regression objective function
    
    :param beta: An array of length D (The number of features)
    :param x: The data the model is being trained on
    :param y: An array of target prediction (Either -1 or 1)
    :param l: Regularization penalty term
    """
    yx = y[:, np.newaxis]*x
    exp_term = 1 + np.exp(-yx.dot(beta))
    grad = (1 / y.shape[0]) * np.sum(-yx * np.exp(-yx.dot(beta[:, np.newaxis])) / exp_term[:, np.newaxis], axis=0)
    grad = grad + (2 * l * beta)
    return grad


def objective(beta, x, y, l=0.1):
    """ Computes the objective value for a given set of coefficients for the target predictions y and the data x.
    
    :param beta: An array of length D (The number of features)
    :param x: The data the model is being trained on
    :param y: An array of target prediction (Either -1 or 1)
    :param l: Regularization penalty term
    """
    obj_score = (1 / y.shape[0]) * np.sum(np.log(1 + np.exp(-y*x.dot(beta))))
    obj_score += l * np.sum(np.square(beta))
    return obj_score


def bt_line_search(beta, x, y, l=0.1, n=1, alpha=0.5, beta_param=0.8, max_iter=100):
    """ Performs backtracking line search to find the optimal step size for an iteration of gradient descent.
    
    :param beta: An array of length D (The number of features)
    :param x: The data the model is being trained on
    :param y: An array of target prediction (Either -1 or 1)
    :param l: Regularization penalty term
    :param n: Step size (eta)
    :param alpha: Scales the minimum amount the objective value must be reduced by to find the ideal step size
    :param beta_param: Amount to reduce step-size by per iteration
    :param max_iter: Maximum number of iterations to run
    """
    grad_x = computegrad(beta, x, y, l)
    norm_grad_x = np.linalg.norm(grad_x)
    found_step = False
    i = 0

    while not found_step and i < max_iter:
        # Check to see if our step size moves the objective value far enough
        if objective(beta - n*grad_x, x, y, l) < objective(beta, x, y, l) - alpha*n*norm_grad_x**2:
            found_step = True
        else:
            # Reduce how long of a step to check
            n = n*beta_param
            i += 1
    return n


def fast_gradient(beta_init, theta_init, x, y, step_init, l=0.1, max_iter=1000):
    """ Finds an optimal set of coefficients for predicting one of two classes using gradient descent.
    Returns a list of coefficients, one for each step of the gradient descent 
    
    :param beta_init: Initial coefficient start values
    :param theta_init: Initial theta starting values
    :param x: Model training data
    :param y: Model target response
    :param step_init: Initial step size
    :param l: Regularization penalty term
    :param max_iter: Maximum number of iterations
    """
    beta = beta_init
    theta = theta_init
    grad_theta = computegrad(theta, x, y, l)
    beta_vals = [beta]
    iter = 0

    while iter < max_iter:
        eta = bt_line_search(theta, x, y, l, step_init)
        # Compute new Beta
        beta_new = theta - eta*grad_theta
        # Compute new Theta
        theta = beta_new + iter/(iter+3)*(beta_new-beta)
        # Add Beta value onto list
        beta_vals.append(beta)
        grad_theta = computegrad(theta, x, y, l)
        beta = beta_new
        iter += 1
    return beta_vals


def plot_objective_values(beta_vals, x, y, l):
    """ Plots a line graph of the objective value at each iteration of beta values within beta vals.
    
    :param beta_vals: List of coefficients at each step of gradient descent
    :param x: The data the model is being trained on
    :param y: An array of target prediction (Either -1 or 1)
    :param l: Regularization penalty term
    """
    indexes = list(range(0, len(beta_vals)))
    objective_vals = []
    for beta in beta_vals:
        objective_vals.append(objective(beta, x, y, l))

    plt.plot(indexes, objective_vals)
    plt.title("Logistic Regression Objective Value")
    plt.show()

def evaluate_performance(beta, x, y):
    """ Returns the accuracy of the model given a set of coefficients and data to predict
    
    :param beta: Final set of coefficients after the model has been trained
    :param x: The data the model is being trained on
    :param y: Expected results
    """
    results = x.dot(beta)
    predicted_vals = []
    for result in results:
        predicted_vals.append(1 if result > 0 else -1)
    return np.mean(predicted_vals == y)