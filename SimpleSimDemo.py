import numpy as np
import os
import pandas as pd
import sys

sys.path.append(os.getcwd())

import src.LogisticRegression as LR

if __name__ == "__main__":
    target_list = [-1] * 5 + [1] * 5
    target = np.array(target_list)
    print("Target Values: \n\t{0}".format(target))
    print()

    df = pd.DataFrame({"A": [3, 1, 4, 5, 10, 15, 20, 42, 12, 34],
                       "B": [-1.1, 1.5, 1.2, 0.5, -5.3, 5.2, -4.3, 4.2, 4, 8.9],
                       "C": np.random.normal(1, 2, 10)})
    print("Data: \n{0}".format(df))
    print()

    beta_init = np.zeros(3)
    theta_init = np.zeros(3)
    beta_vals = LR.fast_gradient(beta_init, theta_init, np.array(df), target, 1, 1, 50)

    print("Final Coefficients: {0}".format(beta_vals[-1]))
    print("Accuracy of Model: {0}".format(
        np.round(LR.evaluate_performance(beta_vals[-1], np.array(df), target)), 4))

    LR.plot_objective_values(beta_vals, np.array(df), target, 1)
