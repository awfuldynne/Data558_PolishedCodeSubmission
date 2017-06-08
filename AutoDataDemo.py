import numpy as np
import os
from sklearn.linear_model import LogisticRegressionCV
import sys

sys.path.append(os.getcwd())

import src.DataPreparation as DP
import src.LogisticRegression as LR

if __name__ == "__main__":
    auto = DP.get_auto_data()

    print("Finding ideal penalty from Sklearn...")
    log_reg = LogisticRegressionCV()
    log_reg.fit(auto.Data, auto.Target)
    print("Ideal Lambda: \n\t{0}".format(log_reg.C_))
    print()

    print("Running Gradient Descent on Auto Data...")
    beta_init = np.zeros(auto.Data.shape[1])
    theta_init = np.zeros(auto.Data.shape[1])
    beta_vals = LR.fast_gradient(beta_init, theta_init, auto.Data, auto.Target, 1, log_reg.C_, 50)

    print("Final Coefficients: \n\t{0}\n\t{1}".format(auto.Columns, beta_vals[-1]))
    print("Accuracy of Model: {0}".format(
        np.round(LR.evaluate_performance(beta_vals[-1], auto.Data, auto.Target)), 4))

    LR.plot_objective_values(beta_vals, auto.Data, auto.Target, log_reg.C_)