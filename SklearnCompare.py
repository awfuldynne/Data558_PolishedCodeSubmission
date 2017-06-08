import numpy as np
import os
from sklearn.linear_model import LogisticRegressionCV
import sys

sys.path.append(os.getcwd())

import src.DataPreparation as DP
import src.LogisticRegression as LR

if __name__ == "__main__":
    auto = DP.get_auto_data()

    print("Running Gradient Descent on Auto Data with Lambda of 1...")
    beta_init = np.zeros(auto.Data.shape[1])
    theta_init = np.zeros(auto.Data.shape[1])
    beta_vals = LR.fast_gradient(beta_init, theta_init, auto.Data, auto.Target, 20, 1, 1000)

    print("Final Coefficients: \n\t{0}\n\t{1}".format(auto.Columns, beta_vals[-1]))
    print()

    print("Finding ideal penalty from Sklearn...")
    log_reg = LogisticRegressionCV()
    log_reg.fit(auto.Data, auto.Target)
    print("Ideal Lambda: \n\t{0}".format(log_reg.C_))
    print()
    print("Sklearn Coefficients: \n\t{0}\n\t{1}".format(auto.Columns, np.round(log_reg.coef_[0], 5)))
    print()

    print("Running Gradient Descent on Auto Data with Ideal Lambda...")
    beta_init = np.zeros(auto.Data.shape[1])
    theta_init = np.zeros(auto.Data.shape[1])
    beta_vals = LR.fast_gradient(beta_init, theta_init, auto.Data, auto.Target, 20, log_reg.C_, 1000)

    print("Final Coefficients: \n\t{0}\n\t{1}".format(auto.Columns, beta_vals[-1]))