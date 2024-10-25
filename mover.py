from kalman_filter import KalmanFilter
import numpy as np

def main():
    
    # initial prediction: 0
    prdn = np.zeros(6, 1)
    est_cov = 1e3 # high estimate uncertainty since first guess is very wrong
    d_t = 0.1 # delta-time is 1 second; msmts taken every 1 sec

    # Should delta-time be changing?

    kf = KalmanFilter(q=0.1, mvar=5)

    # compute kalman gain:
    K = kf.K(est_cov, d_t)

    # update estimate with gain and measurement:
    est = update(prdn, msmt, d_t)
