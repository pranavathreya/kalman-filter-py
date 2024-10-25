from kalman_filter import KalmanFilter
import numpy as np


def main():
    # set-up true values
    tru_vals = np.array([
                        np.linspace(0, 10), # x
                        np.linspace(0, 15), # y
                        np.linspace(0, 20), # z
                        np.concat([np.linspace(0,5, num=10),
                                   np.ones(30) * 5,
                                   np.linspace(5,0, num=10)]), # v-x
                        np.concat([np.linspace(0,5, num=10),
                                   np.ones(30) * 5,
                                   np.linspace(5,0, num=10)]), # v-y
                        np.concat([np.linspace(0,5, num=10),
                                   np.ones(30) * 5,
                                   np.linspace(5,0, num=10)]), # v-z
                        ])
   
    # create measurements:
    mvar = 5
    R = mvar * np.identity(3, 3)
    msmts = # what is the shape of measurements
    
    ## initial prediction: 0
    #prdn = np.zeros(6, 1)
    #est_cov = 1e3 # high estimate uncertainty since first guess is very wrong
    #d_t = 0.1 # delta-time is 0.1 seconds

    #kf = KalmanFilter(q=0.1, mvar=5)

    ## compute kalman gain:
    #K = kf.K(est_cov, d_t)

    ## update estimate with gain and measurement:
    #est = update(prdn, msmt, d_t)

if __name__ == '__main__':
    main()
