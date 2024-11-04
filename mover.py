from kalman_filter import KalmanFilter
import numpy as np
from matplotlib import pyplot as plt

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
    R = mvar * np.identity(3)
    
    # initial prediction: 0
    prdn = np.zeros((6, 1))
    est_cov = 5e2 * np.identity(6) # high estimate uncertainty since first guess
                                   # is very wrong
    d_t = 0.1 # delta-time is 0.1 seconds

    kf = KalmanFilter(q=0.1, mvar=5)

    u_k = [[0], [0], [0]]
    w_k = [[0], [0], [0], [0], [0], [0]]

    ests = []
    msmts = []
    for i in range(tru_vals.shape[1]):
        m = np.asmatrix(tru_vals[:, i]).T
        m = kf.map_msmt(m)
        msmts.append(m)

        # compute kalman gain:
        K = kf.K(est_cov, d_t) 

        # update estimate with gain and measurement:
        est = kf.update(prdn, m, K)

        # update estimate covariance:
        est_cov = kf.update_noise(est_cov, K)
        ests.append(est)

        # make new prediction:
        prdn = kf.predict(est, u_k, w_k, d_t)
        # update predict covariance:
        est_cov = kf.predict_noise(est_cov, w_k, d_t)

    ests = np.hstack(ests)
    msmts = np.hstack(msmts)

    plot(tru_vals[0, :], msmts[0, :].tolist()[0], ests[0, :].tolist()[0], "x (m)")
    plot(tru_vals[1, :], msmts[1, :].tolist()[0], ests[1, :].tolist()[0], "y (m)")
    plot(tru_vals[2, :], msmts[2, :].tolist()[0], ests[2, :].tolist()[0], "z (m)")
    plot_3d(tru_vals, msmts, ests)

def plot(tru_vals, msmts, ests, ylabel):
    time_ = np.arange(len(tru_vals)) * 0.1
    plt.scatter(time_, tru_vals, label="true values")
    plt.scatter(time_, msmts, label="measurements")
    plt.scatter(time_, ests, label="estimates")

    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.show()

def plot_3d(tru_vals, msmts, ests):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(tru_vals[0, :], tru_vals[1, :], tru_vals[2, :], label="true values")
    ax.scatter(msmts[0, :].tolist()[0], msmts[1, :].tolist()[0], msmts[2, :].tolist()[0], label="measurements")
    ax.scatter(ests[0, :].tolist()[0], ests[1, :].tolist()[0], ests[2, :].tolist()[0], label="estimates")

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
   
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
