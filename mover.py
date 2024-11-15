from kalman_filter import KalmanFilter
import numpy as np
from matplotlib import pyplot as plt

def main():
    msts = np.array([
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
    
    # initial conditions 
    est = np.zeros((6, 1))
    est_cov = 5e2 * np.identity(6) # high estimate uncertainty since first guess
                                   # is very wrong
    d_t = 0.1 # delta-time is 0.1 seconds

    kf = KalmanFilter(q=0.1, mvar=5)

    # control input
    u_k = [[1], [1], [1]]

    ests = []
    tru_vals = []
    msmts = []
    for i in range(msts.shape[1]):
        # new predicted state:
        prdn = kf.predict(est, u_k, d_t)
        est_cov_nxt = kf.predict_noise(est_cov, d_t)
        tru_vals.append(prdn)
        
        # calculate kalman gain and measure data:
        K = kf.K(est_cov_nxt, d_t) 
        m = np.asmatrix(msts[:, i]).T
        m = kf.process_msmt(m)
        msmts.append(m)

        # update state matrix and state covariance:
        est = kf.update(prdn, m, K)
        est_cov = kf.update_noise(est_cov, K)
        ests.append(est)


    ests = np.hstack(ests)
    msmts = np.hstack(msmts)
    tru_vals = np.hstack(tru_vals)

    # plot x
    plot(tru_vals[0, :].tolist()[0], 
         msmts[0, :].tolist()[0],
         ests[0, :].tolist()[0], "x (m)", "x positions vs time")

    # plot y
    plot(tru_vals[1, :].tolist()[0],
         msmts[1, :].tolist()[0],
         ests[1, :].tolist()[0], "y (m)", "y positions vs time")

    # plot z
    plot(tru_vals[2, :].tolist()[0], 
         msmts[2, :].tolist()[0],
         ests[2, :].tolist()[0], "z (m)", "z positions vs time")

    # plot x.
    plot_velocities(tru_vals[3, :].tolist()[0], 
         ests[3, :].tolist()[0], "x (mps)", "x velocities vs time")

    # plot y.
    plot_velocities(tru_vals[4, :].tolist()[0],
         ests[4, :].tolist()[0], "y (mps)", "y velocities vs time")

    # plot z.
    plot_velocities(tru_vals[5, :].tolist()[0], 
         ests[5, :].tolist()[0], "z (mps)", "z velocities vs time")

    # calculate RMSE for position and velocity
    print("RMSE(x):", rmse(tru_vals[0, :].tolist()[0], ests[0, :].tolist()[0]))
    print("RMSE(y):", rmse(tru_vals[1, :].tolist()[0], ests[1, :].tolist()[0]))
    print("RMSE(z):", rmse(tru_vals[2, :].tolist()[0], ests[2, :].tolist()[0]))
    print("RMSE(x.):", rmse(tru_vals[3, :].tolist()[0], ests[3, :].tolist()[0]))
    print("RMSE(y.):", rmse(tru_vals[4, :].tolist()[0], ests[4, :].tolist()[0]))
    print("RMSE(z.):", rmse(tru_vals[5, :].tolist()[0], ests[5, :].tolist()[0]))

    # plot 3-D path
    plot_3d(tru_vals, msmts, ests)

def rmse(tru, est):
    return np.sqrt(np.mean((np.array(tru) - np.array(est)))**2)

def plot(tru_vals, msmts, ests, ylabel, ttl):
    plt.figure()
    time_ = np.arange(len(tru_vals)) * 0.1
    plt.scatter(time_, tru_vals, label="true values", marker=".")
    plt.scatter(time_, msmts, label="measurements", marker=".")
    plt.scatter(time_, ests, label="estimates", marker=".")

    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.grid()
    plt.title(ttl)
    plt.savefig(ttl+".png")

def plot_velocities(tru_vals, ests, ylabel, ttl):
    plt.figure()
    time_ = np.arange(len(tru_vals)) * 0.1
    plt.scatter(time_, tru_vals, label="true values", marker=".")
    plt.scatter(time_, ests, label="estimates", marker=".")

    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.grid()
    plt.title(ttl)
    plt.savefig(ttl+".png")

def plot_3d(tru_vals, msmts, ests):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # plot true values
    ax.scatter(tru_vals[0, :].tolist()[0], 
               tru_vals[1, :].tolist()[0], 
               tru_vals[2, :].tolist()[0], label="true values", marker=".")
    
    # plot measurements
    ax.scatter(msmts[0, :].tolist()[0],
               msmts[1, :].tolist()[0],
               msmts[2, :].tolist()[0], label="measurements", marker=".")

    # plot estimates
    ax.scatter(ests[0, :].tolist()[0],
               ests[1, :].tolist()[0],
               ests[2, :].tolist()[0], label="estimates", marker=".")

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    plt.title("3D Trajectory") 
    plt.legend()
    plt.savefig("3D Trajectory")


if __name__ == '__main__':
    main()
