from kalman_filter import KalmanFilter
import numpy as np
from matplotlib import pyplot as plt

def main():
    # initial conditions 
    est = np.zeros((6, 1))
    est_unc = 5e2 * np.identity(6) # high estimate uncertainty since first guess
                                   # is very wrong
    d_t = 0.1 # delta-time is 0.1 seconds

    msts = make_true_values(d_t)
    kf = KalmanFilter(q=0.1, mvar=5)

    # control input
    u_k = [[1], [1], [1]]

    ests = []
    state_space_vals = []
    msmts = []
    np.set_printoptions(suppress=True)
    for i in range(msts.shape[1]):
        # new predicted state:
        prdn = kf.predict(est, u_k, d_t)
        est_unc_prdn = kf.predict_uncertainty(est_unc, d_t)
        state_space_vals.append(prdn)
        
        # calculate kalman gain and measure data:
        K = kf.K(est_unc_prdn, d_t) 
        m = np.asmatrix(msts[:, i]).T
        m = kf.process_msmt(m)
        msmts.append(m)

        # update state matrix and state covariance:
        est = kf.estimate(prdn, m, K)
        est_unc = kf.estimate_uncertainty(est_unc_prdn, K)
        ests.append(est)


    ests = np.hstack(ests)
    msmts = np.hstack(msmts)
    state_space_vals = np.hstack(state_space_vals)

    # plot x
    plot(state_space_vals[0, :].tolist()[0], 
         msmts[0, :].tolist()[0],
         ests[0, :].tolist()[0], "x (m)", "x positions vs time")

    # plot y
    plot(state_space_vals[1, :].tolist()[0],
         msmts[1, :].tolist()[0],
         ests[1, :].tolist()[0], "y (m)", "y positions vs time")
# plot z
    plot(state_space_vals[2, :].tolist()[0], 
         msmts[2, :].tolist()[0],
         ests[2, :].tolist()[0], "z (m)", "z positions vs time")

    # plot x.
    plot_velocities(state_space_vals[3, :].tolist()[0], 
         ests[3, :].tolist()[0], "x (mps)", "x velocities vs time")

    # plot y.
    plot_velocities(state_space_vals[4, :].tolist()[0],
         ests[4, :].tolist()[0], "y (mps)", "y velocities vs time")

    # plot z.
    plot_velocities(state_space_vals[5, :].tolist()[0], 
         ests[5, :].tolist()[0], "z (mps)", "z velocities vs time")

    # calculate RMSE for position and velocity
    print("RMSE(x):", rmse(state_space_vals[0, :].tolist()[0], ests[0, :].tolist()[0]))
    print("RMSE(y):", rmse(state_space_vals[1, :].tolist()[0], ests[1, :].tolist()[0]))
    print("RMSE(z):", rmse(state_space_vals[2, :].tolist()[0], ests[2, :].tolist()[0]))
    print("RMSE(x.):", rmse(state_space_vals[3, :].tolist()[0], ests[3, :].tolist()[0]))
    print("RMSE(y.):", rmse(state_space_vals[4, :].tolist()[0], ests[4, :].tolist()[0]))
    print("RMSE(z.):", rmse(state_space_vals[5, :].tolist()[0], ests[5, :].tolist()[0]))

    # plot 3-D path
    plot_3d(state_space_vals, msmts, ests)

def rmse(tru, est):
    return np.sqrt(np.mean((np.array(tru) - np.array(est)))**2)

def plot(state_space_vals, msmts, ests, ylabel, ttl):
    plt.figure()
    time_ = np.arange(len(state_space_vals)) * 0.1
    plt.plot(time_, state_space_vals, label="state space model", marker=".")
    plt.plot(time_, msmts, label="measurements", marker=".")
    plt.plot(time_, ests, label="estimates", marker=".")

    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.grid()
    plt.title(ttl)
    plt.savefig(ttl+".png")

def plot_velocities(state_space_vals, ests, ylabel, ttl):
    plt.figure()
    time_ = np.arange(len(state_space_vals)) * 0.1
    plt.scatter(time_, state_space_vals, label="state space model", marker=".")
    plt.scatter(time_, ests, label="estimates", marker=".")

    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.grid()
    plt.title(ttl)
    plt.savefig(ttl+".png")

def plot_3d(state_space_vals, msmts, ests):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # plot true values
    ax.plot(state_space_vals[0, :].tolist()[0], 
               state_space_vals[1, :].tolist()[0], 
               state_space_vals[2, :].tolist()[0], label="state space model")
    
    # plot measurements
    ax.scatter(msmts[0, :].tolist()[0],
               msmts[1, :].tolist()[0],
               msmts[2, :].tolist()[0], label="measurements", marker="o")

    # plot estimates
    ax.plot(ests[0, :].tolist()[0],
               ests[1, :].tolist()[0],
               ests[2, :].tolist()[0], label="estimates")

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    plt.title("3D Trajectory") 
    plt.legend()
    plt.savefig("3D Trajectory")
    plt.show()

def make_true_values(d_t):
    F = np.array([[1, 0, 0, d_t,   0,   0, 0.5*(d_t**2),            0,            0],
                  [0, 1, 0,   0, d_t,   0,            0, 0.5*(d_t**2),            0],
                  [0, 0, 1,   0,   0, d_t,            0,            0, 0.5*(d_t**2)],
                  [0, 0, 0,   1,   0,   0,          d_t,            0,            0],
                  [0, 0, 0,   0,   1,   0,            0,          d_t,            0],
                  [0, 0, 0,   0,   0,   1,            0,            0,          d_t],
                  [0, 0, 0,   0,   0,   0,            1,            0,            0],
                  [0, 0, 0,   0,   0,   0,            0,            1,            0],
                  [0, 0, 0,   0,   0,   0,            0,            0,            1]]
                 )

    m = np.array([[0], [0], [0], [0], [0], [0], [1], [1], [1]])
    tru_vals = []
    for i in range(200):
        m = F @ m
        tru_vals.append(m[:6, :])
    tru_vals = np.round(np.hstack(tru_vals))
    return tru_vals
        
                  
if __name__ == '__main__':
    main()
