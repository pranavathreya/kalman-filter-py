import numpy as np
from matplotlib import pyplot as plt

def main():
    # delta-time is 0.1 seconds
    d_t = 0.1 
    t_end = 20
    time_steps = int(t_end / d_t)
    # state space model
    F = np.array([[1, 0, 0, d_t,   0,   0,], 
                  [0, 1, 0,   0, d_t,   0,], 
                  [0, 0, 1,   0,   0, d_t,], 
                  [0, 0, 0,   1,   0,   0,], 
                  [0, 0, 0,   0,   1,   0,], 
                  [0, 0, 0,   0,   0,   1,]]
                 )
                
    # control input 
    G = np.array([
         [0.5*(d_t**2),            0,            0],
         [           0, 0.5*(d_t**2),            0],
         [           0,            0, 0.5*(d_t**2)],
         [         d_t,            0,            0],
         [           0,          d_t,            0],
         [           0,            0,          d_t]]
                 )
    u = np.array([[0.5], [-0.3], [0.2]])

    # initial condition
    x = np.array([[0], [0], [0], [0], [0], [0]])

    # process noise
    q = 0.1 
    Q = np.array([
        [(d_t**4)/4, 0,           0,           (d_t**3)/2, 0,           0],
        [0,           (d_t**4)/4, 0,           0,           (d_t**3)/2, 0],
        [0,           0,           (d_t**4)/4, 0,           0,           (d_t**3)/2],
        [(d_t**3)/2, 0,           0,           d_t**2,      0,           0],
        [0,           (d_t**3)/2, 0,           0,           d_t**2,      0],
        [0,           0,           (d_t**3)/2, 0,           0,           d_t**2],
    ])
    Q = q * Q

    # generate true states
    tru_vals = []
    np.random.seed(42)
    for i in range(time_steps):
        w = np.random.multivariate_normal(mean=np.zeros(6), cov=Q)
        w = w[:, np.newaxis]
        x = F @ x + G @ u + w
        tru_vals.append(x)
    tru_vals = np.hstack(tru_vals)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(tru_vals[0, :], 
            tru_vals[1, :], 
            tru_vals[2, :], label="true state")

    # measurement uncertainty
    mstd = 5
    R = (mstd**2) * np.eye(3)

    # estimate uncertainty
    P = np.eye(6)
    #P = 5e2 * P # set high initially

    # observation matrix
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0]])
    
    
    I = np.eye(6)
    ests = []
    state_space_vals = []
    msmts = []
    np.random.seed(42)
    #np.set_printoptions(suppress=True)
    e = np.array([[0], [0], [0], [0], [0], [0]])
    for i in range(time_steps):
        # new predicted state:
        prdn = F @ e + G @ u
        P = F @ P @ F.T + Q

        # calculate kalman gain and measure data:
        k = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)

        # measurement:
        v = np.random.multivariate_normal(mean=np.zeros(3), cov=R)
        v = v[:, np.newaxis]
        x = tru_vals[:, i][:, np.newaxis]
        z = H @ x + v
        msmts.append(z)        

        # innovation:
        e = prdn + k @ (z - H @ prdn)
        s = I - k @ H
        P =  s @ P @ s.T + k @ R @ k.T
        ests.append(e)
    
    ests = np.hstack(ests)
    msmts = np.hstack(msmts)
    time_ = np.arange(time_steps) * 0.1

    ## plot x
    #plot(state_space_vals[0, :], 
    #     msmts[0, :],
    #     ests[0, :], time_, "x (m)", "x positions vs time")

    ## plot y
    #plot(state_space_vals[1, :],
    #     msmts[1, :],
    #     ests[1, :], time_, "y (m)", "y positions vs time")
    ## plot z
    #plot(state_space_vals[2, :], 
    #     msmts[2, :],
    #     ests[2, :], time_, "z (m)", "z positions vs time")

    ## plot x.
    #plot_velocities(state_space_vals[3, :], 
    #     ests[3, :], time_, "x (mps)", "x velocities vs time")

    ## plot y.
    #plot_velocities(state_space_vals[4, :],
    #     ests[4, :], time_,  "y (mps)", "y velocities vs time")

    ## plot z.
    #plot_velocities(state_space_vals[5, :], 
    #     ests[5, :], time_, "z (mps)", "z velocities vs time")

    # calculate RMSE for position and velocity
    print("RMSE(x):", rmse(tru_vals[0, :].tolist(), ests[0, :].tolist()))
    print("RMSE(y):", rmse(tru_vals[1, :].tolist(), ests[1, :].tolist()))
    print("RMSE(z):", rmse(tru_vals[2, :].tolist(), ests[2, :].tolist()))
    print("RMSE(x.):", rmse(tru_vals[3, :].tolist(), ests[3, :].tolist()))
    print("RMSE(y.):", rmse(tru_vals[4, :].tolist(), ests[4, :].tolist()))
    print("RMSE(z.):", rmse(tru_vals[5, :].tolist(), ests[5, :].tolist()))

    # plot 3-D path
    plot_3d(tru_vals, msmts, ests)

def rmse(tru, est):
    return np.sqrt(np.mean((np.array(tru) - np.array(est)))**2)

def plot(state_space_vals, msmts, ests, time_,  ylabel, ttl):
    plt.figure()
    plt.plot(time_, state_space_vals, label="state space model", marker=".")
    plt.plot(time_, msmts, label="measurements", marker=".")
    plt.plot(time_, ests, label="estimates", marker=".")

    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.grid()
    plt.title(ttl)
    plt.savefig(ttl+".png") 

def plot_velocities(state_space_vals, ests, time_, ylabel, ttl):
    plt.figure()
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
    ax.plot(state_space_vals[0, :], 
            state_space_vals[1, :], 
            state_space_vals[2, :], label="true state")
    
    # plot measurements
    ax.scatter(msmts[0, :],
               msmts[1, :],
               msmts[2, :], label="measurements", marker="o")

    # plot estimates
    ax.plot(ests[0, :],
               ests[1, :],
               ests[2, :], label="estimates")

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    plt.title("3D Trajectory") 
    fig.legend()
    fig.savefig("3D Trajectory")
    plt.show()
                  
if __name__ == '__main__':
    main()
