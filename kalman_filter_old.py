import matplotlib.pyplot as plt
import logging
import numpy as np
logger = logging.getLogger(__name__)

x = np.array([301.5, 298.23, 298.84, 300.42, 301.94, 299.5, 305.98, 301.25,
             299.73, 299.2, 298.62, 301.84, 299.6, 295.3, 299.3, 301.95, 296.3,
             295.11, 295.12, 289.9, 283.51, 276.42, 264.22, 250.25, 236.66, 217.47, 199.75, 
             179.7, 160, 140.92, 113.53, 93.68, 69.71, 45.93, 20.87])

y = np.array([-401.46, -375.44, -346.15, -320.2, -300.08, -274.12, -253.45,
              -226.4, -200.65, -171.62, -152.11, -125.19, -93.4, -74.79, -49.12,
              -28.73, 2.99, 25.65, 49.86, 72.87, 96.34, 120.4, 144.69,
              168.06, 184.99, 205.11, 221.82, 238.3, 253.02, 267.19, 270.71,
              285.86, 299.48, 292.9, 298.77])

# state transition matrix
dt = 1 # seconds
sig_a = 0.2 # random acceleration standard deviation
mst_var_x = 9 # x position variance (m sqd)
mst_var_y = 9
stm_x = np.array([[1, dt, 0.5*pow(dt,2)], # state transition matrix
                  [0, 1,  dt],
                  [0, 0, 1]])

def noise_transition_matrix(stm_x: np.ndarray):
    stm_y = stm_x
    zm = np.zeros(stm_x.shape)
    stm_x = np.concatenate((stm_x, zm), axis=1)
    stm_y = np.concatenate((zm, stm_y), axis=1)
    stm = np.concatenate((stm_x, stm_y))
    return stm

def kalman_gain(pdt_cov, obs_mx, mst_cov):
    kln_gn = np.matmul(pdt_cov, obs_mx.T)
    mst_cp = np.matmul(obs_mx, pdt_cov)
    mst_cp = np.matmul(mst_cp, obs_mx.T)
    mst_cp = mst_cp + mst_cov
    mst_cp = np.linalg.inv(mst_cp)
    kln_gn = np.matmul(kln_gn, mst_cp)
    return kln_gn

def state_update_equation(pdt, kln_gn, mst, obs_mx):
    err_cp = np.matmul(obs_mx, pdt)
    print(f"state_update_equation:\n{err_cp} =\n{obs_mx} *\n{pdt}")
    print(f"state_update_equation:\n{mst} -\n{err_cp}")
    err_cp = mst - err_cp
    err_cp = np.matmul(kln_gn, err_cp)
    return pdt + err_cp

def covariance_update_equation(kln_gn, obs_mx, pdt_cov, mst_cov):
    p = np.identity(kln_gn.shape[0])
    p = p - np.matmul(kln_gn, obs_mx)
    p_t = p.T
    p = np.matmul(p, pdt_cov)
    p = np.matmul(p, p_t)
    k_t = kln_gn.T
    k = np.matmul(kln_gn, mst_cov)
    k = np.matmul(k, k_t)
    p = p + k
    return p

def state_extrapolation_equation(stm, est):
    p = np.matmul(stm, est)
    print(f"state_extrapolation_equation:\n{p} =\n{stm} *\n{est}\n")
    return p

def covariance_extrapolation_equation(stm, est_cov, pcs_nse):
    p = np.matmul(stm, est_cov)
    p = np.matmul(p, stm.T)
    p = p + pcs_nse
    return p

def plot_position(msts_x, msts_y, estimates, predictions):
    plt.figure(figsize=(4,12))
    plt.plot(estimates[:,0], estimates[:,3], 'bs-', label='estimates')
    plt.plot(predictions[:,0], predictions[:,3], 'ro-', label='predictions')
    plt.plot(msts_x, msts_y, 'g^-', label='measurements')

    plt.title('Vehicle Position')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)', rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout(pad=2)
    plt.show()

def plot_velocity(time_, estimates, predictions):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_, estimates[:,2], 'bs-', label='estimates')
    plt.plot(time_, predictions[:,2], 'ro-', label='predictions')

    plt.title('Vehicle Velocity (x)')
    plt.xlabel('time (s)')
    plt.ylabel('velocity (m/s)', rotation=90)
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time_, estimates[:,-2], 'bs-', label='estimates')
    plt.plot(time_, predictions[:,-2], 'ro-', label='predictions')

    plt.title('Vehicle Velocity (y)')
    plt.xlabel('time (s)')
    plt.ylabel('velocity (m/s)', rotation=90)
    plt.legend()
    plt.grid(True)

    plt.tight_layout(pad=2)
    plt.show()

def main():
    log_file = 'mvkf.log'
    with open(log_file, 'w') as f:
        f.truncate(0)

    logging.basicConfig(filename=log_file, level=logging.INFO)
    stm = noise_transition_matrix(stm_x)
    pcs_nse = np.array([[pow(dt,4)/4, pow(dt,3)/2, pow(dt,2)/2],
                        [pow(dt,3)/2, pow(dt,2),   dt],
                        [pow(dt,2)/2, dt,     1]])
    pcs_nse = noise_transition_matrix(pcs_nse) * pow(0.2,2)
    mst_cov = np.diag([mst_var_x, mst_var_y])

    # initial estimates for state variables:
    est = np.array([[0, 0, 0, 0, 0, 0]]).T # x,y : position, velocity, acceleration
    est_cov = np.diag([500 for i in range(6)])
    pdt = np.matmul(stm, est)
    pdt_cov = np.matmul(np.matmul(stm, est_cov), stm.T) + pcs_nse
    print("stm =\n", stm, "\nest =\n", est, "\nest_cov =\n", est_cov,
          "\npdt =\n", pdt, "\npdt_cov =\n", pdt_cov, "\n")

    # observation matrix
    obs_mx = np.array([[1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0]])
    print(f"obs_mx =\n{obs_mx}\n")
    
    estimates = np.empty([6, x.shape[0]]).T
    predictions = np.empty([6, x.shape[0]]).T
    print(estimates)
    for i in range(x.shape[0]):
        kln_gn = kalman_gain(pdt_cov, obs_mx, mst_cov)
        mst = np.array([[x[i], y[i]]]).T
        est = state_update_equation(pdt, kln_gn, mst, obs_mx)
        est_cov = covariance_update_equation(kln_gn, obs_mx, pdt_cov, mst_cov)
        
        pdt = state_extrapolation_equation(stm, est)
        pdt_cov = covariance_extrapolation_equation(stm, est_cov, pcs_nse)

        log_str = (
        f"\n----------------------------------------------------------------------\n"
            f"mst =\n{mst}\n"
            f"kln_gn =\n{kln_gn}\n"
            f"est =\n{est}\n"
            f"est_cov =\n{est_cov}\n{est_cov.shape}\n"
            f"pdt =\n{pdt}\n"
            f"pdt_cov =\n{pdt_cov}\n"
            )

        logger.info(log_str)

        estimates[i] = est.T
        predictions[i] = pdt.T

    plot_position(x, y, estimates, predictions)
    plot_velocity(np.arange(1,36), estimates, predictions)


if __name__ == '__main__':
    main()

