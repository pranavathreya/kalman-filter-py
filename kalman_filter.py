import numpy as np

class KalmanFilter:

    def __init__(self, q, mvar):
        # process noise spectral density
        self.q = q

        # measurement noise variance
        self.mvar = mvar

        # constant measurement uncertainty
        self.R = (self.mvar) * np.identity(3)

        # identity matrix for calculations
        self.I = np.identity(6)

        # Observation matrix: makes measurements' dimensions compatible
        H = np.zeros((3, 6))
        H[0,0] = H[1,1] = H[2,2] = 1
        self.H = H

    # state prediction/transition equation
    def predict(self, x_k, u_k, d_t):
        next_prdn = self.F(d_t) @ x_k + self.B(d_t) @ u_k + self.w_k(d_t)

        return next_prdn

    # covariance prediction/transition equation
    def predict_noise(self, p_k, d_t):
        F = self.F(d_t)
        return np.matmul(np.matmul(F, p_k), F) + self.w_k(d_t)

    # state update equation
    def update(self, prdn, z_k, K_k):
        x_0 = np.matmul(self.H, prdn)
        x = z_k - x_0 
        x_k = prdn + np.matmul(K_k, x)

        
        return x_k

    # covariance update equation
    def update_noise(self, est_cov, K_k):
        x = self.I - np.matmul(K_k, self.H)
        x = np.matmul(x, np.matmul(est_cov, x.T))
        
        y = np.matmul(np.matmul(K_k, self.R), K_k.T)

        return x + y

    # map state vector to measurement space, add noise:
    def process_msmt(self, x_k):
        v_k =  np.random.multivariate_normal(mean=np.zeros(3), cov=self.R)
        v_k = np.matrix(v_k).T
        z_k = self.H @ x_k + v_k

        return z_k 

    # calculate process noise covariance matrix
    def Q(self, d_t):
        Q = np.zeros((6, 6))
        Q[0,0] = Q[1,1] = Q[2,2] = (d_t ** 4) / 4
        Q[3,3] = Q[4,4] = Q[5,5] = (d_t ** 2)
        Q[3,0] = Q[4,1] = Q[5,2] = (d_t ** 3) / 2
        Q[0,3] = Q[1,4] =Q[2,5] = Q[3,0]

        r = self.q * Q

        return r

    # calculate sttate transition matrix:
    def F(self, d_t):
        F = np.identity(6)
        F[0,3] = F[1,4] = F[2,5] = d_t

        return F

    # calculate control input matrix:
    def B(self, d_t):
        B = np.zeros((6, 3))
        B[0,0] = B[1,1] = B[2,2] = 0.5 * (d_t ** 2)
        B[3,0] = B[4,1] = B[5,2] = d_t

        return B

    # calculate kalman gain:
    def K(self, est_cov, d_t):
        d = np.matmul(self.H, est_cov)
        d_2 = np.matmul(d, self.H.T)
        d_3 = d_2 + self.R
        d_4 = np.linalg.inv(d_3)
        d_5  = np.matmul( np.matmul(est_cov, self.H.T), d_4)

        return d_5

    # calculate process noise
    def w_k(self, d_t):
        w_k = np.random.multivariate_normal(mean=np.zeros(6), cov=self.Q(d_t))
        w_k = np.matrix(w_k).T
        return w_k

