import numpy as np

class KalmanFilter:
    # identity matrix for calculations
    I = np.identity(6)

    # Observation matrix: makes measurements' dimensions compatible
    H = np.zeros(3, 6)
    H[0,0] = H[1,1] = H[2,2]

    def __init__(self, q, mvar):
        # process noise spectral density
        self.q = q

        # measurement noise variance
        self.mvar = mvar

        # constant measurement uncertainty
        R = mst_var * np.identity(3)

    def predict(self, x_k, u_k, w_k, d_t):
        x_k_1 = np.matmul(self.F(d_t), x_k) + np.matmul(self.B(d_t), u_k) + w_k

        return x_k_1

    def predict_noise(self, p_k, w_k, d_t):
        F = self.F(d_t)
        return np.matmul(np.matmul(F, p_k), F) + w_k

    def update(self, prev_x_k_1, z_k, K_k):
        x_k = prev_x_k_1 + np.matmul(
                K_k, z_k - np.matmul(self.H, prev_x_k_1))
        
        return x_k

    def update_noise(self, prev_p_k_1, K_k):
        x = self.I - np.matmul(K_k, self.H)
        p_k = np.matmul(prev_p_k_1, x.T)
        
        x = np.matmul(np.matmul(K_k, self.R), K_k.T)
        p_k = p_k + x

        return p_k

    def Q(self, d_t):
        Q = np.zeros(6, 6)
        Q[0,0] = Q[1,1] = Q[2,2] = (d_t ** 4) / 4
        Q[3,3] = Q[4,4] = Q[5,5] = (d_t ** 2)
        Q[3,0] = Q[4,1] = Q[5,2] = (d_t ** 3) / 2
        Q[0,3] = Q[1,4] =Q[2,5] = Q[3,0]

        return q * Q

    # State transition matrix:
    def F(self, d_t):
        F = np.identity(6)
        F[0,3] = F[1,4] = F[2,5] = d_t

        return F

    def B(self, d_t):
        B = np.zeros(6, 3)
        B[0,0] = B[1,1] = B[2,2] = 0.5 * (d_t ** 2)
        B[3,0] = B[4,1] = B[5,2] = d_t

        return B
