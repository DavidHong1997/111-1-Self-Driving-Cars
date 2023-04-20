import numpy as np

class KalmanFilter(object):
    def __init__(self, x, y, yaw, delta_x, delta_y, delta_yaw):
        # State Input [x, y, yaw]
        self.x = np.array([x, y, yaw, delta_x, delta_y, delta_yaw])
        # Transmition matrix
        self.A = None
        self.B = None
        # Error matrix
        self.P = np.array([[1,0,0,0,0,0],
                           [0,1,0,0,0,0],
                           [0,0,1,0,0,0],
                           [0,0,0,1,0,0],
                           [0,0,0,0,1,0],
                           [0,0,0,0,0,1]])

        # Obeservation matrix
        self.H = np.array([[1,0,0,0,0,0],
                           [0,1,0,0,0,0]])
        # State transition error covariance
        self.Q = None
        # Measurement error
        self.R = None
        
        self._I =  np.eye(6)

    def predict(self,u):
        # Predict New state
        self.x = self.A.dot(self.x) + self.B.dot(u)
        # Predict covariance matrix
        self.P = self.A.dot(self.P).dot(self.A.T) + self.Q
        print(self.P)

    def update(self,z):
        # Kalman gain
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        self.K = self.P.dot(self.H.T).dot(np.linalg.inv(S))

        # Resduial
        y = z[:2] - self.H.dot(self.x)

        # New state
        self.x = self.x + self.K.dot(y)
        
        # New Error matrix 
        KH = self.K.dot(self.H)
        I_KH = self._I - KH
        self.P = (I_KH.dot(self.P.dot(I_KH.T)) + self.K.dot(self.R.dot(self.K.T)))
