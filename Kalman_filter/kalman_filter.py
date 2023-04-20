import numpy as np

class KalmanFilter(object):
    def __init__(self, x=0, y=0, yaw=0):
        # State [x, y, yaw]
        self.x = np.array([x, y, yaw])
        # Transition matrix
        self.A = np.identity(3)
        self.B = np.identity(3)
        # Error matrix
        self.P = np.identity(3) * 1
        # Observation matrix
        self.H = np.array([[1, 0, 0],
                           [0, 1, 0]])
        # State transition error covariance
        self.Q = np.array([[0.1,0.1*0.5,0],
                           [0.1*0.5,0.1,0],
                           [0,0,0.1]])
        # Measurement error
        self.R = np.array([[0.75,0],
                           [0,0.75]])

        self.I = np.eye(3)

    def predict(self, u):
        
        # predict next state
        self.x = self.A.dot(self.x) + self.B.dot(u)
        
        # P Error matrix predict
        self.P = self.A.dot(self.P).dot(self.A.T) + self.Q
    
    def update(self, z):
        
        # Caluclate Kalman Gain
        S = self.H.dot(self.P).dot(self.H.T) +self.R
        self.K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        
        # Caluclate margin
        y = z - self.H.dot(self.x)

        # New state use Kalman gain
        self.x = self.x + self.K.dot(y)

        # New P Error matrix 
        KH = self.K.dot(self.H)
        I_KH = self.I - KH
        self.P = (I_KH.dot(self.P.dot(I_KH.T)) + self.K.dot(self.R.dot(self.K.T)))
        return self.x, self.P
