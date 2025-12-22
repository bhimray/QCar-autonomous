# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Imports and Global Variables
import numpy as np
from threading import Thread
import time
import signal

from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.scope import MultiScope
from pal.utilities.math import wrap_to_pi


if not IS_PHYSICAL_QCAR:
    calibrate = False
else:
    calibrate = 'y' in input('do you want to recalibrate?(y/n)')

#================ Experiment Configuration ================
# ===== Timing Parameters
# - tf: experiment duration in seconds.
# - controllerUpdateRate: control update rate in Hz. Shouldn't exceed 500

tf = 10
controllerUpdateRate = 100

# Used to enable safe keyboard triggered shutdown
global KILL_THREAD
KILL_THREAD = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)

#endregion

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

class QcarEKF:

    def __init__(self, x0, P0, Q, R):
        # Nomenclature:
        # - x0: initial estimate
        # - P0: initial covariance matrix estimate
        # - Q: process noise covariance matrix
        # - R: observation noise covariance matrix
        # - xHat: state estimate
        # - P: state covariance matrix
        # - L: wheel base of the QCar
        # - C: output matrix

        self.L = 0.257
        self.I = np.eye(3)
        self.xHat = x0
        self.P = P0
        self.Q = Q
        self.R = R

        self.C = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

    # ==============  SECTION A -  Motion Model ====================
    def f(self, X, u, dt):
        # Kinematic Bicycle Model:
        # - X = [x, y, theta]
        # - u[0] = v (speed in [m/s])
        # - u[1] = delta (steering Angle in [rad])
        # - dt: change in time since last update
        # Ensure X is a 1D array of scalars (handles column vectors)
        Xv = np.asarray(X).ravel()
        theta = float(Xv[2])
        v = float(u[0])
        delta = float(u[1])

        beta = wrap_to_pi(np.arctan(0.5 * np.tan(delta)))
        x_dot = v * np.cos(theta + beta)
        y_dot = v * np.sin(theta + beta)
        theta_dot = (v / self.L) * np.tan(delta)
        theta_next = wrap_to_pi(theta + theta_dot * dt)

        X_next = np.array([[Xv[0] + x_dot * dt], [Xv[1] + y_dot * dt], [theta_next]])
        return X_next

    # ==============  SECTION B -  Motion Model Jacobian ====================
    def Jf(self, X, u, dt):
        # Jacobian for the kinematic bicycle model (see self.f)
        # Convert inputs to scalars to avoid array-shaped elements
        Xv = np.asarray(X).ravel()
        theta = float(Xv[2])
        v = float(u[0])
        delta = float(u[1])

        beta = wrap_to_pi(np.arctan(0.5 * np.tan(delta)))
        phi = theta + beta

        # d(beta)/d(delta)
        dbeta_ddelta = 0.5 / (np.cos(delta)**2 + 0.25 * np.sin(delta)**2)

        # Continuous-time Jacobians (partials)
        A = np.array([
            [0.0, 0.0, -v * np.sin(phi)],
            [0.0, 0.0,  v * np.cos(phi)],
            [0.0, 0.0,  0.0]
        ])

        B = np.array([
            [np.cos(phi), -v * np.sin(phi) * dbeta_ddelta],
            [np.sin(phi),  v * np.cos(phi) * dbeta_ddelta],
            [0.0,          0.0]
        ])

        # Discretize using first-order approximation: F = I + A*dt, G = B*dt
        F = self.I + A * dt
        G = B * dt
        return F, G

    # ==============  SECTION C -  Motion Model Prediction ====================
    def prediction(self, dt, u):
        # 1) state prediction
        self.xHat = self.f(self.xHat, u, dt)

        # 2) covariance prediction
        F, G = self.Jf(self.xHat, u, dt)   # (G optional)
        self.P = F @ self.P @ F.T + self.Q

    # ==============  SECTION D -  Measurement correction ====================

    def correction(self, y):
       # y must be shape (3,1)
        y = np.asarray(y).reshape(-1, 1)

        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)

        self.xHat = self.xHat + K @ (y - self.C @ self.xHat)
        self.P = (self.I - K @ self.C) @ self.P


class GyroKF:

    def __init__(self, x0, P0, Q, R):
        # Nomenclature:
        # - x0: initial estimate
        # - P0: initial covariance matrix estimate
        # - Q: process noise covariance matrix
        # - R: observation noise covariance matrix
        # - xHat: state estimate
        # - P: state covariance matrix
        # - A: state matrix
        # - B: input matrix
        # - C: output matrix

        self.I = np.eye(2)
        self.xHat = x0
        self.P = P0
        self.Q = Q
        self.R = R

        # State Space Representation Matrices
        self.A = np.array([
            [0, -1],
            [0, 0]
        ])
        self.B = np.array([
            [1],
            [0]
        ])
        self.C = np.array([
            [1, 0]
        ])


    # ==========  SECTION F -  Gyro Heading Prediction  ================
    def prediction(self, dt, u):
        # Discrete-time state prediction
        theta = self.xHat[0,0]
        bias  = self.xHat[1,0]

        theta_next = wrap_to_pi(theta + (u - bias) * dt)
        bias_next  = bias

        self.xHat = np.array([[theta_next],
                            [bias_next]])

        # Discrete-time covariance prediction
        F = self.I + self.A * dt
        self.P = F @ self.P @ F.T + self.Q
        pass

    
    # ==========  SECTION G -  GPS Heading Correction  ================
    def correction(self, y):
        # - y: heading measurement from GPS
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.xHat = self.xHat + K @ (y - self.C @ self.xHat)
        self.P = (self.I - K @ self.C) @ self.P
        pass

class StateEstimation:

    def __init__(self, tf=tf, controllerUpdateRate=controllerUpdateRate, calibrate=calibrate):
        self.tf = tf
        self.controllerUpdateRate = controllerUpdateRate
        self.calibrate = calibrate

        # Scopes will be created when `setup_scopes` is called or on run
        self.scope = None
        self.biasScope = None

        self._control_thread = None

    def setup_scopes(self):
        if IS_PHYSICAL_QCAR:
            fps = 10
        else:
            fps = 30

        # Scope for displaying estimated gyroscope bias
        self.biasScope = MultiScope(
            rows=2,
            cols=1,
            title='Heading Kalman Filter',
            fps=fps
        )
        self.biasScope.addAxis(row=0, col=0, xLabel='Time [s]', yLabel='Heading Angle [rad]', timeWindow=self.tf)
        self.biasScope.axes[0].attachSignal()
        self.biasScope.addAxis(row=1, col=0, xLabel='Time [s]', yLabel='Gyroscope Bias [rad/s]', timeWindow=self.tf)
        self.biasScope.axes[1].attachSignal()

        # Scope for comparing performance of various estimator types
        self.scope = MultiScope(rows=3, cols=2, title='QCar State Estimation', fps=fps)

        self.scope.addAxis(row=0, col=0, timeWindow=self.tf, yLabel='x Position [m]')
        self.scope.axes[0].attachSignal(name='x_dr')
        self.scope.axes[0].attachSignal(name='x_ekf_gps')
        self.scope.axes[0].attachSignal(name='x_ekf_sf')

        self.scope.addAxis(row=1, col=0, timeWindow=self.tf, yLabel='y Position [m]')
        self.scope.axes[1].attachSignal(name='y_dr')
        self.scope.axes[1].attachSignal(name='y_ekf_gps')
        self.scope.axes[1].attachSignal(name='y_ekf_sf')

        self.scope.addAxis(row=2, col=0, timeWindow=self.tf, yLabel='Heading Angle [rad]')
        self.scope.axes[2].xLabel = 'Time [s]'
        self.scope.axes[2].attachSignal(name='th_dr')
        self.scope.axes[2].attachSignal(name='th_ekf_gps')
        self.scope.axes[2].attachSignal(name='th_ekf_sf')

        self.scope.addXYAxis(row=0, col=1, rowSpan=3, xLabel='x Position [m]', yLabel='y Position [m]', xLim=(-1.5, 1.5), yLim=(-0.5, 2.5))
        self.scope.axes[3].attachSignal(name='ekf_dr')
        self.scope.axes[3].attachSignal(name='ekf_gps')
        self.scope.axes[3].attachSignal(name='ekf_sf')

    def start(self):
        # ensure scopes created
        if self.scope is None or self.biasScope is None:
            self.setup_scopes()

        self._control_thread = Thread(target=self.controlLoop)
        self._control_thread.start()

        try:
            while self._control_thread.is_alive() and (not KILL_THREAD):
                MultiScope.refreshAll()
                time.sleep(0.01)
        finally:
            # request stop
            KILL_THREAD = True

    def controlLoop(self):
        global KILL_THREAD
        # used to limit data sampling to 10hz
        countMax = self.controllerUpdateRate / 10
        count = 0

        # Estimators Setup
        x0 = np.zeros((3,1))
        P0 = np.eye(3)

        ekf_dr = QcarEKF(x0=x0, P0=P0, Q=np.diagflat([0.01, 0.01, 0.01]), R=None)
        ekf_gps = QcarEKF(x0=x0, P0=P0, Q=np.diagflat([0.01, 0.01, 0.01]), R=np.diagflat([0.2, 0.2, 0.1]))

        kf = GyroKF(x0=np.zeros((2,1)), P0=np.eye(2), Q=np.diagflat([0.00001, 0.00001]), R=np.diagflat([.1]))

        C_combined = np.eye(3)
        C_headingOnly = np.array([[0, 0, 1]])
        R_combined = np.diagflat([0.8, 0.8, 0.01])
        R_headingOnly = np.diagflat([0.01])

        ekf_sf = QcarEKF(x0=x0, P0=P0, Q=np.diagflat([0.0001, 0.0001, 0.0001]), R=R_combined)

        # Main Control Loop
        qcar = QCar(readMode=1, frequency=self.controllerUpdateRate)
        gps = QCarGPS(initialPose=x0[:,0], calibrate=self.calibrate)

        with qcar, gps:
            t0 = time.time()
            t = 0
            while (t < self.tf) and (not KILL_THREAD):
                tp = t
                t = time.time() - t0
                dt = t - tp

                # QCar I/O
                qcar.read()
                speed_tach = qcar.motorTach
                th_gyro = wrap_to_pi(qcar.gyroscope[2])

                u = 0.1
                delta = np.pi/12
                qcar.write(u, delta)

                # Predictions
                ekf_dr.prediction(dt, [speed_tach, delta])
                ekf_gps.prediction(dt, [speed_tach, delta])
                ekf_sf.prediction(dt, [speed_tach, delta])
                kf.prediction(dt, th_gyro)

                # Corrections
                if gps.readGPS():
                    ekf_sf.C = C_combined
                    ekf_sf.R = R_combined

                    position = gps.position
                    th_gps = wrap_to_pi(float(gps.orientation[2]))
                    kf.correction(th_gps)

                    y = np.array([[position[0]], [position[1]], [th_gps]])
                    ekf_sf.correction(y)
                else:
                    ekf_sf.C = C_headingOnly
                    ekf_sf.R = R_headingOnly
                    th_kf = wrap_to_pi(float(kf.xHat[0,0]))
                    ekf_sf.correction(np.array([[th_kf]]))

                # Update scopes
                count += 1
                if count >= countMax:
                    self.scope.axes[0].sample(t, [ekf_dr.xHat[0,0], ekf_gps.xHat[0,0], ekf_sf.xHat[0,0]])
                    self.scope.axes[1].sample(t, [ekf_dr.xHat[1,0], ekf_gps.xHat[1,0], ekf_sf.xHat[1,0]])
                    self.scope.axes[2].sample(t, [ekf_dr.xHat[2,0], ekf_gps.xHat[2,0], ekf_sf.xHat[2,0]])
                    self.scope.axes[3].sample(t, [[ekf_dr.xHat[0,0], ekf_dr.xHat[1,0]], [ekf_gps.xHat[0,0], ekf_gps.xHat[1,0]], [ekf_sf.xHat[0,0], ekf_sf.xHat[1,0]]])

                    self.biasScope.axes[0].sample(t, [kf.xHat[0]])
                    self.biasScope.axes[1].sample(t, [kf.xHat[1]])

                    count = 0

        return