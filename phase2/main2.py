# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports

import cv2
import numpy as np
from scipy.special import logit, expit
from scipy import ndimage
from threading import Thread, Lock
import time
import pyqtgraph as pg
import signal
from ultralytics import YOLO
from helperFunction.yoloObjectDetection import yoloObjectDetection
from env_interpretation import EnvInterpretation
from control.SQPMPCController import SQPMPCController
from control.KinematicBicycleModel import KinematicBicycleModel
import PathProgressTracker
from pit.YOLO.nets import YOLOv8
from pit.YOLO.utils import QCar2DepthAligned
from hal.content.qcar_functions import ObjectDetection
from hal.utilities.control import StanleyController
from pal.products.qcar import QCar, QCarRealSense, QCarGPS, IS_PHYSICAL_QCAR, QCAR_CONFIG
from pal.utilities.scope import MultiScope
from pal.utilities.math import find_overlap, wrap_to_2pi, wrap_to_pi
from hal.content.qcar_functions import QCarEKF, QCarDriveController
from hal.products.mats import SDCSRoadMap
from hal.content.qcar_functions import LaneKeeping
from frenetControl.FrenetBicycleModel import FrenetBicycleModel
from frenetControl.Path2D import Path2D
from frenetControl.FrenetSQPMPCController import FrenetSQPMPCController
import matplotlib.pyplot as plt
import numpy as np

#endregion

# ================ Experiment Configuration ================
# ===== Timing Parameters
tf = 100
startDelay = 1
controllerUpdateRate = 100
sampleRate     = 30.0
sampleTime     = 1/sampleRate
# ===== Vehicle Controller Parameters
enableVehicleControl = True
v_ref = 0.3  # m/s
nodeSequence = [0, 20, 0]
# nodeSequence = [9, 14, 9]

# ===== Occupancy Grid Parameters
cellWidth = 0.02
r_res = 0.05
r_max = 4
p_low = 0.4
p_high = 0.6

# ===== Bird's-Eye View (BEV) Parameters
bevShape = [800,800]
bevWorldDims = [0,20,-10,10]

# ===== Lane Keeping Parameters
Kdd = 0
ldMin = 10 
ldMax = 20
maxSteer = 0

#region Initial Setup
lock = Lock()

roadmap = SDCSRoadMap()
waypointSequence = roadmap.generate_path(nodeSequence)
initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
x_hat = initialPose
t_hat = 0

if not IS_PHYSICAL_QCAR:
    import qlabs_setup
    hQCar = qlabs_setup.setup(
        initialPosition=[initialPose[0], initialPose[1], 0],
        initialOrientation=[0, 0, initialPose[2]]
    )
    calibrate=False
else:
    calibrate =  'y' in input('do you want to recalibrate?(y/n)')
# Used to enable safe keyboard triggered shutdown
KILL_THREAD = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)

gps = QCarGPS(initialPose=initialPose,calibrate=calibrate)
while (not KILL_THREAD) and (gps.readGPS() or  gps.readLidar()):
    pass

detected_objects = None
det_lock = Lock()
vision_lock = Lock()

shared_vision = {
    "targets": None,
    "timestamp": 0.0
}


qcarImg = QCar2DepthAligned()
myDetector = yoloObjectDetection()
lr = lf = 0.128 # meters

KILL_THREAD = False

W = 0.257         # lane width [m] (example)
b = 0.08         # buffer [m]
ey_min = -(W/2 - b)
ey_max = +(W/2 - b)

bounds = {
        "delta": (-np.pi/6, np.pi/6),
        "v_cmd": (0, v_ref),
        "ey": (ey_min, ey_max)
    }
weights = {
        # [s, ey, epsi, v]
        "Q":   [ 1e-4,  1,  1,  0.0 ],
        "R":   [ 1e-5,  1e-3 ], #delta, v_cmd
        "Sdu": [ 1e-5,  1e-3 ] #delta, v_cmd
    }
path = Path2D(waypointSequence.T)  # waypointSequence is (2,N), convert to (N,2)



def controlLoop():
    global KILL_THREAD, x_hat, t_hat
    u = 0 
    delta = 0 
    # used to limit data sampling to 10hz 
    countMax = controllerUpdateRate / 10 
    count = 0
    # arrow2 = pg.ArrowItem(
    #         angle=180,
    #         tipAngle=60,
    #         headLen=10,
    #         tailLen=10,
    #         tailWidth=5,
    #         pen={'color': 'w', 'width': 1},
    #         brush='r'
    #     )
    # scope.axes[0].plot.addItem(arrow2)

    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    ekf  = QCarEKF(x_0=x_hat)
    Ts = 1.0 / controllerUpdateRate
    # in controlLoop init
    model_f = FrenetBicycleModel(L=lf+lr, Ts=Ts, tau_v=0.3)
    f_controller = FrenetSQPMPCController(path, model_f, N=20, bounds=bounds, weights=weights, sqp_iters=2)

    with qcar:
        t0 = time.time()
        t  = 0.0
        tp = 0.0

        while not KILL_THREAD and t < tf + startDelay:
            # --- Timing ---
            tp = t
            t  = time.time() - t0
            dt = t - tp

            # --- Sensors ---
            qcar.read()

            # --- EKF ---
            if gps.readGPS():
                y_gps = np.array([
                    gps.position[0],
                    gps.position[1],
                    gps.orientation[2]
                ])
                ekf.update(
                    [qcar.motorTach, delta],
                    dt,
                    y_gps,
                    qcar.gyroscope[2],
                )
            else:
                ekf.update(
                    [qcar.motorTach, delta],
                    dt,
                    None,
                    qcar.gyroscope[2],
                )
            with lock:
                t_hat = time.time()
                x_hat = ekf.x_hat[:]

            x = ekf.x_hat[0,0] # you need estimation cuz gps update rate is slower
            y = ekf.x_hat[1,0]
            v = qcar.motorTach # m/s
            th = ekf.x_hat[2,0]

            p = np.array([x, y]) + 0.2*np.array([np.cos(th), np.sin(th)])
            
            # --- Controller ---
            delta_cmd, v_cmd = f_controller.compute_control(
                x_world=p, psi=th, v_meas=v, v_max=v_ref, delta = delta,
                verbose=False
            )
            
            delta = delta_cmd
            print(f"t={t:.2f}s, x={x:.2f}, y={y:.2f}, th={th:.2f}, delta_cmd={delta_cmd:.2f}, v_cmd/vmeas/v={v_cmd:.2f}/{v:.2f}")
            qcar.write(v_cmd, delta_cmd)
            # --- Get vision targets (non-blocking) ---
            # with vision_lock:
            #     targets = shared_vision["targets"]

            # #region : Update Scopes
            # count += 1
            # if count >= countMax and t > startDelay:
            #     scope.axes[0].sample(t, [[p[0], p[1]]])
            #     arrow2.setPos(p[0],p[1])
            #     arrow2.setStyle(angle=180-th*180/np.pi)
            #     count = 0
                
            # #endregion

if __name__ == '__main__':

    fps = 30.0

    # scope = MultiScope(
    #     rows=1,
    #     cols=1,
    #     title='Phase 2: Autonomous Lane Keeping and Navigation',
    #     fps=fps
    # )
    
    # # Generated Map and followed trajectory
    # scope.addXYAxis(
    #     row=0,
    #     col=0,
    #     xLabel='x Position [m]',
    #     yLabel='y Position [m]',
    #     xLim=(-4, 3),
    #     yLim=(-2, 6)
    # )
    # scope.axes[0].attachSignal(name='Measured', width=2, style='--.')
    # scope.axes[0].attachImage()

    # referencePath = pg.PlotDataItem(
    #     pen={'color': (85,168,104), 'width': 2},
    #     name='Reference'
    # )
    # referencePath.setData(waypointSequence[0, :], waypointSequence[1, :])
    # scope.axes[0].plot.addItem(referencePath)
    # --------------------
    # Threads
    # --------------------
    controlThread = Thread(target=controlLoop)

    refresh_dt = 1.0 / fps

    try:
        controlThread.start()
        while controlThread.is_alive() and not KILL_THREAD:
            time.sleep(refresh_dt)
    finally:
        KILL_THREAD = True
        controlThread.join()
        cv2.destroyAllWindows()
        gps.terminate()

    if not IS_PHYSICAL_QCAR:
        qlabs_setup.terminate()

    input('Experiment complete. Press any key to exit...')



# def PlotCurvature():

#     scale = 0.5  # visual scaling only

#     plt.figure()
#     plt.plot(path.wp[:,0], path.wp[:,1], 'k-', label="Path")

#     for i in range(0, path.N, max(1, path.N // 40)):
#         x, y = path.wp[i]
#         psi = path.psi[i]
#         kappa = path.kappa[i]

#         # normal direction
#         nx = -np.sin(psi)
#         ny =  np.cos(psi)

#         plt.arrow(
#             x, y,
#             scale * kappa * nx,
#             scale * kappa * ny,
#             head_width=0.03,
#             color='r'
#         )

#     plt.figure()
#     plt.axis("equal")
#     plt.title("Path with Curvature Normals")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

    # psi_unwrapped = np.unwrap(path.psi)
    # dpsi_ds = np.gradient(psi_unwrapped, path.s)

    # plt.figure()
    # plt.plot(path.s, path.kappa, label="κ(s) from code")
    # plt.plot(path.s, dpsi_ds, '--', label="dψ/ds (numerical)")
    # plt.legend()
    # plt.grid(True)
    # plt.title("Curvature consistency check")
    # plt.show()

