# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports

"""
environment_interpretation.py

Skills activity code for environment interpretation lab guide.
Please review the accompanying "Lab Guide - Environment Interpretation" PDF
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import logit, expit
from scipy import ndimage
from threading import Thread, Lock
import time
import pyqtgraph as pg
import signal

from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR, QCAR_CONFIG
from pal.utilities.scope import MultiScope
from pal.utilities.math import find_overlap, wrap_to_2pi, wrap_to_pi
from hal.content.qcar_functions import QCarEKF, QCarDriveController
from hal.products.mats import SDCSRoadMap
import pal.resources.images as images
from frenetControl.Path2D import Path2D
from frenetControl.FrenetBicycleModel import FrenetBicycleModel
from frenetControl.FrenetSQPMPCController import FrenetSQPMPCController

#endregion

# ================ Experiment Configuration ================
# ===== Timing Parameters
# - tf: experiment duration in seconds.
# - startDelay: delay to give filters time to settle in seconds.
# - controllerUpdateRate: control update rate in Hz. Shouldn't exceed 500
tf = 100
startDelay = 1
controllerUpdateRate = 100

# ===== Vehicle Controller Parameters
# - enableVehicleControl: If true, the QCar will drive through the specified
#   node sequence. If false, the QCar will remain stationary.
# - v_ref: desired velocity in m/s
# - nodeSequence: list of nodes from roadmap. Used for trajectory generation.
enableVehicleControl = True
enableSteeringControl = True
v_ref = 0.3
# nodeSequence = [0, 20, 0]
nodeSequence = [9, 14, 9]
# ===== Occupancy Grid Parameters
# - cellWidth: edge length for occupancy grid cells (in meters)
# - r_res: range resolution for polar grid cells (in meters)
# - r_max: maximum range of the lidar (in meters)
# - p_low: likelihood value for vacant cells (according to lidar scan)
# - p_high: likelihood value for occupied cells (according to lidar scan)
cellWidth = 0.02
r_res = 0.05
r_max = 4
p_low = 0.4
p_high = 0.6


#region Initial Setup
lock = Lock()

roadmap = SDCSRoadMap()
waypointSequence = roadmap.generate_path(nodeSequence)
initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
x_hat = initialPose
t_hat = 0
print("initial position", x_hat)
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
#endregion

W = 0.257         # lane width [m] (example)
b = 0.1       # buffer [m]
ey_min = -(W/2 - b)
ey_max = +(W/2 - b)
lr = lf = 0.128 # meters

bounds = {
        "delta": (-np.pi/6, np.pi/6),
        "v_cmd": (0, v_ref),
        "ey": (ey_min, ey_max)
    }
weights = {
        # [s, ey, epsi, v]
        "Q":   [ 1e-8,  5,  5,  0.0 ],
        "R":   [ 1e-5,  1e-5 ],#delta, v_cmd
        "Sdu": [ 1e-8,  1e-5 ] #delta, v_cmd
    }
path = Path2D(waypointSequence.T)  # waypointSequence is (2,N), convert to (N,2)

def controlLoop():
    #region controlLoop setup
    global KILL_THREAD, x_hat, t_hat
    u = 0
    delta = 0
    # used to limit data sampling to 5 Hz
    countMax = controllerUpdateRate / 5
    count = 0
    last_print_t = 0.0
    #endregion

    #region QCar interface setup
    with lock:
        ekf = QCarEKF(x_0=x_hat)
    driveController = QCarDriveController(waypointSequence, cyclic=False)

    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    Ts = 1.0 / controllerUpdateRate
    # in controlLoop init
    model_f = FrenetBicycleModel(L=lf+lr, Ts=Ts, tau_v=0.3)
    f_controller = FrenetSQPMPCController(path, model_f, N=20, bounds=bounds, weights=weights, sqp_iters=2)
    #endregion

    with qcar:
        t0 = time.time()
        t = 0
        while (t < tf+startDelay) and (not KILL_THREAD):
            #region : Loop timing update
            tp = t
            t = time.time() - t0
            dt = t-tp
            #endregion

            #region : Update QCar state estimates and drive controller
            qcar.read()
            v = qcar.motorTach
            if gps.readGPS():
                y_gps = np.array([
                    gps.position[0],
                    gps.position[1],
                    gps.orientation[2]
                ])
                ekf.update(
                    [v, delta],
                    dt,
                    y_gps,
                    qcar.gyroscope[2],
                )
            else:
                ekf.update(
                    [v, delta],
                    Ts,
                    None,
                    qcar.gyroscope[2],
                )
            with lock:
                t_hat = time.time()
                x_hat = ekf.x_hat[:]

            x = ekf.x_hat[0, 0]
            y = ekf.x_hat[1, 0]
            th = ekf.x_hat[2, 0]
            p = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.5

            if t < startDelay or (not enableVehicleControl):
                u = 0.0000
                delta = 0.0000
            else:
                # --- Controller ---
                delta_cmd, v_cmd = f_controller.compute_control(
                    x_world=p, psi=th, v_meas=v, delta = delta,
                    verbose=False
                )
                delta = delta_cmd
                u = v_cmd
            qcar.write(u, delta)
            if t - last_print_t >= 0.2:
                print(f"th={th:.2f}, delta_cmd={delta:.2f}, v_cmd/vmeas={u:.2f}/{v:.2f}")
                last_print_t = t
            #endregion
            
            #region : Update Scopes
            count += 1
            if count >= countMax and t > startDelay:
                t_plot = t - startDelay
                # Speed control scope
                speedScope.axes[0].sample(t_plot, [v, v_ref])
                speedScope.axes[1].sample(t_plot, [v_ref-v])
                speedScope.axes[2].sample(t_plot, [u])

                # Steering control scope
                if enableSteeringControl:
                    steeringScope.axes[4].sample(t_plot, [[p[0],p[1]]])

                    p[0] = ekf.x_hat[0,0]
                    p[1] = ekf.x_hat[1,0]

                    # x_ref = steeringController.p_ref[0]
                    # y_ref = steeringController.p_ref[1]
                    # th_ref = steeringController.th_ref

                    x_ref = gps.position[0]
                    y_ref = gps.position[1]
                    th_ref = gps.orientation[2]

                    steeringScope.axes[0].sample(t_plot, [p[0], x_ref])
                    steeringScope.axes[1].sample(t_plot, [p[1], y_ref])
                    steeringScope.axes[2].sample(t_plot, [th, th_ref])
                    steeringScope.axes[3].sample(t_plot, [delta])

                    arrow.setPos(p[0], p[1])
                    arrow.setStyle(angle=180-th*180/np.pi)

                count = 0
            #endregion

            if driveController.steeringController.pathComplete:
                return
            continue
        with lock:
            print('Control thread terminated')



def PlotCurvature():

    scale = 0.5  # visual scaling only

    plt.figure()
    plt.plot(path.wp[:,0], path.wp[:,1], 'k-', label="Path")

    for i in range(0, path.N, max(1, path.N // 40)):
        x, y = path.wp[i]
        psi = path.psi[i]
        kappa = path.kappa[i]

        # normal direction
        nx = -np.sin(psi)
        ny =  np.cos(psi)

        plt.arrow(
            x, y,
            scale * kappa * nx,
            scale * kappa * ny,
            head_width=0.03,
            color='r'
        )

    plt.figure()
    plt.axis("equal")
    plt.title("Path with Curvature Normals")
    plt.legend()
    plt.grid(True)
    plt.show()

    psi_unwrapped = np.unwrap(path.psi)
    dpsi_ds = np.gradient(psi_unwrapped, path.s)

    plt.figure()
    plt.plot(path.s, path.kappa, label="κ(s) from code")
    plt.plot(path.s, dpsi_ds, '--', label="dψ/ds (numerical)")
    plt.legend()
    plt.grid(True)
    plt.title("Curvature consistency check")
    plt.show()

#region : Setup and run experiment
if __name__ == '__main__':
#region : Setup scopes
    if IS_PHYSICAL_QCAR:
        fps = 10
    else:
        fps = 30
        
    # PlotCurvature()
    # Scope for monitoring speed controller
    speedScope = MultiScope(
        rows=3,
        cols=1,
        title='Vehicle Speed Control',
        fps=fps
    )
    speedScope.addAxis(
        row=0,
        col=0,
        timeWindow=tf,
        yLabel='Vehicle Speed [m/s]',
        yLim=(0, 1)
    )
    speedScope.axes[0].attachSignal(name='v_meas', width=2)
    speedScope.axes[0].attachSignal(name='v_ref')

    speedScope.addAxis(
        row=1,
        col=0,
        timeWindow=tf,
        yLabel='Speed Error [m/s]',
        yLim=(-0.5, 0.5)
    )
    speedScope.axes[1].attachSignal()

    speedScope.addAxis(
        row=2,
        col=0,
        timeWindow=tf,
        xLabel='Time [s]',
        yLabel='Throttle Command [%]',
        yLim=(-0.3, 0.3)
    )
    speedScope.axes[2].attachSignal()

    # Scope for monitoring steering controller
    steeringScope = MultiScope(
            rows=4,
            cols=2,
            title='Vehicle Steering Control',
            fps=fps
        )

    steeringScope.addAxis(
            row=0,
            col=0,
            timeWindow=tf,
            yLabel='x Position [m]',
            yLim=(-2.5, 2.5)
        )
    steeringScope.axes[0].attachSignal(name='x_meas')
    steeringScope.axes[0].attachSignal(name='x_ref')

    steeringScope.addAxis(
            row=1,
            col=0,
            timeWindow=tf,
            yLabel='y Position [m]',
            yLim=(-1, 5)
        )
    steeringScope.axes[1].attachSignal(name='y_meas')
    steeringScope.axes[1].attachSignal(name='y_ref')

    steeringScope.addAxis(
            row=2,
            col=0,
            timeWindow=tf,
            yLabel='Heading Angle [rad]',
            yLim=(-3.5, 3.5)
        )
    steeringScope.axes[2].attachSignal(name='th_meas')
    steeringScope.axes[2].attachSignal(name='th_ref')

    steeringScope.addAxis(
            row=3,
            col=0,
            timeWindow=tf,
            yLabel='Steering Angle [rad]',
            yLim=(-0.6, 0.6)
        )
    steeringScope.axes[3].attachSignal()
    steeringScope.axes[3].xLabel = 'Time [s]'

    steeringScope.addXYAxis(
            row=0,
            col=1,
            rowSpan=4,
            xLabel='x Position [m]',
            yLabel='y Position [m]',
            xLim=(-2.5, 2.5),
            yLim=(-1, 5)
        )

    im = cv2.imread(
            images.SDCS_CITYSCAPE,
            cv2.IMREAD_GRAYSCALE
        )

    steeringScope.axes[4].attachImage(
            scale=(-0.002035, 0.002035),
            offset=(1125,2365),
            rotation=180,
            levels=(0, 255)
        )
    steeringScope.axes[4].images[0].setImage(image=im)

    referencePath = pg.PlotDataItem(
            pen={'color': (85,168,104), 'width': 2},
            name='Reference'
        )
    steeringScope.axes[4].plot.addItem(referencePath)
    referencePath.setData(waypointSequence[0, :],waypointSequence[1, :])

    steeringScope.axes[4].attachSignal(name='Estimated', width=2)

    arrow = pg.ArrowItem(
            angle=180,
            tipAngle=60,
            headLen=10,
            tailLen=10,
            tailWidth=5,
            pen={'color': 'w', 'fillColor': [196,78,82], 'width': 1},
            brush=[196,78,82]
        )
    arrow.setPos(initialPose[0], initialPose[1])
    steeringScope.axes[4].plot.addItem(arrow)
    #endregion
    # PlotCurvature()
    #region : Setup threads, then run experiment
    controlThread = Thread(target=controlLoop)
    controlThread.start()

    try:
        while controlThread.is_alive():
            MultiScope.refreshAll()
            time.sleep(0.01)
    finally:
        KILL_THREAD = True
        controlThread.join()
        gps.terminate()
    #endregion

    if not IS_PHYSICAL_QCAR:
        qlabs_setup.terminate()

    input('Experiment complete. Press any key to exit...')
#endregion



