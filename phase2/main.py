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

from pal.products.qcar import QCar, QCarRealSense, QCarGPS, IS_PHYSICAL_QCAR, QCAR_CONFIG
from pal.utilities.scope import MultiScope
from pal.utilities.math import find_overlap, wrap_to_2pi, wrap_to_pi
from hal.content.qcar_functions import QCarEKF, QCarDriveController
from hal.products.mats import SDCSRoadMap
from hal.content.qcar_functions import LaneKeeping

#endregion

# ================ Experiment Configuration ================
# ===== Timing Parameters
# - tf: experiment duration in seconds.
# - startDelay: delay to give filters time to settle in seconds.
# - controllerUpdateRate: control update rate in Hz. Shouldn't exceed 500
tf = 100
startDelay = 1
controllerUpdateRate = 100
sampleRate     = 30.0
sampleTime     = 1/sampleRate
# ===== Vehicle Controller Parameters
# - enableVehicleControl: If true, the QCar will drive through the specified
#   node sequence. If false, the QCar will remain stationary.
# - v_ref: desired velocity in m/s
# - nodeSequence: list of nodes from roadmap. Used for trajectory generation.
enableVehicleControl = True
v_ref = 3.0
nodeSequence = [0, 20, 0]

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

# ===== Bird's-Eye View (BEV) Parameters
# - bevShape: width and height of the BEV image
# - bevWorldDims: [min x, max x, min y, max y] of the world represented by the BEV image 
bevShape = [800,800]
bevWorldDims = [0,20,-10,10]

# ===== Lane Keeping Parameters
# - Kdd: gain for calculating look-ahead distance
# - ldMin: minimum look-ahead distance
# - ldMax: maximum look-ahead distance
# - maxSteer: maximum steering angle in either direction
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
    import main_setup as qlabs_setup
    from qvl.qcar import QLabsQCar
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

detected_objects = None
det_lock = Lock()
vision_lock = Lock()

shared_vision = {
    "targets": None,
    "timestamp": 0.0
}


qcarImg = QCar2DepthAligned()
myDetector = yoloObjectDetection()

KILL_THREAD = False

#start region : reference building function
def build_reference(waypoints, v_ref=2.0):
    """
    waypoints: (N,2) array [[x,y],...]
    """
    N = waypoints.shape[0]
    z_ref = np.zeros((N, 4))

    z_ref[:, 0:2] = waypoints
    z_ref[:, 3] = v_ref

    for k in range(N-1):
        dx = waypoints[k+1,0] - waypoints[k,0]
        dy = waypoints[k+1,1] - waypoints[k,1]
        z_ref[k,2] = np.arctan2(dy, dx)

    z_ref[-1,2] = z_ref[-2,2]
    return z_ref

def mappingLoop():
    global KILL_THREAD, x_hat, t_hat

    og = EnvInterpretation(
        cellWidth=cellWidth,
        r_res=r_res,
        r_max=r_max,
        p_low=p_low,
        p_high=p_high
    )

    #region Configure Plots
    scope.axes[0].images[0].rotation = 90
    scope.axes[0].images[0].scale = (og.r_res, -og.phiRes*180/np.pi)
    scope.axes[0].images[0].offset = (0, 0)
    scope.axes[0].images[0].levels = (0, 1)

    scope.axes[1].images[0].scale = (og.r_res, -og.r_res)
    scope.axes[1].images[0].offset = (-og.nPatch/2, -og.nPatch/2)
    scope.axes[1].images[0].levels = (0, 1)

    scope.axes[2].images[0].scale = (og.cellWidth, -og.cellWidth)
    scope.axes[2].images[0].offset = (
        og.x_min/og.cellWidth,
        -og.y_max/og.cellWidth
    )
    scope.axes[2].images[0].levels = (0, 1)
    #endregion

    t0 = time.time()
    while time.time()-t0 < startDelay:
        gps.readLidar()

    while (not KILL_THREAD):
        #region Get latest pose estimate
        with lock:
            t = t_hat
            x = x_hat[0]
            y = x_hat[1]
            th = x_hat[2]

        x += 0.125 * np.cos(th)
        y += 0.125 * np.sin(th)
        #endregion

        #Read from Lidar and Update Occupancy Grid
        gps.readLidar()

        if gps.scanTime < t and QCAR_CONFIG['cartype'] == 1:
            continue

        og.updateMap(x, y, th, gps.angles,gps.distances)

        scope.axes[0].images[0].setImage(image=expit(og.polarPatch))
        scope.axes[1].images[0].setImage(image=expit(og.patch))
        scope.axes[2].images[0].setImage(image=expit(og.map))

    with lock:
        print('Mapping thread terminated')

def objectDetectionLoop():
    global detected_objects
    
    # Initialize the model once; reloading every frame is expensive
    myDetector.yolo = YOLOv8()

    while not KILL_THREAD:
        #Loop Timing Update
        start = time.time()

        task = 'classify'  # 'threshold' or 'detect' or 'classify'
        mode = 'yolo'    # 'hsv' or 'rgb' for 'threshold'
        
        qcarImg.read()
        img = qcarImg.rgb
        depth = qcarImg.depth
        detectedMask,detectedName,detectedBbox = myDetector.obj_detect(img,task=task,mode=mode) # mask and name is for each object detected 
        detectedDist = myDetector.find_distance(depth,detectedMask) # if there is two objects, two distances will be calculated
        with det_lock:
            detected_objects = (detectedName, detectedBbox, detectedDist) 
        
        end = time.time()
        compute_time = end - start
        sleepTime = sampleTime - ( compute_time % sampleTime )

        #show detected objects on image
        img = qcarImg.rgb.copy()
        myDetector.annotate(img, detectedName, detectedBbox, detectedDist)
        cv2.imshow('Object Detection', img)
        # Pause/sleep for sleepTime in milliseconds
        msSleepTime = int(1000*sleepTime)
        if msSleepTime <= 0:
            msSleepTime = 1
        cv2.waitKey(msSleepTime)

def visionLoop():
    global KILL_THREAD, shared_vision

    qcarCam = QCarRealSense(
        mode='RGB',
        frameWidthRGB=640,
        frameHeightRGB=480
    )
    myLaneKeeping = LaneKeeping(
        Kdd=Kdd,
        ldMin=ldMin,
        ldMax=ldMax,
        maxSteer=maxSteer,
        bevShape=bevShape,
        bevWorldDims=bevWorldDims
    )

    while not KILL_THREAD:

        #Request sensor signals from QCar
        qcarCam.read_RGB()
        img = qcarCam.imageBufferRGB
        
        # ==============  SECTION A -  Lane Marking  ====================
        laneMarking = np.zeros((480,640),dtype=np.uint8)
        # if cameraIntrinsics is not None and cameraDistortion is not None: 
        #     undistortedImage = cv2.undistort(image, cameraIntrinsics, cameraDistortion) 
        # else: 
        #     undistortedImage = image.copy()

        # ============= SECTION C2 - Image Filtering =============
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        blur = cv2.GaussianBlur(gray, (5,5), 0) 
        # reduces noise 
        # Use Canny for crisp edges; thresholds are tunable low_thresh, 
        low_thresh, high_thresh = 20, 150 
        filteredImage = cv2.Canny(blur, low_thresh, high_thresh)

        # ============= SECTION C3 - Feature Extraction =============
        lines = cv2.HoughLinesP(filteredImage, rho=1, theta=np.pi/180, threshold=10, minLineLength=40, maxLineGap=10) 
        linesImage = img.copy() 
        if lines is not None: 
            for x1,y1,x2,y2 in lines.reshape(-1,4): 
                cv2.line(linesImage, (x1,y1), (x2,y2), (0,255,0), 2) 
                cv2.line(laneMarking, (x1,y1), (x2,y2), 255, 2)
        imageDisplayed = linesImage
        

        # ==============       END OF SECTION A      ====================

        # Creating Bird's-Eye view of Camera Feed and Lane Markings
        bev = myLaneKeeping.ipm.create_bird_eye_view(imageDisplayed)
        bevLaneMarking = myLaneKeeping.ipm.create_bird_eye_view(laneMarking)

        # Process Lane Markings and Extract Availabe Lane Centers (Pure Pursuit Targets)
        processedLaneMarking = myLaneKeeping.preprocess(bevLaneMarking)
        isolated = myLaneKeeping.isolate_lane_markings(processedLaneMarking)
        # targets = myLaneKeeping.find_target(isolated,v)

        # --- Publish targets ---
        # with vision_lock:
        #     shared_vision["targets"]   = targets
        #     shared_vision["timestamp"] = time.time()

        # --- Visualization ONLY here ---
        # cv2.imshow("camera", imageDisplayed)
        # cv2.imshow("lane marking", laneMarking)
        # cv2.imshow("lane marking BEV",bevLaneMarking)
        # cv2.imshow("processed BEV",processedLaneMarking)
        # Visualization of the detected objects
        img = qcarImg.rgb.copy()
        with det_lock:
            if detected_objects:
                myDetector.annotate(img, *detected_objects)
        cv2.imshow("camera BEV", bev)
        cv2.imshow('Image',img) 
        cv2.waitKey(1)

def controlLoop():
    global KILL_THREAD, x_hat, t_hat
    u = 0 
    delta = 0 
    # used to limit data sampling to 10hz 
    countMax = controllerUpdateRate / 10 
    count = 0
    
    arrow1 = pg.ArrowItem( angle=180, 
                          tipAngle=60, 
                          headLen=10, 
                          tailLen=10, 
                          tailWidth=5, 
                          pen={'color': 'w', 'width': 1}, 
                          brush='r' 
                        )
    arrow1.setPos(0,0) 
    scope.axes[1].plot.addItem(arrow1)
    
    arrow2 = pg.ArrowItem( angle=180, 
                          tipAngle=60, 
                          headLen=10, 
                          tailLen=10, 
                          tailWidth=5, 
                          pen={'color': 'w', 'width': 1}, 
                          brush='r' 
                          )
    scope.axes[2].plot.addItem(arrow2)
    
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    ekf  = QCarEKF(x_0=x_hat)
    model=KinematicBicycleModel(
            lf=0.1,
            lr=0.1,
            Ts=controllerUpdateRate)
    bounds= {
            "delta": (-np.pi/6, np.pi/6),
            "a": (-1.0, 1.0),
            "delta_rate": (-np.pi/3, np.pi/3),
            "a_rate": (-10.0, 10.0),
            "v": (0.0, 2.0)
        }
    weights= {
            "Q": [10.0, 10.0, 20.0, 50.0],
            "R": [1.0, 1.0],
            "Rrate": [7.0, 5.0]
        }
    if waypointSequence is not None:
        progressTracker = PathProgressTracker.PathProgressTracker(
            waypointSequence
        )
    else:
        progressTracker = None
    driveController = SQPMPCController(
        model,
        N=10,
        bounds=bounds,
        weights=weights,
        sqp_iters=2
    )
    print("waypointSequence", waypointSequence[0,:].shape)
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
            v = qcar.motorTach

            # --- EKF ---
            if gps.readGPS():
                y_gps = np.array([
                    gps.position[0],
                    gps.position[1],
                    gps.orientation[2]
                ])
            else:
                y_gps = None

            ekf.update(
                [qcar.motorTach, delta],
                dt,
                y_gps,
                qcar.gyroscope[2]
            )

            x = ekf.x_hat[0,0]
            y = ekf.x_hat[1,0]
            th = ekf.x_hat[2,0]

            p = np.array([x, y]) + 0.2*np.array([np.cos(th), np.sin(th)])

            # --- Get vision targets (non-blocking) ---
            # with vision_lock:
            #     targets = shared_vision["targets"]

            # --- Control ---
            if t < startDelay or not enableVehicleControl:
                u = 0
                delta = 0
            else:
                progressTracker.update_progress(x, y, v, th, dt)
                waypointIndices = progressTracker.get_reference_window(lookahead=5.0)
                z_ref_xy = waypointSequence[:, waypointIndices].T
                # z_ref = np.hstack([z_ref_xy, np.zeros((z_ref_xy.shape[0], 2))])
                z_ref = build_reference(z_ref_xy, v_ref=v_ref)
                # print("z_ref", z_ref, z_ref.shape, len(waypointIndices))
                driveController.N = len(waypointIndices)
                delta, a = driveController.compute_control(
                    np.array([x, y, th, v, delta, u]),
                    z_ref
                )
                u = a
                qcar.write(u, delta)
                print(f"t={t:.2f}s, u={u:.2f} m/s, delta={delta:.2f} deg")
            # with det_lock:
            #     if detected_objects:
            #         names, boxes, dists = detected_objects
            #         if "stop sign" in names:
            #             print("Stop Sign Detected - Stopping QCar", names, dists)
            #             qcar.write(u, delta)
            #         elif "yield sign" in names:
            #             print("Yield Sign Detected - Slowing QCar", names, dists)
            #             qcar.write(u, delta)
            #         else:
            #             qcar.write(u, delta)
            #     else:
            #         qcar.write(u, delta)

            # --- MultiScope ---
            #region : Update Scopes 
            count += 1 
            if count >= countMax and t > startDelay: 
                scope.axes[2].sample(t, [[p[0], p[1]]]) 
                arrow1.setStyle(angle=180-th*180/np.pi) 
                arrow2.setPos(p[0],p[1]) 
                arrow2.setStyle(angle=180-th*180/np.pi) 
                count = 0 
            #endregion

if __name__ == '__main__':

    # --------------------
    # Setup Scopes
    # --------------------
    fps = 10 if IS_PHYSICAL_QCAR else 30

    scope = MultiScope(
        rows=2,
        cols=2,
        title='Autonomous QCar',
        fps=fps
    )

    scope.addXYAxis(
        row=0,
        col=0,
        xLabel='Angle [deg]',
        yLabel='Range [m]'
    )
    scope.axes[0].attachImage()

    scope.addXYAxis(
        row=1,
        col=0,
        xLabel='x Position [m]',
        yLabel='y Position [m]'
    )
    scope.axes[1].attachImage()


    scope.addXYAxis(
        row=0, 
        col=1, 
        rowSpan=2,
        xLabel='x Position [m]',
        yLabel='y Position [m]',
        xLim=(-4, 3),
        yLim=(-2, 6)
    )
    scope.axes[2].attachSignal(name='Measured', width=2, style='--.')
    scope.axes[2].attachImage()

    referencePath = pg.PlotDataItem(
        pen={'color': (85,168,104), 'width': 2},
        name='Reference'
    )
    referencePath.setData(waypointSequence[0, :], waypointSequence[1, :])
    scope.axes[2].plot.addItem(referencePath)

    # --------------------
    # Threads
    # --------------------
    controlThread = Thread(target=controlLoop)
    objectDetectionThread = Thread(target=objectDetectionLoop)
    # mappingThread = Thread(target=mappingLoop)
    # visionThread  = Thread(target=visionLoop)

    controlThread.start()
    objectDetectionThread.start()
    # mappingThread.start()
    # visionThread.start()

    refresh_dt = 1.0 / fps

    try:
        while controlThread.is_alive() and not KILL_THREAD:
            MultiScope.refreshAll()
            time.sleep(refresh_dt)

    finally:
        KILL_THREAD = True
        controlThread.join()
        objectDetectionThread.join()
        # mappingThread.join()
        # visionThread.join()
        cv2.destroyAllWindows()
        gps.terminate()

    if not IS_PHYSICAL_QCAR:
        qlabs_setup.terminate()

    input('Experiment complete. Press any key to exit...')
