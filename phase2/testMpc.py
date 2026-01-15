import numpy as np
from control.SQPMPCController import SQPMPCController
from control.KinematicBicycleModel import KinematicBicycleModel

x_test = np.array([0, 0, 0, 0.0, 0.0, 0.0])
z_test = np.array([[1, 0, 0, 1.0]] * 10)

driveController = SQPMPCController(
    model=KinematicBicycleModel(lf=0.1, lr=0.1, Ts=0.1),
    N=10,
    bounds={
        "delta": (-np.pi/6, np.pi/6),
        "v": (0.0, 2.0),
    },
    weights={
        "Q": (10.0, 10.0, 5.0, 1.0),
        "R": (1.0, 0.5),
        "Rrate": (5.0, 1.0),
    },
    sqp_iters=2
)
delta_cmd, v_cmd = driveController.compute_control(x_test, z_test)
print(delta_cmd, v_cmd)
