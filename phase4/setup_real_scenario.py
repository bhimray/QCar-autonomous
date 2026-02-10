
import os
# imports to important libraries
import time
import math

from qvl.qcar2 import QLabsQCar2
from qvl.real_time import QLabsRealTime
from qvl.qlabs import QuanserInteractiveLabs
from qvl.free_camera import QLabsFreeCamera
from qvl.crosswalk import QLabsCrosswalk
from qvl.roundabout_sign import QLabsRoundaboutSign
from qvl.system import QLabsSystem
from qvl.yield_sign import QLabsYieldSign
from qvl.stop_sign import QLabsStopSign
from qvl.traffic_cone import QLabsTrafficCone
from qvl.traffic_light import QLabsTrafficLight


# specify if you want the signage on the left or right side of the road.
right_hand_driving = True

def main(right_hand_driving, initialPosition = [-11.136, -8.446, 0.005], initialOrientation = [0, 0, -44.7]):

    # creates a server connection with Quanser Interactive Labs and manages the communications
    qlabs = QuanserInteractiveLabs()

    # trying to connect to QLabs and open the instance we have created - program will end if this fails
    print("Connecting to QLabs...")
    if (not qlabs.open("localhost")):
        print("Unable to connect to QLabs")
        return    

    print("Connected") 

    # destroying any spawned actors in our QLabs that currently exist
    qlabs.destroy_all_spawned_actors()

    # Use hSystem to set the tutorial title on the qlabs display screen
    hSystem = QLabsSystem(qlabs)
    hSystem.set_title_string('Complete Road Signage Tutorial')
    
    spawn_crosswalk(qlabs)
    spawn_signs(qlabs, right_hand_driving)
    spawn_traffic_lights(qlabs, right_hand_driving)
    spawn_cones(qlabs)
    spawn_qcar(qlabs, initialPosition, initialOrientation)

    # Closing qlabs
    qlabs.close()
    print('Done!')


def spawn_qcar(qlabs, initialPosition, initialOrientation):

    x_offset = 0.13
    y_offset = 1.67

    # Spawn a QCar at the given initial pose
    car2 = QLabsQCar2(qlabs)
    car2.spawn_id(actorNumber=0, 
                location=initialPosition, 
                rotation=initialOrientation,
                waitForConfirmation=True)
    
    rtModel = os.path.normpath(os.path.join(os.environ['RTMODELS_DIR'], 'QCar2/QCar2_Workspace_studio'))
    QLabsRealTime().start_real_time_model(rtModel)

    # Create a new camera view and attach it to the QCar
    hcamera = QLabsFreeCamera(qlabs)
    hcamera.spawn([8.484, 1.973, 12.209], [-0, 0.748, 0.792])
    car2.possess()

def spawn_crosswalk(qlabs):
    # Create a crosswalk in this qlabs instance. Since we don't need
    # to access the actors again after creating them, we can use a single
    # class object to spawn all the varieties. We also don't need to use
    # the waitForConfirmation because we don't need to store the actor ID
    # for future reference.

    crosswalk = QLabsCrosswalk(qlabs)

    # spawn crosswalk with degrees in config 0
    crosswalk.spawn_degrees(location=[-12.992, -7.407, 0.005], rotation=[0,0,48], scale=[1,1,1], configuration=0, waitForConfirmation=False)
    
    # spawn crosswalk with degrees in config 1
    crosswalk.spawn_degrees(location=[-6.788, 45, 0.00], rotation=[0,0,90], scale=[1,1,1], configuration=1, waitForConfirmation=False)
    
    # spawn crosswalk with degrees in config 2
    crosswalk.spawn_degrees(location=[21.733, 3.347, 0.005], rotation=[0,0,0], scale=[1,1,1], configuration=2, waitForConfirmation=False)

    # spawn the last crosswalk with waitForConfirmation=True to confirm everything is flushed from the send buffers
    crosswalk.spawn_degrees(location=[21.733, 16, 0.005], rotation=[0,0,0], scale=[1,1,1], configuration=2, waitForConfirmation=True)


def spawn_signs(qlabs, right_hand_driving):
    # Like the crosswalks, we don't need to access the actors again after
    # creating them.

    roundabout_sign = QLabsRoundaboutSign(qlabs)
    yield_sign = QLabsYieldSign(qlabs)
    stop_sign = QLabsStopSign(qlabs)

    if (right_hand_driving):
        stop_sign.spawn_degrees([17.561, 17.677, 0.215], [0,0,90])
        stop_sign.spawn_degrees([24.3, 1.772, 0.2], [0,0,-90])
        stop_sign.spawn_degrees([14.746, 6.445, 0.215], [0,0,180])

        roundabout_sign.spawn_degrees([3.551, 40.353, 0.215], [0,0,180])
        roundabout_sign.spawn_degrees([10.938, 28.824, 0.215], [0,0,-135])
        roundabout_sign.spawn_degrees([24.289, 32.591, 0.192], [0,0,-90])

        yield_sign.spawn_degrees([-2.169, -12.594, 0.2], [0,0,180])
    else:
        stop_sign.spawn_degrees([24.333, 17.677, 0.215], [0,0,90])
        stop_sign.spawn_degrees([18.03, 1.772, 0.2], [0,0,-90])
        stop_sign.spawn_degrees([14.746, 13.01, 0.215], [0,0,180])

        roundabout_sign.spawn_degrees([16.647, 28.404, 0.215], [0,0,-45])
        roundabout_sign.spawn_degrees([6.987, 34.293, 0.215], [0,0,-130])
        roundabout_sign.spawn_degrees([9.96, 46.79, 0.2], [0,0,-180])

        yield_sign.spawn_degrees([-21.716, 7.596, 0.2], [0,0,-90])


def spawn_traffic_lights(qlabs, right_hand_driving):
    # In this case, we want to track each traffic light individually so we
    # can subsequently set the color state.  By using spawning with an ID,
    # we'll know exactly which one is which and this will allow us to also
    # reference them in separate programs, and we can also spawn without
    # waiting for confirmation because the object already knows its own ID.


    # initialize four traffic light instances in qlabs
    trafficLight1 = QLabsTrafficLight(qlabs)
    trafficLight2 = QLabsTrafficLight(qlabs)
    trafficLight3 = QLabsTrafficLight(qlabs)
    trafficLight4 = QLabsTrafficLight(qlabs)

    if (right_hand_driving):
        
        trafficLight1.spawn_id_degrees(actorNumber=0, location=[5.889, 16.048, 0.215], rotation=[0,0,0], configuration=0, waitForConfirmation=False)
        trafficLight2.spawn_id_degrees(actorNumber=1, location=[-2.852, 1.65, 0], rotation=[0,0,180], configuration=0, waitForConfirmation=False)
        trafficLight1.set_color(color=trafficLight1.COLOR_GREEN, waitForConfirmation=False)
        trafficLight2.set_color(color=trafficLight2.COLOR_GREEN, waitForConfirmation=False)

        trafficLight3.spawn_id_degrees(actorNumber=3, location=[8.443, 5.378, 0], rotation=[0,0,-90], configuration=0, waitForConfirmation=False)
        trafficLight4.spawn_id_degrees(actorNumber=4, location=[-4.202, 13.984, 0.186], rotation=[0,0,90], configuration=0, waitForConfirmation=False)
        trafficLight3.set_color(color=trafficLight3.COLOR_RED, waitForConfirmation=False)
        trafficLight4.set_color(color=trafficLight4.COLOR_RED, waitForConfirmation=False)  

    else:
        trafficLight1.spawn_id_degrees(actorNumber=0, location=[-2.831, 16.643, 0.186], rotation=[0,0,180], configuration=1, waitForConfirmation=False)
        trafficLight2.spawn_id_degrees(actorNumber=1, location=[5.653, 1.879, 0], rotation=[0,0,0], configuration=1, waitForConfirmation=False)
        trafficLight1.set_color(color=trafficLight1.COLOR_GREEN, waitForConfirmation=False)
        trafficLight2.set_color(color=trafficLight2.COLOR_GREEN, waitForConfirmation=False)

        trafficLight3.spawn_id_degrees(actorNumber=3, location=[8.779, 13.7, 0.215], rotation=[0,0,90], configuration=1, waitForConfirmation=False)
        trafficLight4.spawn_id_degrees(actorNumber=4, location=[-4.714, 4.745, 0], rotation=[0,0,-90], configuration=1, waitForConfirmation=False)
        trafficLight3.set_color(color=trafficLight3.COLOR_RED, waitForConfirmation=False)
        trafficLight4.set_color(color=trafficLight4.COLOR_RED, waitForConfirmation=False)                



def spawn_cones(qlabs):
    
    # We'll assume the cones don't need to be referenced after they're spawned so a
    # single class object will suffice for spawning.
    
    cone = QLabsTrafficCone(qlabs)

    for count in range(10):
        # Since we're going to set the color, we need to wait for QLabs to assign
        # an actor number.  This can be executed more quickly if you spawn by ID
        # instead and manually assign the numbers.
        #
        # Also note that since this are physics objects, it's a good idea to
        # spawn the actors slight above the surface so they can fall into place.
        # If you spawn exactly at ground level, they may "pop" up from the surface.

        cone.spawn(location=[-15.313, 35.374+count*-1.3, 0.25], configuration=1, waitForConfirmation=True)
        cone.set_material_properties(materialSlot=0, color=[0,0,0],roughness=1,metallic=False)
        cone.set_material_properties(materialSlot=1, color=HSVtoRGB([count/10, 1, 1]))        


def HSVtoRGB(hsv):

    H = hsv[0]
    S = hsv[1]
    V = hsv[2]

    kr = (5+H*6) % 6
    kg = (3+H*6) % 6
    kb = (1+H*6) % 6

    r = 1 - max(min(min(kr, 4-kr), 1), 0)
    g = 1 - max(min(min(kg, 4-kg), 1), 0)
    b = 1 - max(min(min(kb, 4-kb), 1), 0)
    
    return [r, g, b]


if __name__ == "__main__":
    main(right_hand_driving)