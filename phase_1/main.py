#region
from pal.products.qcar import QCar, IS_PHYSICAL_QCAR
import sensor_interfacing
import state_estimation
#endregion



#region : Main Execution

# for virtual QCar setup
if not IS_PHYSICAL_QCAR:
    import main_setup as qlabs_setup
    qlabs_setup.setup()


def sensor_main():
    try:
        sensor_data = sensor_interfacing.sensorInterfacing(taskRate=120, specifiedSamples=600)
        sensor_data.start(mode="sensor_stats") # read or sensor_stats
        state_estimate = state_estimation.StateEstimation(tf = 10, controllerUpdateRate=100, 
                                                            calibrate = False)
        state_estimate.start()
    finally:
        print("Execution completed.")

if __name__ == "__main__":
    sensor_main()
