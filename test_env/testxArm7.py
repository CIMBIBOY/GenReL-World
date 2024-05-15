import time
import mujoco
import mujoco.viewer

import ctypes

# source /Users/czimbermark/Documents/Reinf/MetaWorld/GenReL-World/.venv/bin/activate

m = mujoco.MjModel.from_xml_path('GenReL-World/test_env/xarm7.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        # Rotate joints for visualization
        for i in range(m.nu):
            d.qpos[m.jnt_qposadr[i]] += 0.01  # Adjust the increment as needed for desired speed
            d.qpos[m.jnt_qposadr[i]] %= 2 * 3.14159  # Ensure within bounds of 2*pi

        # Update physics state and sync viewer
        mujoco.mj_step(m, d)
        viewer.sync()

        # Delay to control the speed of movement
        time.sleep(0.01)  # Adjust as needed for desired speed