import signal
import sys
import time

import mujoco
import mujoco.viewer

signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

xml = """
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="5 5 0.1" rgba=".9 .9 .9 1"/>
    <body pos="0 0 2" name="box">
      <freejoint/>
      <geom type="box" size=".1 .1 .1" rgba="1 0 0 1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print("Запуск MuJoCo viewer... (Ctrl+C для выхода)")
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
