"""A tiny tabletop scene defined inline, so the demo needs no model downloads."""

MODEL_XML = """
<mujoco model="camera_reaching_demo">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <size njmax="1000" nconmax="200"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <quality shadowsize="2048"/>
  </visual>
  <default>
    <joint damping="1.5" armature="0.01" limited="true"/>
    <geom friction="0.8 0.1 0.1" density="700"/>
  </default>
  <worldbody>
    <light pos="0 0 3" directional="false" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
    <camera name="overhead" pos="0 0 2" fovy="40"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.12 0.14 0.17 1"/>
    <geom name="table" type="box" pos="0 0 0.07" size="0.75 0.75 0.07"
          rgba="0.78 0.73 0.62 1"/>

    <body name="arm_base" pos="0 0 0.21">
      <geom name="base_visual" type="cylinder" size="0.075 0.10" pos="0 0 -0.05"
            rgba="0.20 0.24 0.30 1"/>
      <body name="upper_arm">
        <joint name="shoulder_joint" type="hinge" axis="0 0 1" range="-2.8 2.8"/>
        <geom name="upper_link" type="capsule" fromto="0 0 0 0.42 0 0" size="0.035"
              rgba="0.15 0.38 0.78 1"/>
        <body name="forearm" pos="0.42 0 0">
          <joint name="elbow_joint" type="hinge" axis="0 0 1" range="-2.6 2.6"/>
          <geom name="forearm_link" type="capsule" fromto="0 0 0 0.34 0 0" size="0.030"
                rgba="0.18 0.53 0.90 1"/>
          <site name="end_effector" type="sphere" pos="0.34 0 0" size="0.045"
                rgba="0.15 0.82 0.38 1"/>
        </body>
      </body>
    </body>

    <body name="target" pos="0.52 0.12 0.165">
      <geom name="target_visual" type="cylinder" size="0.045 0.025"
            rgba="0.95 0.08 0.04 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_position" joint="shoulder_joint" kp="220" kv="24"
              forcerange="-120 120"/>
    <position name="elbow_position" joint="elbow_joint" kp="180" kv="20"
              forcerange="-80 80"/>
  </actuator>
</mujoco>
"""
