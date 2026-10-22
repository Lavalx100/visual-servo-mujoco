# Three-arm cooperative pick and lift

[Project overview](../../README.md) · [Demo gallery](../video-gallery.md)

## Demo

Three articulated arms close six sliding fingers around one free MuJoCo payload,
raise it together, hold it above the table, lower it, and release. Arm bases,
links, palms, and fingertip pads have colliders. A render-only wrist adapter
overlaps each wrist link and palm box. Palm-mounted rails meet the sliding
carriages, whose finger shanks terminate at the pads. These visible connection
pieces are non-colliding; the pads remain the modeled finger contact surfaces.
Per-arm collision masks allow the arms to contact one another, the payload,
table, and ground while suppressing same-arm self-collision. The payload is not
teleported or attached with a weld; the lift is produced by simulated pad
contact and friction.

- [Oblique hero video](../../assets/demos/cooperative_pick_triarm.mp4) ·
  [overhead video](../../assets/demos/cooperative_pick_triarm_top.mp4) ·
  [gripper close-up](../../assets/demos/cooperative_pick_triarm_closeup.mp4) ·
  [close-up GIF preview](../../assets/demos/cooperative_pick_triarm_closeup.gif)
- [Recorded run metrics](../../assets/demos/cooperative_pick_triarm.json)

![Three articulated arms hold the canister above the tabletop](../../assets/demos/cooperative_pick_triarm.gif)

![Top and side view of the arm collision masks and separated finger contact bands](../../assets/demos/cooperative_pick_collision_layout.svg)

## Mechanism and control

The scene contains three arms arranged 120 degrees apart. Each arm has three
revolute joints in its vertical work plane and a two-finger gripper with
independent position actuators. Their wrist centers stop 84 mm from the
payload center, leaving about 9 mm between the 19 mm wrist-link radius and the
56 mm payload radius. Small pads sit 22 mm inward from each wrist. When closed,
the six pad centers lie at 60-degree intervals around one equatorial contact
band; their nominal radial component is 62 mm and their tangential component
is 35.8 mm. The canister is 112 mm in diameter and 120 mm tall.

A closed-form elbow-up 3R inverse-kinematics calculation converts the shared
payload-height trajectory into joint references. The elbow-up branch keeps the
links above the tabletop. During the open approach, tool heights are staggered
45 mm to avoid pad crowding. As the jaw offset closes below 60 mm, the
trajectory smoothly brings all pads to the payload equator. Arm collision bits
are 1, 2, and 4;
each arm's affinity includes the other two arm bits, payload bit 8, table bit
16, and ground bit 32, but excludes its own bit. The payload, table, and ground
affinities accept the arm bits, so collision detection is enabled across
different arms and between each arm and the workcell. Same-arm self-contact is
suppressed by the masks. MuJoCo position actuators and rigid-body contact
generate the motion and grasp forces.

The [collision layout diagram](../../assets/demos/cooperative_pick_collision_layout.svg)
shows the radial wrist clearance, six finger positions, staggered approach,
equatorial grasp, and contact-bit rule. The implementation is in
[`cooperative_pick_demo.py`](../../src/visual_servo_mujoco/cooperative_pick_demo.py);
the mask and physics assertions are in
[`test_cooperative_pick_demo.py`](../../tests/test_cooperative_pick_demo.py).

This original demonstration does not use the project's overhead camera,
target detector, or image-feedback controller. It remains the scripted
centered-pick baseline. The separate
[camera-guided extension](camera-guided-cooperative-pick.md) connects the same
collision scene and trajectory to the project's existing target detector and
pixel-to-table projection, then measures randomized camera and oracle
conditions. The two experiments have separate methods and results.

## Recorded outcome

The checked-in clips are three synchronized views of one deterministic
simulation run, not a benchmark. The free payload rose 9.48 cm from its initial
height. It remained at least 8.98 cm above its starting height during the 1.6 s
hold, with at least five finger-payload contact points and no more than 2.48° of
tilt. There were no inter-arm contacts during grasping, lifting, or the hold.
Opening the grippers for release produced brief finger-pad contacts, with a
maximum measured penetration of 0.72 mm. The run had zero arm-table,
arm-ground, or non-finger arm-to-payload contacts. The object returned to the
table at its starting height after release. The generated
[JSON record](../../assets/demos/cooperative_pick_triarm.json) contains these
measurements.

## Limits

- The arm trajectories and pickup pose are scripted and known in advance.
- Same-arm self-collision is suppressed by collision masks, so the demo checks
  inter-arm and environment collisions but does not model a robot's internal
  link self-collision constraints.
- The simulator's contact solver permits small pad overlap during release;
  this run remained below the documented 1 mm limit. It recorded no
  cross-arm contacts during the grasp and lift phases.
- The three arms each move in a fixed vertical plane; this clip demonstrates a
  shared vertical lift, not arbitrary coordinated transport.
- The payload, friction, camera views, and lighting are selected simulator
  conditions. The video is not evidence of hardware performance.
- This is one successful example. It does not measure success rate, robustness,
  perception, grasp planning, or task completion under uncertain object pose.

## Reproduce

From a checkout with the `demo` extra installed:

```bash
uv sync --locked --extra dev --extra demo
uv run --locked visual-servo-cooperative-pick
```

The command writes three MP4 camera views, a GIF preview, and JSON run record to
`assets/demos/`. The physics check used by the test suite runs the same task
without rendering and asserts collision-mask compatibility, zero inter-arm
contacts, and a contact-driven lift. See the
[reproduction guide](../reproduction.md) and
[video gallery](../video-gallery.md) for the other project outputs.
