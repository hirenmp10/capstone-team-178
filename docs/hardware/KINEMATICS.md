# Planar 4-DoF Kinematics, Calibration & Mechanics Specification

**Author / Lane Owner:** Gowtham B. (Mechanics, Kinematics, Calibration)  
**Target Mechanism:** 4-DoF Planar Hobby Manipulator (Base Yaw + 3 Pitch Links + Parallel Gripper)

---

## 1. Coordinate Frames & Conventions

### 1.1 Base & Table Frame
- **Origin ($O_{\text{base}}$):** Intersection of the vertical base yaw axis with the table surface ($Z = 0$).
- **$+Z$ Axis:** Points vertically upward normal to the table plane.
- **$+X$ Axis:** Arm zero-bearing pointing forward in the robot base reference frame.
- **$+Y$ Axis:** Completes the right-handed Cartesian coordinate system ($+Y = +Z \times +X$).
- **Table Frame:** Identical to the Robot Base frame ($Z=0$ is the table plane).

```
        +Z (Up)
          ^
          |
          |       +X (Bearing 0)
          |      /
          |     /
          |    /
          |   /
  (0,0,0) +------------------> +Y (Left)
   [Base Origin on Table]
```

---

## 2. Joint Order, Zero Pose & Sign Conventions

### 2.1 Joint Order
The manipulator consists of exactly 4 active joints in serial kinematic chain:
1. `joint 0: base_yaw` ($q_1$) — Turntable rotation about $+Z$.
2. `joint 1: shoulder` ($q_2$) — Pitch in the vertical radial plane.
3. `joint 2: elbow` ($q_3$) — Pitch in the vertical radial plane.
4. `joint 3: wrist` ($q_4$) — Pitch in the vertical radial plane.

### 2.2 Zero Pose $[0, 0, 0, 0]$ (radians)
- In the zero configuration $[q_1, q_2, q_3, q_4] = [0, 0, 0, 0]$:
  - Upper arm link ($L_1$), forearm link ($L_2$), and tool link ($L_3$) are completely collinear along $+Z$.
  - The entire arm points vertically straight upward.
  - TCP coordinates at zero pose: $[0, 0, h + L_1 + L_2 + L_3]$.

### 2.3 Sign Convention
- **$q_1$ (base_yaw):** Right-handed rotation around $+Z$ axis.
- **$q_2$ (shoulder):** Positive pitch rotates the upper arm link outward and downward away from $+Z$ toward the table.
- **$q_3$ (elbow):** Positive pitch bends the forearm link further outward/downward relative to the upper arm.
- **$q_4$ (wrist):** Positive pitch bends the tool link further outward/downward relative to the forearm.

---

## 3. Geometric Parameterization

All link parameters are defined in `HardwareArmConfig`:

| Parameter | Symbol | Description | Unit | Status |
|---|---|---|---|---|
| `base_height` | $h$ | Table plane to shoulder pitch axis along yaw axis | m | `PLACEHOLDER — MEASURE` |
| `shoulder_offset` | $s$ | Horizontal radial offset from yaw axis to shoulder axis | m | `PLACEHOLDER — MEASURE` |
| `upper_arm` | $L_1$ | Distance between shoulder axis and elbow axis | m | `PLACEHOLDER — MEASURE` |
| `forearm` | $L_2$ | Distance between elbow axis and wrist axis | m | `PLACEHOLDER — MEASURE` |
| `tool` | $L_3$ | Distance between wrist axis and TCP (jaw midpoint) | m | `PLACEHOLDER — MEASURE` |

---

## 4. Forward Kinematics (FK)

Given joint angles $\mathbf{q} = [q_1, q_2, q_3, q_4]^T$:

### 4.1 Cumulative Pitch Angles
$$\phi_1 = q_2$$
$$\phi_2 = q_2 + q_3$$
$$\phi_3 = q_2 + q_3 + q_4$$

### 4.2 Radial Reach ($r$) and Height ($z$)
$$r = s + L_1 \sin(\phi_1) + L_2 \sin(\phi_2) + L_3 \sin(\phi_3)$$
$$z = h + L_1 \cos(\phi_1) + L_2 \cos(\phi_2) + L_3 \cos(\phi_3)$$

### 4.3 3D Cartesian Position
$$x = r \cos(q_1)$$
$$y = r \sin(q_1)$$
$$\mathbf{p}_{\text{tcp}} = [x, y, z]^T$$

### 4.4 TCP Orientation & Frame
- **Local $+Z$ (Approach Axis $\mathbf{z}_{\text{tcp}}$):**
  $$\mathbf{z}_{\text{tcp}} = \begin{bmatrix} \sin(\phi_3) \cos(q_1) \\ \sin(\phi_3) \sin(q_1) \\ \cos(\phi_3) \end{bmatrix}$$
  - $\phi_3 = 0$: points straight up $[0, 0, 1]^T$.
  - $\phi_3 = \pi$: points straight down $[0, 0, -1]^T$ (top-down grasp).
  - $\phi_3 = \pi/2$: points horizontally outward $[\cos(q_1), \sin(q_1), 0]^T$.

- **Local $+Y$ (Jaw Closing Axis $\mathbf{y}_{\text{tcp}}$):**
  - For `jaw_axis = "tangential"`:
    $$\mathbf{y}_{\text{tcp}} = \begin{bmatrix} -\sin(q_1) \\ \cos(q_1) \\ 0 \end{bmatrix}, \quad \mathbf{x}_{\text{tcp}} = \mathbf{y}_{\text{tcp}} \times \mathbf{z}_{\text{tcp}}$$
  - For `jaw_axis = "radial"`:
    $$\mathbf{x}_{\text{tcp}} = \begin{bmatrix} -\sin(q_1) \\ \cos(q_1) \\ 0 \end{bmatrix}, \quad \mathbf{y}_{\text{tcp}} = \mathbf{z}_{\text{tcp}} \times \mathbf{x}_{\text{tcp}}$$

- **Rotation Matrix & Quaternion:**
  $$\mathbf{R} = [\mathbf{x}_{\text{tcp}} \quad \mathbf{y}_{\text{tcp}} \quad \mathbf{z}_{\text{tcp}}]$$
  $$\mathbf{q}_{\text{quat}} = \text{matrix\_to\_quat}(\mathbf{R}) \quad (\text{scalar-first } [w, x, y, z])$$

---

## 5. Inverse Kinematics (IK)

Given target TCP `Pose` ($\mathbf{p} = [x, y, z]^T, \mathbf{R} = [\mathbf{x}_{\text{target}} \; \mathbf{y}_{\text{target}} \; \mathbf{z}_{\text{target}}]$):

### 5.1 Base Yaw ($q_1$)
$$r_{xy} = \sqrt{x^2 + y^2}$$
If $r_{xy} < 10^{-6}$: return `None` (on yaw axis).
$$q_1 = \text{atan2}(y, x)$$

### 5.2 Orientation Consistency Checks
Let $\hat{r} = [\cos q_1, \sin q_1, 0]^T$, $\hat{t} = [-\sin q_1, \cos q_1, 0]^T$.
1. **Approach Planarity:** $|\mathbf{z}_{\text{target}} \cdot \hat{t}| \le 10^{-3}$. (Reject if out of radial plane).
2. **Jaw Axis Alignment:**
   - If `tangential`: $|\mathbf{y}_{\text{target}} \cdot \hat{t}| \ge 1.0 - 10^{-3}$.
   - If `radial`: $|\mathbf{y}_{\text{target}} \cdot \hat{t}| \le 10^{-3}$.

### 5.3 Pitch Sum ($\phi_3$) and Wrist Center
$$\phi_3 = \text{atan2}(\mathbf{z}_{\text{target}} \cdot \hat{r}, \mathbf{z}_{\text{target}}[2])$$
$$\mathbf{p}_w = \mathbf{p} - L_3 \mathbf{z}_{\text{target}}$$
$$r_w = \mathbf{p}_w \cdot \hat{r}, \quad z_w = \mathbf{p}_w[2]$$
$$r_{\text{local}} = r_w - s, \quad z_{\text{local}} = z_w - h$$
$$D = \sqrt{r_{\text{local}}^2 + z_{\text{local}}^2}$$

### 5.4 Two-Link Subproblem ($q_3, q_2, q_4$)
Reachability check:
$$|L_1 - L_2| - \epsilon \le D \le L_1 + L_2 + \epsilon \quad (\text{return None if outside})$$
$$\cos(q_3) = \text{clamp}\left(\frac{D^2 - L_1^2 - L_2^2}{2 L_1 L_2}, -1.0, 1.0\right)$$
$$\sin(q_3)_{\text{up}} = -\sqrt{1 - \cos^2 q_3}, \quad \sin(q_3)_{\text{down}} = +\sqrt{1 - \cos^2 q_3}$$

For each branch ($k \in \{\text{up, down}\}$):
$$q_3 = \text{atan2}(\sin q_{3,k}, \cos q_3)$$
$$A = L_1 + L_2 \cos q_3, \quad B = L_2 \sin q_{3,k}$$
$$q_2 = \text{atan2}(A r_{\text{local}} - B z_{\text{local}}, A z_{\text{local}} + B r_{\text{local}})$$
$$q_4 = \text{wrap}(\phi_3 - q_2 - q_3)$$

### 5.5 Branch Selection & Tie-Breaking
- **Primary:** Preference dictated by `arm.elbow_up` (`True` chooses elbow-up branch $q_3 \le 0$).
- **Fallback:** If primary violates joint limits, evaluate alternate branch.
- **Deterministic Seed:** When both branches satisfy joint limits and `seed` is provided, the branch with minimal Euclidean joint-space distance $\|\mathbf{q} - \mathbf{q}_{\text{seed}}\|_2$ is returned.
- If neither branch satisfies joint limits: return `None`.

---

## 6. Python API (`mfw.hardware.kinematics.PlanarKinematics`)

```python
from mfw.config.schema import HardwareArmConfig
from mfw.core.types import Pose
from mfw.hardware.kinematics import PlanarKinematics

arm_cfg = HardwareArmConfig(...)
kin = PlanarKinematics(arm_cfg, ("base_yaw", "shoulder", "elbow", "wrist"))

# Forward Kinematics
pose: Pose = kin.fk([0.0, 0.8, -0.6, 0.8])

# Inverse Kinematics
q: np.ndarray | None = kin.ik(pose, seed=[0.0, 0.0, 0.0, 0.0])

# Joint Limits Check
is_valid: bool = kin.within_limits([0.0, 0.8, -0.6, 0.8])
```

---

## 7. Configuration Ownership & Boundaries

```
+-------------------------------------------------------------------------+
|                              OWNERSHIP                                  |
+-------------------------------------------------------------------------+
|  configs/hardware.yaml (Gowtham B.)                                     |
|    - Kinematic link lengths: base_height, upper_arm, forearm, tool      |
|    - Kinematic joint limits in radians (joint_lower, joint_upper)       |
|    - Jaw closing orientation: jaw_axis ("tangential" / "radial")        |
|    - Camera table homography matrix (exterior_camera.homography)        |
|    - Object nominal bounding box sizes (hardware.object_sizes)          |
+-------------------------------------------------------------------------+
                                    |
                            [DO NOT CROSS]
                                    |
+-------------------------------------------------------------------------+
|  jetson/servo_calibration.json (Hiren M. P.)                            |
|    - Servo raw pulse mapping (pulse_zero_us, pulse_min_us, pulse_max_us)|
|    - Microsecond scaling factors (us_per_rad)                           |
|    - Servo direction polarities (+1 / -1)                               |
+-------------------------------------------------------------------------+
```

---

## 8. Camera-to-Table Homography Calibration

Script: [`scripts/calibrate_table.py`](file:///c:/Users/acer/Desktop/Gowtham_B_Venture_Creed_Case_Study/Capstone/scripts/calibrate_table.py)

### 8.1 Mathematical Model
The mapping from camera image pixel coordinates $(u, v)$ to table coordinates $(X, Y, 1)$ in the robot base frame is modeled by a $3 \times 3$ projective homography matrix $\mathbf{H}$:
$$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = \mathbf{H} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \quad X = \frac{x'}{w'}, \quad Y = \frac{y'}{w'}$$

### 8.2 Modes of Operation
- `--dry-run`: Runs synthetic calibration simulation and verifies numerical residual $< 0.1$ mm.
- `--measure`: Manual correspondence entry without robot movement.
- `--touch`: Interactive robot touch calibration via `jetson/robot_server.py` RPC.
- `--solve`: Solves $\mathbf{H}$ using `cv2.findHomography` / DLT and displays residuals in mm.
- `--write`: Writes the solved 9-element array to `exterior_camera.homography` in `configs/hardware.yaml`.

---

## 9. Physical Measurement Checklist (Bench Procedures)

Before operating on real physical hardware, perform the following physical caliper/ruler measurements:

1. **`base_height` ($h$):** Measure vertical distance from table surface to the center of the shoulder pitch bolt.
2. **`shoulder_offset` ($s$):** Measure horizontal radial distance from turntable center axis to shoulder pitch bolt. (Typically $0.0$ for concentric turntable).
3. **`upper_arm` ($L_1$):** Measure center-to-center distance between shoulder pivot bolt and elbow pivot bolt.
4. **`forearm` ($L_2$):** Measure center-to-center distance between elbow pivot bolt and wrist pivot bolt.
5. **`tool` ($L_3$):** Measure distance along the tool axis from wrist pivot bolt to the center of the rubber gripper pads when closed.
6. **`joint_lower` / `joint_upper`:** Verify physical hard-stops for each joint so the kinematic limits never drive into mechanical collisions.
7. **Object Bounding Sizes:** Measure caliper dimensions $[w, d, h]$ in metres for:
   - `marker`, `banana`, `box`, `cube`, `bowl`, `bin`.

---

## 10. Safety Architecture & Operational Limits

- **Maximum Joint Step:** Touch calibration moves with incremental joint steps $\le 0.03$ rad.
- **Confirmation Gate:** Interactive user confirmation prompt is enforced before every robot motion command.
- **Fail-Safe Interrupt:** Ctrl-C (`SIGINT`) triggers safe halt / E-Stop through `robot_server` RPC.
- **No Direct Hardware Access:** Client scripts never open raw serial ports directly; all hardware communication routes through `jetson/robot_server.py`.
