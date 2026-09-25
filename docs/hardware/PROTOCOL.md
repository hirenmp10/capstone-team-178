# Hardware Lane Protocol Specification

## 1.1 Endpoint Table

| Port | Transport | Service | Runs on | Owner |
|---|---|---|---|---|
| `5560` | ZMQ ROUTER (server) / REQ (client), msgpack | `jetson/robot_server.py` | Jetson (laptop for `--fake-hardware`) | Hiren |
| `5558` | TCP newline-JSON | `jetson/detector_service.py` | Jetson (NanoOWL) or laptop (Florence-2/scripted) | Shivanand |
| `5557` | ZMQ | `scripts/llm_worker.py` (Qwen) | laptop | Adyanth |
| `5555` | ZMQ msgpack | `scripts/groot_server.py` | laptop GPU | Adyanth |
| `/dev/ttyACM0` 115200 8N1 | USB-CDC serial | Arduino `servo_bridge.ino` | Jetson <-> Uno | Hiren (Stage B - PLANNED) |

---

## 1.2 RPC Envelope

All ZeroMQ RPC messages over port 5560 use msgpack serialization (`use_bin_type=True`, `raw=False`).

### Request Structure
```json
{
  "id": 1,
  "method": "get_state",
  "params": {}
}
```

### Reply Structure
Replies always return a dictionary containing `"ok": bool`.

- **Success**: Result fields are merged flat into the reply dictionary.
  ```json
  {
    "ok": true,
    "estopped": true
  }
  ```
- **Failure**: Includes `"error"` and `"error_type"`.
  ```json
  {
    "ok": false,
    "error": "<message>",
    "error_type": "<ClassName>"
  }
  ```
  The client (`ZmqRpcClient`) raises `RpcError(error_type, message)`.

### Server Socket Architecture
The server binds a `ZMQ.ROUTER` socket so that requests from multiple clients (or concurrent channels) can be processed concurrently. In particular, an `estop` request from a second client is answered out-of-band while a first client is blocked awaiting completion of a motion request (`follow_trajectory`, `home`, `set_gripper`).

Motion requests block until the physical or simulated motion completes or is aborted, after which the reply is dispatched.

---

## 1.3 Methods

All replies include `"ok": bool`.

### `ping`
- **Params**: `{}`
- **Reply**:
  ```json
  {
    "ok": true,
    "server": "robot_server",
    "version": 1,
    "driver": "fake" | "uno_serial" | "pca9685",
    "uptime_s": 12.34
  }
  ```

### `get_state`
- **Params**: `{}`
- **Reply**:
  ```json
  {
    "ok": true,
    "joint_positions": [0.0, 0.0, 0.0, 0.0],
    "joint_names": ["base_yaw", "shoulder", "elbow", "wrist"],
    "gripper_width_m": 0.08,
    "attached": false,
    "moving": false,
    "estopped": false,
    "last_request_age_s": 0.002
  }
  ```
  *Note: Any incoming request (including `get_state`) resets the host inactivity timer (heartbeat).*

### `estop`
- **Params**: `{}`
- **Reply**:
  ```json
  {
    "ok": true,
    "estopped": true
  }
  ```
  *Behavior: Immediately detaches all servos, aborts any currently running motion (the blocked caller receives `ok: false, error_type: "EStopped"`), and latches the stopped state. Handled out-of-band; never queued behind running motions.*

### `clear_estop`
- **Params**: `{}`
- **Reply**:
  ```json
  {
    "ok": true,
    "estopped": false
  }
  ```
  *Behavior: Clears the E-stop latch. Does NOT re-attach servos; the arm remains detached until the next motion request.*

### `home`
- **Params**: `{"duration_s": float = 2.0}`
- **Reply**:
  ```json
  {
    "ok": true,
    "joint_positions": [0.0, 0.0, 0.0, 0.0]
  }
  ```
  *Behavior: Moves joints to calibrated zero pose (`[0.0, 0.0, 0.0, 0.0]` rad) with gripper open (`width_open_m`).*

### `follow_trajectory`
- **Params**:
  ```json
  {
    "points": [
      {"t": 0.0, "q": [0.0, 0.0, 0.0, 0.0]},
      {"t": 1.0, "q": [0.1, -0.2, 0.3, 0.0]}
    ],
    "gripper_width_m": 0.04 | null
  }
  ```
  *Note: `t` represents seconds from chunk start (monotonic, first may be 0); `q` is a 4-element joint vector in radians.*
- **Reply**:
  ```json
  {
    "ok": true,
    "joint_positions": [0.1, -0.2, 0.3, 0.0],
    "gripper_width_m": 0.04,
    "aborted": false
  }
  ```
  *Behavior: Linearly interpolates between waypoints at the follower tick rate (20 ms). Joint targets are clamped to calibration limits. If any target requires clamping by more than 0.02 rad, the server returns `{"ok": false, "error_type": "JointLimit", "error": "..."}` BEFORE initiating any motion.*

### `set_gripper`
- **Params**: `{"width_m": float, "duration_s": float = 0.5}`
- **Reply**:
  ```json
  {
    "ok": true,
    "gripper_width_m": 0.04
  }
  ```

### `get_frame`
- **Params**: `{}`
- **Reply**:
  ```json
  {
    "ok": true,
    "jpeg": "<bytes>",
    "width": 640,
    "height": 480,
    "stamp_s": 1727271234.56
  }
  ```
  *Behavior: Returns `"error_type": "NoCamera"` if no camera is attached. When started with `--fake-camera`, generates a synthetic 640x480 JPEG (flat grey with text label).*

### `get_world`
- **Params**: `{}`
- **Reply**:
  ```json
  {
    "ok": true,
    "objects": {
      "marker": [0.18, 0.05, 0.0],
      "bowl": [0.15, -0.12, 0.0]
    },
    "attached": null,
    "tcp": null
  }
  ```
  *Behavior: Supported only for fake driver (`"error_type": "NotSupported"` otherwise). In Stage A, returns static scene parsed from `--fake-world` with `attached=null` and `tcp=null`. Attachment physics requires `mfw.hardware.kinematics` (TODO: integrate when `mfw.hardware.kinematics` is added).*

### `get_calibration`
- **Params**: `{}`
- **Reply**: Dictionary containing the serialized `ServoCalibration`.

### Error Responses
- Unknown method: `{"ok": false, "error": "Unknown method: <name>", "error_type": "UnknownMethod"}`
- Bad parameters: `{"ok": false, "error": "<reason>", "error_type": "BadRequest"}`

---

## 1.4 Safety Behaviour

1. **Host Timeout (`host_timeout_s`, default 5.0 s)**:
   If no RPC request is received within `host_timeout_s`, the server automatically detaches all servos to prevent overheating or holding torque unattended. Any ongoing motion counts as alive.
2. **Driver Watchdog (`watchdog_ms`, default 500 ms)**:
   The follower thread feeds the driver at >= 10 Hz (every 20 ms) while attached. If the driver is not updated within `watchdog_ms`, it detaches itself and `get_state` reports `attached: false`. In `FakeDriver`, this simulates the Uno firmware watchdog and proves keepalive integrity.
3. **Pulse Clamping**:
   Every pulse command sent to a servo channel is clamped to `[pulse_min_us, pulse_max_us]`.
4. **Boot State**:
   Servos boot in the **detached** state. No torque is applied until the first motion command (`home`, `follow_trajectory`, `set_gripper`) is executed.

---

## 1.5 Serial Protocol: Uno <-> Jetson (PLANNED, Stage B)

ASCII line-oriented protocol, 115200 baud, 8N1.

### Host -> Uno Commands
- `P p0 p1 ... pN\n` : Set servo pulse targets (in µs) immediately.
- `T ms p0 ... pN\n` : Interpolate to servo pulse targets over `ms` milliseconds.
- `D\n` : Detach all servos.
- `W ms\n` : Configure watchdog timeout in milliseconds.
- `S\n` : Query status.
- `V\n` : Query firmware version.

### Uno -> Host Responses
- `OK\n` : Command accepted.
- `ERR <reason>\n` : Error executing command.
- `S <attached:0|1> <moving:0|1> p0 ... pN\n` : Current status and channel pulses.
- `V servo_bridge 1 <nch>\n` : Firmware version and channel count.

### Safety Rules
- Servos start detached at boot until the first `P` or `T` command.
- If no command is received within `watchdog_ms`, all servos detach immediately.
