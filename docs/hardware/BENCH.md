# Hardware Lane Bench Procedure

> Record every smoke-test result in [BENCH_LOG.md](BENCH_LOG.md) as a dated row.

---

## 5.1 Jetson Setup Checklist

**OS / platform**: JetPack 6.2.x on NVMe (Jetson Orin Nano).

### Network

```bash
# Static IP (example — adapt to your network)
sudo nmcli con mod "Wired connection 1" ipv4.method manual \
     ipv4.addresses 192.168.1.200/24 ipv4.gateway 192.168.1.1 \
     ipv4.dns "8.8.8.8 8.8.4.4"
sudo nmcli con up "Wired connection 1"
```

Put the Jetson's static IP into `configs/hardware.yaml`:
```yaml
hardware:
  jetson_host: "192.168.1.200"
```

### Serial access

```bash
sudo usermod -aG dialout $USER   # re-login required
# Udev symlink for the Uno (use lsusb -v to find VID:PID; stock Uno is 2341:0043)
sudo tee /etc/udev/rules.d/99-servo-bridge.rules <<'EOF'
SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="0043", \
  SYMLINK+="servo_bridge", MODE="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

> **CH340 clones**: the `ch341` kernel module is missing on some JetPack 6 builds.
> Fallback: use a PCA9685 I2C board (`--driver pca9685`, Stage C) or a genuine Uno.

### Python dependencies (system Python 3.10)

```bash
pip3 install pyzmq msgpack numpy pyserial
# OpenCV: prefer JetPack's bundled OpenCV; if absent: pip3 install opencv-python
```

### systemd service

Install `jetson/robot_server.service` to start the robot server automatically:

```bash
sudo cp jetson/robot_server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now robot_server
sudo systemctl status robot_server
```

Service file: [`jetson/robot_server.service`](../../jetson/robot_server.service)

### NanoOWL (Shivanand)

Install via [jetson-containers](https://github.com/dusty-nv/jetson-containers) —
see Shivanand's setup notes. The `detector_service.py` is independent of `robot_server.py`.

---

## 5.2 Power & Safety

| Item | Requirement |
|---|---|
| Servo supply voltage | 5–6 V DC |
| Minimum current rating | ≥ 3 A (five MG90S stall ≈ 0.7 A each) |
| Common ground | Supply GND and Uno GND **must** be tied together |
| E-stop switch | Normally-closed switch in the **+V servo supply** line |
| First attach | Support the arm by hand; keep fingers clear of the elbow |
| USB power | **Never** power servos from the Uno's 5 V pin or from USB |

---

## 5.3 Smoke Tests

Record each result in [BENCH_LOG.md](BENCH_LOG.md) with date, initials, and actual vs. expected.

---

### S1 — Uno alone, serial monitor

**Setup**: Uno powered via USB only (no servo supply). Open serial monitor at 115200 baud.

**Steps**:
```
V        → expect: V servo_bridge 1 5
S        → expect: S 0 0 500 <p0..p4>    (attached=0 at boot)
P 1500 1500 1500 1500 1500  → OK          (would attach if supply present)
```
Stop typing for > 500 ms, then:
```
S        → expect: S 0 0 500 …           (watchdog fired, attached=0)
```

**PASS criterion**: Version reply includes `servo_bridge 1 5`; watchdog detaches within 500 ms of last frame.

**Status**: NOT RUN

---

### S2 — One servo on bench supply

**Setup**: One servo on channel 0 (pin 3), bench supply at 5–6 V.

**Steps**:
1. `P 1000 1500 1500 1500 1500` — servo moves toward min.
2. `P 2000 1500 1500 1500 1500` — servo moves toward max.
3. Verify no buzzing or stalling at the endpoints.
4. Run `scripts/calibrate_servos.py --port /dev/servo_bridge --calibration jetson/servo_calibration.json`; set `min`, `zero`, `max` for channel 0. Record in calibration file.

**PASS criterion**: Servo moves smoothly 1000–2000 µs; no brown-out; calibration file saved.

**Status**: NOT RUN

---

### S3 — All 5 channels, current check

**Setup**: All 5 servos wired to the bench supply. Ammeter in the +V line.

**Steps**:
```
T 2000 1000 1000 1000 1000 1000   → interpolate all to ~1000 µs over 2 s
T 2000 2000 2000 2000 2000 2000   → interpolate all to ~2000 µs over 2 s
```
Verify:
- Uno does **not** reset mid-motion (no second `V` banner).
- Supply current stays within supply rating.

**PASS criterion**: All 5 servos move; no Uno reset (no unexpected V banner); peak current within supply spec.

**Status**: NOT RUN

---

### S4 — Jetson -> Uno: full RPC path

**Setup**: Uno connected to Jetson via USB. Start the server:
```bash
python3 jetson/robot_server.py --driver uno_serial --host 0.0.0.0 --port 5560
```

From the **laptop**, run:
```python
from mfw.hardware.zmq_rpc import ZmqRpcClient
c = ZmqRpcClient("<jetson_host>", 5560)
print(c.ping())       # expect: ok=True, driver=uno_serial
print(c.call("get_state"))
print(c.call("home", {"duration_s": 3.0}))   # arm moves to zero pose
print(c.call("get_state"))                    # joint_positions near [0,0,0,0]
c.close()
```

**PASS criterion**: `ping` returns `driver=uno_serial`; `home` causes physical movement to zero pose; `get_state` shows `attached=True`.

**Status**: NOT RUN

---

### S5 — Estop under motion

**Setup**: Server running (S4 passed). Two terminal windows on the laptop.

**Terminal 1** — start a 5-second trajectory:
```python
c1 = ZmqRpcClient("<jetson_host>", 5560)
c1.call("follow_trajectory", {"points": [
    {"t": 0.0, "q": [0.0,  0.0,  0.0, 0.0]},
    {"t": 5.0, "q": [0.5, -0.3,  0.4, 0.2]},
]})
```

**Terminal 2** — within 2 s, estop:
```python
c2 = ZmqRpcClient("<jetson_host>", 5560)
print(c2.call("estop"))    # estopped=True; arm goes limp
c2.close()
```

Also: mid-motion, flip the **E-stop power switch** and verify arm goes limp < 1 s.

**PASS criterion**: Estop reply from c2 within 0.3 s; Terminal 1 raises `RpcError("EStopped")`; `get_state` shows `attached=False`; power-switch cuts motion within 1 s.

**Status**: NOT RUN

---

### S6 — Host-loss recovery

**Setup**: Server running; arm attached after `home`.

**Test A — host timeout (kill laptop client)**:
- Start laptop heartbeat: call `get_state` in a loop every 1 s for 3 s; then kill the process.
- After `host_timeout_s` (default 5 s), the server detaches servos.
- Verify: `get_state` from a new client shows `attached=False`.

**Test B — Uno watchdog (unplug USB mid-idle)**:
- Arm attached; server running.
- Unplug the USB cable between Jetson and Uno while arm is idle.
- Within 500 ms, the Uno watchdog fires; servos go limp.

**PASS criterion**: Test A: arm detaches ≤ 6 s after last host request. Test B: arm detaches ≤ 500 ms after USB pull.

**Status**: NOT RUN
