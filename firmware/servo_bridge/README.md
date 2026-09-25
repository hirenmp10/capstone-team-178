# servo_bridge — Arduino Uno firmware

Bridges the Jetson's USB serial port to five MG90S hobby servos (or any
SG90-class servo) via ASCII line commands.

---

## Hardware requirements

| Item | Value |
|---|---|
| MCU | Arduino Uno R3 (ATmega328P) |
| Servo library | Built-in `Servo` (no extra install) |
| Channels | 5 (base_yaw, shoulder, elbow, wrist, jaw) |
| Signal pins | 3, 5, 6, 9, 10 (edit `SERVO_PINS[]` in the sketch) |
| Servo supply | 5–6 V, ≥ 3 A (separate from USB; MG90S stalls ≈ 0.7 A each) |
| Baud | 115200 8N1 |

---

## How to flash

### Option A — Arduino IDE (GUI)

1. Open `firmware/servo_bridge/servo_bridge.ino` in Arduino IDE 2.x.
2. **Tools → Board → Arduino Uno**.
3. **Tools → Port → `/dev/ttyACM0`** (Linux/Jetson) or `COM<n>` (Windows).
4. Click **Upload** (⇧⌘U / Ctrl+U).

### Option B — arduino-cli (headless, Jetson or CI)

```bash
# Install arduino-cli (one-time)
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh

# Install the AVR core (one-time)
arduino-cli core install arduino:avr

# Compile and upload
arduino-cli compile --fqbn arduino:avr:uno firmware/servo_bridge
arduino-cli upload  --fqbn arduino:avr:uno --port /dev/ttyACM0 firmware/servo_bridge
```

---

## Quick test in the serial monitor

Open the serial monitor at **115200 baud**, line endings **Newline only**.

```
V                                 → V servo_bridge 1 5
S                                 → S 0 0 500 1500 1500 1500 1500 1500
P 1500 1500 1500 1500 1500        → OK   (arm moves to mid-point, all 5 channels)
```

Now **stop typing for > 500 ms** (the default watchdog):

```
S                                 → S 0 0 500 …   (attached=0: watchdog fired)
```

Sweep the base joint slowly:

```
T 2000 1000 1500 1500 1500 1500   → OK   (interpolates over 2 s)
```

Send a detach at any time:

```
D                                 → OK   (servos go limp immediately)
```

---

## Safety checklist before first power-on

- [ ] Servo +V from the bench supply — **not** the Uno 5 V pin.
- [ ] Common GND: supply GND and Uno GND connected.
- [ ] E-stop switch wired in the +V line.
- [ ] Arm supported by hand for the first P command.
- [ ] Fingers clear of the elbow joint.
