/*
 * servo_bridge.ino -- Arduino Uno as a USB-CDC servo bridge for the hobby arm.
 *
 * Why an Uno at all: hardware-timed 50 Hz pulses that keep coming while the
 * Jetson's Python is stalled in a CUDA allocation, a free watchdog that goes
 * limp instead of holding a stall current, and no Jetson.GPIO/pinmux work on
 * JetPack 6. The Jetson only ever sends *targets*; this sketch owns timing.
 *
 * Wire order everywhere (frames, pins, calibration YAML):
 *     0 base_yaw  D3      1 shoulder_pitch  D5      2 elbow_pitch  D6
 *     3 wrist_pitch D9    4 gripper  D10
 *
 * Protocol, 115200 baud, newline-terminated ASCII, one reply line per frame:
 *     P<us>,<us>,<us>,<us>,<us>\n      targets; slewed at SLEW_US_PER_TICK per
 *                                       TICK_MS so a big jump cannot snap the
 *                                       PLA linkages           -> OK
 *     T<us>,<us>,<us>,<us>,<us>,<ms>\n  targets reached by linear interpolation
 *                                       over <ms> milliseconds -> OK
 *     K\n                               keepalive: refreshes the watchdog and
 *                                       nothing else (no attach, no target
 *                                       change, no effect on a running T
 *                                       move)                  -> OK
 *     ?\n                               -> Q<us>,<us>,<us>,<us>,<us>,<attached>,<known>
 *                                       <known> is have_position: 0 until the
 *                                       first P/T frame since power-up/reset,
 *                                       so the host can tell a freshly reset
 *                                       Uno (1500 us is a library default, not
 *                                       a position) from a detached one
 *     S<n>\n                            sync: echoes S<n> (n = decimal nonce).
 *                                       The host sends it after a reply
 *                                       timeout and reads until the echo, so a
 *                                       late reply can never leave the stream
 *                                       one frame behind       -> S<n>
 *     D\n                               detach all (arm goes limp) -> OK
 *     anything else                     -> ERR <why>
 *     a line of LINE_MAX (64) bytes or more is discarded WHOLE: nothing of it
 *     runs, and it gets exactly one reply, ERR line too long, at its newline
 *     (one reply per line is what keeps the host's reply stream in step)
 *
 * Safety behaviour, measured to matter:
 *   - Servos stay DETACHED until the first valid P/T frame. Attaching at boot
 *     drives every servo to 1500 us at full speed; on this arm that swings the
 *     forearm through the table.
 *   - Every pulse is clamped to per-joint MIN_US/MAX_US. Record each joint's
 *     mechanical end-stop in smoke test S5 and tighten these BEFORE mounting
 *     linkages -- a servo pushed past the end-stop stalls at 1.4-2.5 A.
 *     jetson/robot_server.py mirrors these tables (SKETCH_ARM_PULSE_US /
 *     SKETCH_GRIPPER_PULSE_US) and refuses a calibration that reaches past
 *     them, because clampUs() is silent and still answers OK. Change both.
 *   - WATCHDOG_MS without any frame detaches everything. A stalled Jetson
 *     must not leave a stalled servo cooking. The Jetson feeds the watchdog
 *     with K frames while idle (every 150 ms); P/T frames feed it too, and so
 *     do '?' and S: they prove the host process is alive just as well, and a
 *     '?' whose reply was lost used to hold the Jetson's serial lock for two
 *     reply timeouts without feeding anything -- one lost reply tripped the
 *     watchdog in 7 of 8 measured phases. None of them attaches anything.
 *     The Jetson ALSO stops feeding (and sends D) when no laptop request has
 *     arrived for host_timeout_s, so a crashed laptop cannot leave the arm
 *     energised and the jaw squeezing indefinitely.
 *   - A RE-ATTACH NEVER SNAPS TO THE NEW TARGET. After a detach (D, watchdog)
 *     current_us keeps the last pulsed position; the next P/T frame attaches
 *     THERE and slews / interpolates to its target at the normal rate. The
 *     old behaviour (current_us = target_us on attach) made every recovery
 *     after an estop a full-speed jump of all five servos. The one exception
 *     is the very first frame after power-up: no position is known, 1500 us
 *     is only the library default, so attaching there and slewing would
 *     itself be a snap. The first frame attaches AT its target, instantly --
 *     a T frame's <ms> has nothing to interpolate from -- which is why the
 *     run book says to hand-pose the arm to home before the first frame.
 *     Opening the serial port resets the Uno (DTR), so EVERY robot_server
 *     start is a power-up here. The Q reply reports have_position so the
 *     server refuses every motion but `home` until the first attach.
 *   - The loop never blocks: no delay(), no blocking reads. Serial bytes are
 *     accumulated one at a time so a slow sender cannot stall the tick.
 */

#include <Servo.h>

static const uint8_t  N_SERVOS        = 5;
static const uint8_t  PINS[N_SERVOS]  = {3, 5, 6, 9, 10};

// Per-joint pulse limits, microseconds. TIGHTEN AFTER SMOKE TEST S5.
static const int MIN_US[N_SERVOS] = { 600,  600,  600,  600,  900};
static const int MAX_US[N_SERVOS] = {2400, 2400, 2400, 2400, 2100};

static const unsigned long BAUD             = 115200;
static const unsigned long TICK_MS          = 20;    // servo frame period
static const int           SLEW_US_PER_TICK = 20;    // P-frame rate limit
static const unsigned long WATCHDOG_MS      = 500;
static const uint8_t       LINE_MAX         = 64;

Servo  servos[N_SERVOS];
int    current_us[N_SERVOS];   // what is being pulsed right now
int    target_us[N_SERVOS];    // where we are heading
long   step_us[N_SERVOS];      // T-frame: per-tick increment in 1/256 us
long   accum_us[N_SERVOS];     // T-frame: fixed-point position, 1/256 us
unsigned long ticks_left = 0;  // T-frame: ticks remaining in the move
bool   attached          = false;
bool   interpolating     = false;
bool   have_position     = false;  // false until the first attach: current_us is
                                   // the library default, not a known position

unsigned long last_tick_ms  = 0;
unsigned long last_frame_ms = 0;

char    line[LINE_MAX];
uint8_t line_len = 0;
bool    discarding = false;  // inside an overlong line: drop to its newline

// ---------------------------------------------------------------------------

static int clampUs(uint8_t i, long us) {
  if (us < MIN_US[i]) return MIN_US[i];
  if (us > MAX_US[i]) return MAX_US[i];
  return (int)us;
}

static void attachAll() {
  if (attached) return;
  for (uint8_t i = 0; i < N_SERVOS; ++i) {
    // writeMicroseconds before attach so the very first pulse is the target,
    // not the library's 1500 us default.
    servos[i].writeMicroseconds(current_us[i]);
    servos[i].attach(PINS[i], MIN_US[i], MAX_US[i]);
    servos[i].writeMicroseconds(current_us[i]);
  }
  attached      = true;
  have_position = true;
}

static void detachAll() {
  for (uint8_t i = 0; i < N_SERVOS; ++i) servos[i].detach();
  attached      = false;
  interpolating = false;
  ticks_left    = 0;
}

static void writeAll() {
  if (!attached) return;
  for (uint8_t i = 0; i < N_SERVOS; ++i) servos[i].writeMicroseconds(current_us[i]);
}

// Parse up to `want` comma-separated non-negative integers starting at *p.
// Returns the count parsed; *p is left after the last character consumed.
// A comma after the last wanted value is NOT consumed, so "P1,2,3,4,5," is
// a bad frame here exactly as in robot_server.decode_command_frame.
static uint8_t parseInts(const char **p, long *out, uint8_t want) {
  uint8_t n = 0;
  while (n < want) {
    const char *s = *p;
    if (*s < '0' || *s > '9') break;
    long v = 0;
    while (*s >= '0' && *s <= '9') { v = v * 10 + (*s - '0'); ++s; }
    out[n++] = v;
    *p = s;
    if (*s == ',' && n < want) { ++(*p); } else { break; }
  }
  return n;
}

static void replyQuery() {
  Serial.print('Q');
  for (uint8_t i = 0; i < N_SERVOS; ++i) {
    Serial.print(current_us[i]);
    Serial.print(',');
  }
  Serial.print(attached ? 1 : 0);
  Serial.print(',');
  Serial.println(have_position ? 1 : 0);
}

// A frame that proves the host is alive but must not attach anything.
static void feedWatchdogIfAttached() {
  if (attached) last_frame_ms = millis();
}

static void handleLine() {
  if (line_len == 0) return;
  const char cmd = line[0];
  const char *p  = line + 1;
  long vals[N_SERVOS + 1];

  if (cmd == '?') {
    feedWatchdogIfAttached();
    replyQuery();
    return;
  }

  if (cmd == 'S') {
    // Sync echo. The nonce is echoed verbatim so the host can skip every
    // late reply queued before it; digits only, like every other frame.
    const char *q = p;
    if (*q < '0' || *q > '9') {
      Serial.println(F("ERR bad frame"));
      return;
    }
    while (*q >= '0' && *q <= '9') ++q;
    if (*q != '\0') {
      Serial.println(F("ERR bad frame"));
      return;
    }
    feedWatchdogIfAttached();
    Serial.print('S');
    Serial.println(p);
    return;
  }

  if (cmd == 'D') {
    detachAll();
    Serial.println(F("OK"));
    return;
  }

  if (cmd == 'K') {
    // Keepalive. Only the watchdog timestamp moves: no attach, no target
    // change, and a T move in progress keeps interpolating.
    if (*p != '\0') {
      Serial.println(F("ERR bad frame"));
      return;
    }
    feedWatchdogIfAttached();
    Serial.println(F("OK"));
    return;
  }

  if (cmd == 'P' || cmd == 'T') {
    const uint8_t want = (cmd == 'T') ? N_SERVOS + 1 : N_SERVOS;
    const uint8_t got  = parseInts(&p, vals, want);
    if (got != want || *p != '\0') {
      Serial.println(F("ERR bad frame"));
      return;
    }
    for (uint8_t i = 0; i < N_SERVOS; ++i) target_us[i] = clampUs(i, vals[i]);

    if (!attached) {
      if (!have_position) {
        // Very first frame after power-up: nothing is known about where the
        // servos are, so attach AT the target (the run book has the arm
        // hand-posed to home, and home is what the Jetson sends first).
        for (uint8_t i = 0; i < N_SERVOS; ++i) current_us[i] = target_us[i];
      }
      // Otherwise current_us is the last pulsed position (kept across the
      // detach): attach there and let the slew / T interpolation below move
      // to the target at the normal rate. Never jump to the target.
      attachAll();
    }

    if (cmd == 'T') {
      long ms = vals[N_SERVOS];
      if (ms < 0) ms = 0;
      unsigned long ticks = (unsigned long)((ms + TICK_MS - 1) / TICK_MS);
      if (ticks == 0) ticks = 1;
      for (uint8_t i = 0; i < N_SERVOS; ++i) {
        accum_us[i] = (long)current_us[i] * 256L;
        step_us[i]  = ((long)(target_us[i] - current_us[i]) * 256L) / (long)ticks;
      }
      ticks_left    = ticks;
      interpolating = true;
    } else {
      interpolating = false;
      ticks_left    = 0;
    }
    last_frame_ms = millis();
    Serial.println(F("OK"));
    return;
  }

  Serial.print(F("ERR unknown command "));
  Serial.println(cmd);
}

static void tick() {
  if (!attached) return;
  if (interpolating) {
    if (ticks_left > 1) {
      for (uint8_t i = 0; i < N_SERVOS; ++i) {
        accum_us[i] += step_us[i];
        current_us[i] = clampUs(i, accum_us[i] / 256L);
      }
      --ticks_left;
    } else {
      for (uint8_t i = 0; i < N_SERVOS; ++i) current_us[i] = target_us[i];
      ticks_left    = 0;
      interpolating = false;
    }
  } else {
    for (uint8_t i = 0; i < N_SERVOS; ++i) {
      int d = target_us[i] - current_us[i];
      if (d >  SLEW_US_PER_TICK) d =  SLEW_US_PER_TICK;
      if (d < -SLEW_US_PER_TICK) d = -SLEW_US_PER_TICK;
      current_us[i] += d;
    }
  }
  writeAll();
}

// ---------------------------------------------------------------------------

void setup() {
  Serial.begin(BAUD);
  for (uint8_t i = 0; i < N_SERVOS; ++i) {
    current_us[i] = 1500;
    target_us[i]  = 1500;
    step_us[i]    = 0;
    accum_us[i]   = 0;
  }
  attached      = false;   // deliberately: nothing moves until told to
  last_tick_ms  = millis();
  last_frame_ms = millis();
}

void loop() {
  // 1. Drain serial one byte at a time; never block.
  while (Serial.available() > 0) {
    const char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (discarding) {
        // End of an overlong line: its ONE reply, and nothing of it is run.
        discarding = false;
        Serial.println(F("ERR line too long"));
      } else {
        line[line_len] = '\0';
        handleLine();
      }
      line_len = 0;
    } else if (discarding) {
      // Still inside an overlong line: drop bytes up to its newline.
    } else if (line_len < LINE_MAX - 1) {
      line[line_len++] = c;
    } else {
      // Overlong line: discard it WHOLE -- skip to its newline, act on no
      // fragment of it, and answer exactly once at that newline. The old
      // code reset the buffer here and then ran the tail as a frame, so one
      // bad line got two replies (the ERR plus the tail's) and the host's
      // reply stream shifted by one without noticing.
      discarding = true;
      line_len = 0;
    }
  }

  const unsigned long now = millis();

  // 2. Watchdog: silence from the host means the host is not in control.
  if (attached && (now - last_frame_ms) > WATCHDOG_MS) {
    detachAll();
  }

  // 3. Servo tick.
  if ((now - last_tick_ms) >= TICK_MS) {
    last_tick_ms = now;
    tick();
  }
}
