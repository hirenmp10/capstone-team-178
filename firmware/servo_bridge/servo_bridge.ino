/*
 * servo_bridge.ino — USB-CDC serial bridge for the hardware lane (Stage B)
 *
 * Target: Arduino Uno R3 (ATmega328P)
 * Library: built-in Servo (wraps AVR timer, 1 per 12 channels)
 *
 * ============================================================
 *  WIRING RULES (READ BEFORE POWERING)
 * ============================================================
 * - Signal wires: connect servo signal pins to the SERVO_PINS below.
 * - Servo power:  ALL servos draw from a SEPARATE 5–6 V supply (>= 3 A
 *   for five MG90S at stall).  Do NOT power servos from the Uno's 5 V pin
 *   or from USB — USB current is 500 mA max, stall is ~700 mA per servo.
 * - Common ground: tie the external supply GND and the Uno GND together.
 * - E-stop switch: place a normally-closed switch in the +V servo supply
 *   line.  Opening it cuts servo power without touching the MCU.
 * - First power-on: support the arm by hand before the first P/T command.
 *   Keep fingers clear of the elbow joint.
 *
 * ============================================================
 *  SERIAL PROTOCOL  (115200 8N1, ASCII lines, '\n' terminated)
 * ============================================================
 * Commands (host -> Uno):
 *   P p0 p1 ... p{N-1}\n   set targets immediately (ms = 0)
 *   T ms p0 ... p{N-1}\n   interpolate to targets over ms ms
 *   D\n                     detach all servos (E-stop)
 *   W ms\n                  set watchdog timeout (0=off, else 50..10000)
 *   S\n                     status query (also resets watchdog)
 *   V\n                     firmware version query
 *
 * Replies (Uno -> host):
 *   OK\n                    command accepted
 *   ERR line\n              line too long (> MAX_LINE)
 *   ERR argc\n              wrong number of arguments
 *   ERR range\n             pulse out of [PULSE_MIN_US, PULSE_MAX_US]; no change applied
 *   ERR cmd\n               unknown command
 *   S <att> <mov> <wd_ms> p0 ... p{N-1}\n   status reply
 *   V servo_bridge 1 <N_CH>\n               version reply
 *
 * ============================================================
 */

#include <Servo.h>

/* ------------------------------------------------------------------ */
/* EDIT TO YOUR BUILD                                                   */
/* ------------------------------------------------------------------ */
static const uint8_t  N_CH              = 5;
static const uint8_t  SERVO_PINS[N_CH]  = {3, 5, 6, 9, 10};  // base_yaw,shoulder,elbow,wrist,jaw
static const uint16_t PULSE_MIN_US      = 500;   // absolute hardware floor (calibration clamps tighter)
static const uint16_t PULSE_MAX_US      = 2500;  // absolute hardware ceiling
static const uint32_t DEFAULT_WATCHDOG_MS = 500; // detach after this many ms without a P/T/S frame
static const uint32_t TICK_MS           = 20;    // interpolation step, ms
static const uint32_t BAUD              = 115200;
/* ------------------------------------------------------------------ */

static const uint8_t  MAX_LINE          = 96;    // bytes; longer lines -> ERR line

/* ---- servo state ------------------------------------------------- */
Servo      servos[N_CH];
uint16_t   current_us[N_CH];   // last commanded pulse
uint16_t   target_us[N_CH];    // interpolation target
bool       attached  = false;
bool       moving    = false;

/* ---- interpolation ----------------------------------------------- */
uint32_t   move_start_ms  = 0;
uint32_t   move_dur_ms    = 0;
uint16_t   move_start_us[N_CH];

/* ---- watchdog ----------------------------------------------------- */
uint32_t   wd_ms       = DEFAULT_WATCHDOG_MS;
uint32_t   last_frame_ms = 0;  // millis() of last P, T, or S frame
bool       wd_tripped  = false;

/* ---- line buffer -------------------------------------------------- */
char       line_buf[MAX_LINE + 2];
uint8_t    line_len    = 0;
bool       line_overflow = false;

/* ================================================================== */
/* helpers                                                              */
/* ================================================================== */

static void do_attach() {
    if (!attached) {
        for (uint8_t i = 0; i < N_CH; i++) {
            servos[i].attach(SERVO_PINS[i]);
            servos[i].writeMicroseconds(current_us[i]);
        }
        attached = true;
    }
}

static void do_detach() {
    for (uint8_t i = 0; i < N_CH; i++) {
        servos[i].detach();
    }
    attached = false;
    moving   = false;
}

static uint16_t clamp_pulse(uint16_t p) {
    if (p < PULSE_MIN_US) return PULSE_MIN_US;
    if (p > PULSE_MAX_US) return PULSE_MAX_US;
    return p;
}

/* Parse up to N_CH pulse values from a token array starting at offset.
   Returns true on success, false if wrong count or out-of-range.
   On success writes into out[]; on failure writes nothing.            */
static bool parse_pulses(char **tokens, int start, int count, uint16_t out[]) {
    if (count != N_CH) return false;
    uint16_t tmp[N_CH];
    for (int i = 0; i < count; i++) {
        long v = atol(tokens[start + i]);
        if (v < PULSE_MIN_US || v > PULSE_MAX_US) return false;
        tmp[i] = (uint16_t)v;
    }
    for (int i = 0; i < count; i++) out[i] = tmp[i];
    return true;
}

/* Tokenize buf in-place; returns token count. */
static int tokenize(char *buf, char **toks, int max_toks) {
    int n = 0;
    char *p = buf;
    while (*p && n < max_toks) {
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;
        toks[n++] = p;
        while (*p && *p != ' ' && *p != '\t') p++;
        if (*p) { *p = '\0'; p++; }
    }
    return n;
}

/* ================================================================== */
/* command handler                                                      */
/* ================================================================== */

static void handle_line(char *buf) {
    char *toks[N_CH + 4];
    int   n = tokenize(buf, toks, N_CH + 4);
    if (n == 0) return;  // blank line: ignore, no reply

    char cmd = toks[0][0];

    if (cmd == 'V' && n == 1) {
        /* V — version */
        Serial.print("V servo_bridge 1 ");
        Serial.print(N_CH);
        Serial.print('\n');
        return;
    }

    if (cmd == 'S' && n == 1) {
        /* S — status (counts as keepalive) */
        last_frame_ms = millis();
        wd_tripped    = false;
        Serial.print("S ");
        Serial.print(attached ? 1 : 0);
        Serial.print(' ');
        Serial.print(moving   ? 1 : 0);
        Serial.print(' ');
        Serial.print(wd_ms);
        for (uint8_t i = 0; i < N_CH; i++) {
            Serial.print(' ');
            Serial.print(current_us[i]);
        }
        Serial.print('\n');
        return;
    }

    if (cmd == 'D' && n == 1) {
        /* D — detach (E-stop) */
        do_detach();
        Serial.print("OK\n");
        return;
    }

    if (cmd == 'W' && n == 2) {
        /* W ms — set watchdog */
        long ms = atol(toks[1]);
        if (ms != 0 && (ms < 50 || ms > 10000)) {
            Serial.print("ERR range\n");
            return;
        }
        wd_ms = (uint32_t)ms;
        Serial.print("OK\n");
        return;
    }

    if (cmd == 'P' && n == 1 + N_CH) {
        /* P p0 ... pN-1 — set immediately */
        uint16_t new_us[N_CH];
        if (!parse_pulses(toks, 1, N_CH, new_us)) {
            Serial.print("ERR range\n");
            return;
        }
        last_frame_ms = millis();
        wd_tripped    = false;
        for (uint8_t i = 0; i < N_CH; i++) {
            current_us[i] = new_us[i];
            target_us[i]  = new_us[i];
        }
        do_attach();
        for (uint8_t i = 0; i < N_CH; i++) {
            servos[i].writeMicroseconds(current_us[i]);
        }
        moving = false;
        Serial.print("OK\n");
        return;
    }

    if (cmd == 'T' && n == 2 + N_CH) {
        /* T ms p0 ... pN-1 — interpolate */
        long ms = atol(toks[1]);
        if (ms < 0 || ms > 30000) {
            Serial.print("ERR range\n");
            return;
        }
        uint16_t new_us[N_CH];
        if (!parse_pulses(toks, 2, N_CH, new_us)) {
            Serial.print("ERR range\n");
            return;
        }
        last_frame_ms = millis();
        wd_tripped    = false;
        for (uint8_t i = 0; i < N_CH; i++) {
            move_start_us[i] = current_us[i];
            target_us[i]     = new_us[i];
        }
        do_attach();
        if (ms == 0) {
            /* T 0 is just P */
            for (uint8_t i = 0; i < N_CH; i++) {
                current_us[i] = target_us[i];
                servos[i].writeMicroseconds(current_us[i]);
            }
            moving = false;
        } else {
            move_start_ms = millis();
            move_dur_ms   = (uint32_t)ms;
            moving        = true;
        }
        Serial.print("OK\n");
        return;
    }

    /* Wrong arg count for a known command? */
    if (cmd == 'P' || cmd == 'T' || cmd == 'W') {
        Serial.print("ERR argc\n");
        return;
    }

    Serial.print("ERR cmd\n");
}

/* ================================================================== */
/* Arduino entry points                                                 */
/* ================================================================== */

void setup() {
    Serial.begin(BAUD);

    /* Detached at boot — nothing moves until first P or T. */
    for (uint8_t i = 0; i < N_CH; i++) {
        current_us[i]    = (PULSE_MIN_US + PULSE_MAX_US) / 2;  // midpoint = safe default
        target_us[i]     = current_us[i];
        move_start_us[i] = current_us[i];
    }
    last_frame_ms = millis();
}

void loop() {
    uint32_t now = millis();

    /* ---- 1. Read serial into line buffer (non-blocking) ---------- */
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n') {
            if (line_overflow) {
                Serial.print("ERR line\n");
            } else {
                line_buf[line_len] = '\0';
                handle_line(line_buf);
            }
            line_len      = 0;
            line_overflow = false;
        } else if (c != '\r') {
            if (line_len < MAX_LINE) {
                line_buf[line_len++] = c;
            } else {
                line_overflow = true;
            }
        }
    }

    /* ---- 2. Interpolation tick ----------------------------------- */
    if (moving) {
        uint32_t elapsed = now - move_start_ms;
        if (elapsed >= move_dur_ms) {
            /* Snap to final */
            for (uint8_t i = 0; i < N_CH; i++) {
                current_us[i] = target_us[i];
                servos[i].writeMicroseconds(current_us[i]);
            }
            moving = false;
        } else {
            /* Linear interpolation per channel */
            for (uint8_t i = 0; i < N_CH; i++) {
                int32_t delta = (int32_t)target_us[i] - (int32_t)move_start_us[i];
                uint16_t interp = (uint16_t)((int32_t)move_start_us[i] +
                                             delta * (int32_t)elapsed / (int32_t)move_dur_ms);
                current_us[i] = clamp_pulse(interp);
                servos[i].writeMicroseconds(current_us[i]);
            }
        }
    }

    /* ---- 3. Watchdog --------------------------------------------- */
    if (attached && wd_ms > 0) {
        if ((now - last_frame_ms) >= wd_ms) {
            do_detach();
            wd_tripped = true;
        }
    }
}
