"""
calibrate.py — Klick-Kalibrierung für den Rocket League AI Bot

Anleitung:
  1. Rocket League im Fenster-Modus starten
  2. python calibrate.py
  3. Im Fenster auf das Objekt klicken das du kalibrieren willst
  4. Mehrmals klicken für bessere Genauigkeit (sammelt Pixel)
  5. Tasten zum Wechseln des Modus:
       1 = Ball           2 = Boostpad (groß)
       3 = Eigenes Tor    4 = Gegnerisches Tor
       5 = Gegner-Auto    r = Reset aktueller Modus
       s = Speichern      q = Beenden + config.py Werte ausgeben

Neu in dieser Version:
  - Live-Vorschau zeigt Spielfeld-ROI (grauer Rahmen)
  - Objekte außerhalb des ROI werden vom Bot ignoriert
  - ROI-Grenzen lassen sich mit W/A/S/D feinjustieren
"""

import sys
import cv2
import numpy as np
from collections import defaultdict

try:
    import mss
    _MSS = True
except ImportError:
    _MSS = False

try:
    from PIL import ImageGrab
    _PIL = True
except ImportError:
    _PIL = False

if not _MSS and not _PIL:
    print("FEHLER: pip install mss  oder  pip install Pillow")
    sys.exit(1)

# ── Konfiguration ─────────────────────────────────────────────────────────────

SAMPLE_RADIUS = 8
HSV_MARGIN    = [8, 40, 40]   # [H, S, V]

MODES = {
    ord("1"): "Ball",
    ord("2"): "Boostpad",
    ord("3"): "Eigenes Tor",
    ord("4"): "Gegnerisches Tor",
    ord("5"): "Gegner-Auto",
}

MODE_COLORS_BGR = {
    "Ball":             (0,   140, 255),
    "Boostpad":         (0,   220, 220),
    "Eigenes Tor":      (255, 100,   0),
    "Gegnerisches Tor": (0,    50, 220),
    "Gegner-Auto":      (200,   0, 200),
}

CONFIG_KEYS = {
    "Ball":             ("ball_hsv_lower",       "ball_hsv_upper"),
    "Boostpad":         ("boost_pad_hsv_lower",  "boost_pad_hsv_upper"),
    "Eigenes Tor":      ("goal_own_hsv_lower",   "goal_own_hsv_upper"),
    "Gegnerisches Tor": ("goal_enemy_hsv_lower", "goal_enemy_hsv_upper"),
    "Gegner-Auto":      ("enemy_hsv_lower",      "enemy_hsv_upper"),
}

# ROI-Startwerte (Anteil des Bildes, 0.0–1.0)
roi = {"top": 0.15, "bottom": 0.92, "left": 0.05, "right": 0.95}
ROI_STEP = 0.01   # Schrittweite bei W/A/S/D

# ── Globaler State ────────────────────────────────────────────────────────────

current_mode = "Ball"
samples: dict[str, list] = defaultdict(list)
ranges:  dict[str, tuple] = {}
last_frame: np.ndarray | None = None
DISPLAY_W, DISPLAY_H = 1280, 760
UI_TOP = 90


def grab_frame() -> np.ndarray:
    if _MSS:
        with mss.mss() as sct:
            mon = sct.monitors[1]
            img = sct.grab(mon)
            return np.array(img)[:, :, :3]
    img = ImageGrab.grab()
    return np.array(img)[:, :, ::-1]


def sample_pixels_around(frame_bgr: np.ndarray, x: int, y: int) -> list:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    fh, fw = hsv.shape[:2]
    pixels = []
    for dy in range(-SAMPLE_RADIUS, SAMPLE_RADIUS + 1):
        for dx in range(-SAMPLE_RADIUS, SAMPLE_RADIUS + 1):
            if dx*dx + dy*dy > SAMPLE_RADIUS**2:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < fw and 0 <= ny < fh:
                pixels.append(hsv[ny, nx].tolist())
    return pixels


def calc_range(pixel_list: list) -> tuple[list, list]:
    arr = np.array(pixel_list, dtype=np.float32)
    lo  = np.percentile(arr, 5,  axis=0).astype(int).tolist()
    hi  = np.percentile(arr, 95, axis=0).astype(int).tolist()
    lo  = [max(0,   lo[i] - HSV_MARGIN[i]) for i in range(3)]
    hi  = [min(255 if i > 0 else 179, hi[i] + HSV_MARGIN[i]) for i in range(3)]
    lo[0] = max(0,   lo[0])
    hi[0] = min(179, hi[0])
    return lo, hi


def draw_mask_overlay(frame_bgr: np.ndarray, lo: list, hi: list,
                      color_bgr: tuple) -> np.ndarray:
    hsv    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask   = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ROI-Maske anwenden (zeigt was der Bot auch sieht)
    h, w = mask.shape[:2]
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    y1 = int(h * roi["top"]);    y2 = int(h * roi["bottom"])
    x1 = int(w * roi["left"]);   x2 = int(w * roi["right"])
    roi_mask[y1:y2, x1:x2] = 255
    mask = cv2.bitwise_and(mask, roi_mask)

    result = frame_bgr.copy()
    hl     = np.zeros_like(frame_bgr)
    hl[mask > 0] = color_bgr
    result = cv2.addWeighted(result, 0.55, hl, 0.45, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 15:
            continue
        cv2.drawContours(result, [c], -1, color_bgr, 2)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(result, (cx, cy), 5, (255, 255, 255), -1)
    return result


def draw_roi_overlay(frame: np.ndarray) -> np.ndarray:
    """Zeichnet den aktiven ROI-Rahmen auf das (bereits skalierte) Bild."""
    h, w = frame.shape[:2]
    x1 = int(w * roi["left"])
    x2 = int(w * roi["right"])
    y1 = int(h * roi["top"])
    y2 = int(h * roi["bottom"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 200, 80), 1)
    cv2.putText(frame, "Bot-Suchbereich (ROI)",
                (x1 + 4, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 200, 80), 1)
    return frame


def draw_ui(frame: np.ndarray, click_marks: list) -> np.ndarray:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, UI_TOP), (20, 20, 20), -1)
    col = MODE_COLORS_BGR.get(current_mode, (255, 255, 255))
    cv2.putText(frame, f"Modus: {current_mode}  — Klicke auf das Objekt",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2)
    cv2.putText(frame,
                "1=Ball  2=Boost  3=EigenTor  4=GegnerTor  5=GegnerAuto  "
                "r=Reset  q=Beenden",
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
    cv2.putText(frame,
                "ROI anpassen: W=oben hoch  S=oben runter  A=links  D=rechts  "
                "I=unten hoch  K=unten runter",
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 180, 100), 1)
    cv2.putText(frame,
                "Mehrmals klicken = genauer  |  Grüner Rahmen = Bot-Suchbereich",
                (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)

    # Rechtes Panel
    px = w - 300
    cv2.rectangle(frame, (px - 8, UI_TOP), (w, h), (12, 12, 12), -1)
    y = UI_TOP + 22
    cv2.putText(frame, "Status:", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y += 8
    for name in CONFIG_KEYS:
        y += 22
        c = MODE_COLORS_BGR.get(name, (100, 100, 100))
        if name in ranges:
            lo, hi = ranges[name]
            cv2.putText(frame, f"[OK] {name}", (px, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, c, 1)
        else:
            n = len(samples.get(name, []))
            lbl = f"{n}px" if n else "—"
            cv2.putText(frame, f"[ ] {name}  {lbl}", (px, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, (100, 100, 100), 1)

    if current_mode in ranges:
        lo, hi = ranges[current_mode]
        y += 28
        cv2.putText(frame, f"lo={lo}", (px, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1)
        y += 20
        cv2.putText(frame, f"hi={hi}", (px, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1)
        n_clicks = len(samples.get(current_mode, []))
        y += 20
        cv2.putText(frame, f"{n_clicks} Pixel gesammelt", (px, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

    # ROI-Werte anzeigen
    y += 32
    cv2.putText(frame, "ROI:", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (80, 200, 80), 1)
    y += 18
    cv2.putText(frame, f"  top={roi['top']:.2f}  bot={roi['bottom']:.2f}", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 200, 80), 1)
    y += 16
    cv2.putText(frame, f"  left={roi['left']:.2f}  right={roi['right']:.2f}", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 200, 80), 1)

    col = MODE_COLORS_BGR.get(current_mode, (255, 255, 255))
    for (mx, my) in click_marks:
        cv2.circle(frame, (mx, my + UI_TOP), SAMPLE_RADIUS, col, 2)
        cv2.circle(frame, (mx, my + UI_TOP), 2,             col, -1)

    return frame


click_marks: list[tuple[int, int]] = []


def on_click(event, x, y, flags, param):
    global last_frame
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if last_frame is None or y < UI_TOP:
        return

    fh, fw = last_frame.shape[:2]
    img_display_h = DISPLAY_H - UI_TOP
    real_x = int(x / DISPLAY_W * fw)
    real_y = int((y - UI_TOP) / img_display_h * fh)
    real_x = max(0, min(fw - 1, real_x))
    real_y = max(0, min(fh - 1, real_y))

    new_pixels = sample_pixels_around(last_frame, real_x, real_y)
    samples[current_mode].extend(new_pixels)
    click_marks.append((x, y - UI_TOP))

    if len(samples[current_mode]) >= 10:
        lo, hi = calc_range(samples[current_mode])
        ranges[current_mode] = (lo, hi)
        n = len(samples[current_mode])
        print(f"  [{current_mode}] ({real_x},{real_y}) → lo={lo}  hi={hi}  [{n} Pixel]")
    else:
        left = 10 - len(samples[current_mode])
        print(f"  [{current_mode}] ({real_x},{real_y}) — noch {left} Pixel bis erste Berechnung")


def print_config():
    print("\n" + "=" * 65)
    print("Kalibrierte HSV-Werte — kopiere das in deine DetectionConfig:")
    print("=" * 65)
    for name, (lo_key, hi_key) in CONFIG_KEYS.items():
        if name in ranges:
            lo, hi = ranges[name]
            print(f"\n  # {name}")
            print(f"  {lo_key}: list = field(default_factory=lambda: {lo})")
            print(f"  {hi_key}: list = field(default_factory=lambda: {hi})")
        else:
            print(f"\n  # {name}: nicht kalibriert (Standardwert bleibt)")
    print("\n  # ROI-Werte")
    print(f"  roi_top:    float = {roi['top']}")
    print(f"  roi_bottom: float = {roi['bottom']}")
    print(f"  roi_left:   float = {roi['left']}")
    print(f"  roi_right:  float = {roi['right']}")
    print("\n" + "─" * 65)
    print("Dann: python main.py")
    print("=" * 65 + "\n")


def main():
    global current_mode, last_frame, click_marks

    WIN = "RL Kalibrierung — Klicke auf Objekte"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DISPLAY_W, DISPLAY_H)
    cv2.setMouseCallback(WIN, on_click)

    print("\nKalibrierungs-Tool gestartet.")
    print("Klicke 3–5x auf ein Objekt → HSV-Bereich wird automatisch berechnet.")
    print("1=Ball  2=Boost  3=EigenTor  4=GegnerTor  5=GegnerAuto")
    print("ROI: W/S = obere Grenze | I/K = untere Grenze | A/D = Seiten")
    print("r=Reset  q=Ende\n")

    while True:
        raw = grab_frame()
        last_frame = raw.copy()

        if current_mode in ranges:
            lo, hi = ranges[current_mode]
            col    = MODE_COLORS_BGR.get(current_mode, (0, 255, 0))
            vis    = draw_mask_overlay(raw, lo, hi, col)
        else:
            vis = raw.copy()

        img_h = DISPLAY_H - UI_TOP
        vis_resized = cv2.resize(vis, (DISPLAY_W, img_h))

        # ROI-Rahmen auf das skalierte Bild zeichnen
        draw_roi_overlay(vis_resized)

        canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
        canvas[UI_TOP:, :] = vis_resized
        canvas = draw_ui(canvas, click_marks)

        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r"):
            samples[current_mode].clear()
            click_marks.clear()
            if current_mode in ranges:
                del ranges[current_mode]
            print(f"  [{current_mode}] zurückgesetzt.")

        # ROI-Anpassung
        if key == ord("w"):
            roi["top"] = max(0.0, roi["top"] - ROI_STEP)
        elif key == ord("s"):
            roi["top"] = min(roi["bottom"] - 0.05, roi["top"] + ROI_STEP)
        elif key == ord("i"):
            roi["bottom"] = max(roi["top"] + 0.05, roi["bottom"] - ROI_STEP)
        elif key == ord("k"):
            roi["bottom"] = min(1.0, roi["bottom"] + ROI_STEP)
        elif key == ord("a"):
            roi["left"] = max(0.0, roi["left"] - ROI_STEP)
        elif key == ord("d"):
            roi["left"] = min(roi["right"] - 0.05, roi["left"] + ROI_STEP)

        new_mode = MODES.get(key)
        if new_mode and new_mode != current_mode:
            current_mode = new_mode
            click_marks.clear()
            print(f"\n  Modus: {current_mode}  — jetzt auf {current_mode} klicken")

    cv2.destroyAllWindows()
    if ranges:
        print_config()
    else:
        print("Keine Werte kalibriert.")


if __name__ == "__main__":
    main()
