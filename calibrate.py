"""
calibrate.py — HSV-Kalibrierungs-Tool für den Rocket League AI Bot

Starte dieses Skript WÄHREND Rocket League im Fenster läuft:
    python calibrate.py

Steuerung:
    1 = Ball-Maske anzeigen       (orange)
    2 = Boostpad-Maske anzeigen   (gelb)
    3 = Eigenes Tor               (blau)
    4 = Gegnerisches Tor          (rot)
    5 = Gegner-Autos              (lila)
    6 = Hough-Kreise (Ball)
    0 = Original-Frame
    q = Beenden + Werte ausgeben

Schieberegler:
    H_lo / H_hi  = Farbton (0–179)
    S_lo / S_hi  = Sättigung (0–255)
    V_lo / V_hi  = Helligkeit (0–255)

Wenn du eine gute Maske siehst (Ball leuchtet weiß, Rest schwarz)
→ notiere die Werte und trage sie in config.py ein.
"""

import sys
import cv2
import numpy as np

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

# ── Standard-Startwerte (aus config.py) ──────────────────────────────────────
PRESETS = {
    "1_ball":        ([5,  150, 150], [20, 255, 255]),
    "2_boost_pad":   ([20, 100, 150], [35, 255, 255]),
    "3_own_goal":    ([100, 80,  80], [130, 255, 255]),
    "4_enemy_goal":  ([0,   80,  80], [10,  255, 255]),
    "5_enemy_car":   ([140, 80,  80], [170, 255, 255]),
}

MODE_NAMES = {
    ord("1"): "1_ball",
    ord("2"): "2_boost_pad",
    ord("3"): "3_own_goal",
    ord("4"): "4_enemy_goal",
    ord("5"): "5_enemy_car",
    ord("6"): "6_hough",
    ord("0"): "0_original",
}

WINDOW = "RL Kalibrierung  |  1=Ball 2=Boost 3=EigenTor 4=GegnerTor 5=Gegner 6=Hough 0=Original q=Beenden"


def grab_frame() -> np.ndarray:
    if _MSS:
        with mss.mss() as sct:
            mon = sct.monitors[1]
            img = sct.grab(mon)
            return np.array(img)[:, :, :3]
    img = ImageGrab.grab()
    return np.array(img)[:, :, ::-1]


def nothing(_): pass


def create_trackbars(win: str, lo: list, hi: list):
    cv2.createTrackbar("H_lo", win, lo[0], 179, nothing)
    cv2.createTrackbar("H_hi", win, hi[0], 179, nothing)
    cv2.createTrackbar("S_lo", win, lo[1], 255, nothing)
    cv2.createTrackbar("S_hi", win, hi[1], 255, nothing)
    cv2.createTrackbar("V_lo", win, lo[2], 255, nothing)
    cv2.createTrackbar("V_hi", win, hi[2], 255, nothing)


def get_trackbar_vals(win: str) -> tuple[list, list]:
    lo = [
        cv2.getTrackbarPos("H_lo", win),
        cv2.getTrackbarPos("S_lo", win),
        cv2.getTrackbarPos("V_lo", win),
    ]
    hi = [
        cv2.getTrackbarPos("H_hi", win),
        cv2.getTrackbarPos("S_hi", win),
        cv2.getTrackbarPos("V_hi", win),
    ]
    return lo, hi


def set_trackbars(win: str, lo: list, hi: list):
    cv2.setTrackbarPos("H_lo", win, lo[0])
    cv2.setTrackbarPos("H_hi", win, hi[0])
    cv2.setTrackbarPos("S_lo", win, lo[1])
    cv2.setTrackbarPos("S_hi", win, hi[1])
    cv2.setTrackbarPos("V_lo", win, lo[2])
    cv2.setTrackbarPos("V_hi", win, hi[2])


def apply_mask(frame_bgr: np.ndarray, lo: list, hi: list) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Konturen zeichnen
    result = frame_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 20:
            continue
        cv2.drawContours(result, [c], -1, (0, 255, 0), 2)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(result, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(result, f"{area:.0f}px²", (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    # Maske als Overlay (links = Maske, rechts = Frame mit Konturen)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([
        cv2.resize(mask_bgr, (640, 360)),
        cv2.resize(result,   (640, 360)),
    ])
    return combined


def apply_hough(frame_bgr: np.ndarray) -> np.ndarray:
    """Zeigt Hough-Kreise auf dem Frame."""
    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=30,
        param1=60, param2=30,
        minRadius=4, maxRadius=60,
    )
    result = frame_bgr.copy()
    if circles is not None:
        for (x, y, r) in np.round(circles[0]).astype(int):
            cv2.circle(result, (x, y), r,    (0, 255,   0), 2)
            cv2.circle(result, (x, y), 2,    (0,   0, 255), 3)
            cv2.putText(result, f"r={r}", (x + r + 3, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    label = cv2.resize(result, (1280, 360))
    return label


def add_overlay(img: np.ndarray, mode_name: str, lo: list, hi: list) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, f"Modus: {mode_name}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(out, f"HSV_lo={lo}  HSV_hi={hi}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(out, "Gruen=erkannte Objekte | Links=Maske | Rechts=Frame", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return out


def print_config(results: dict):
    print("\n" + "=" * 60)
    print("Kalibrierte HSV-Werte — in config.py eintragen:")
    print("=" * 60)
    mapping = {
        "1_ball":       ("ball_hsv_lower",       "ball_hsv_upper"),
        "2_boost_pad":  ("boost_pad_hsv_lower",  "boost_pad_hsv_upper"),
        "3_own_goal":   ("goal_own_hsv_lower",   "goal_own_hsv_upper"),
        "4_enemy_goal": ("goal_enemy_hsv_lower", "goal_enemy_hsv_upper"),
        "5_enemy_car":  ("enemy_hsv_lower",      "enemy_hsv_upper"),
    }
    for key, (lo_name, hi_name) in mapping.items():
        if key in results:
            lo, hi = results[key]
            print(f"  {lo_name}: list = field(default_factory=lambda: {lo})")
            print(f"  {hi_name}: list = field(default_factory=lambda: {hi})")
    print("=" * 60)
    print("\nDanach in config.py: dummy_mode = False")
    print("Dann: python main.py\n")


def main():
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 500)

    # Starte mit Ball-Preset
    current_mode = "1_ball"
    lo, hi = PRESETS[current_mode]
    create_trackbars(WINDOW, lo, hi)

    saved: dict = {}
    print("\nKalibrierungs-Tool gestartet.")
    print("Drücke 1–6 um Modus zu wechseln, q zum Beenden.\n")

    while True:
        frame = grab_frame()
        frame = cv2.resize(frame, (1280, 720))

        lo, hi = get_trackbar_vals(WINDOW)

        if current_mode == "0_original":
            display = cv2.resize(frame, (1280, 360))
        elif current_mode == "6_hough":
            display = apply_hough(cv2.resize(frame, (640, 360)))
        else:
            display = apply_mask(cv2.resize(frame, (640, 360)), lo, hi)

        display = add_overlay(display, current_mode, lo, hi)
        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            # Aktuellen Modus speichern
            if current_mode not in ("0_original", "6_hough"):
                saved[current_mode] = (lo[:], hi[:])
            break

        if key == ord("s"):
            # Aktuellen Modus manuell speichern
            if current_mode not in ("0_original", "6_hough"):
                saved[current_mode] = (lo[:], hi[:])
                print(f"  Gespeichert: {current_mode} → lo={lo} hi={hi}")

        new_mode = MODE_NAMES.get(key)
        if new_mode and new_mode != current_mode:
            # Alten Modus speichern
            if current_mode not in ("0_original", "6_hough"):
                saved[current_mode] = (lo[:], hi[:])
                print(f"  Gespeichert: {current_mode} → lo={lo} hi={hi}")
            # Neuen Modus laden
            current_mode = new_mode
            if new_mode in PRESETS:
                lo, hi = PRESETS[new_mode]
                set_trackbars(WINDOW, lo, hi)
            print(f"  Modus gewechselt: {new_mode}")

    cv2.destroyAllWindows()
    if saved:
        print_config(saved)
    else:
        print("Keine Werte gespeichert (s=speichern, dann q=beenden).")


if __name__ == "__main__":
    main()
