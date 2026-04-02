"""
calibrate_boost.py — Boost-Anzeige ROI kalibrieren

Starte während Rocket League läuft:
    python calibrate_boost.py

Steuerung:
    Maus ziehen = ROI aufziehen
    s           = Speichern + Werte ausgeben
    r           = Reset
    q           = Beenden
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

try:
    import mss
    def grab():
        with mss.mss() as sct:
            img = sct.grab(sct.monitors[1])
            return np.array(img)[:, :, :3]
except ImportError:
    from PIL import ImageGrab
    def grab():
        return np.array(ImageGrab.grab())[:, :, ::-1]

WIN = "Boost ROI Kalibrierung — Ziehe ein Rechteck um die Boost-Anzeige | s=Speichern q=Beenden"
DISPLAY_W, DISPLAY_H = 1280, 720

# Zustand
drawing   = False
start_x   = start_y = end_x = end_y = 0
roi_final = None
base_frame: np.ndarray | None = None


def on_mouse(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y, roi_final
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        end_x,   end_y   = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_x, end_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing   = False
        end_x, end_y = x, y
        roi_final = (min(start_x, end_x), min(start_y, end_y),
                     max(start_x, end_x), max(start_y, end_y))


def px_to_norm(x1, y1, x2, y2, fw, fh):
    """Pixel-Koordinaten → normierte config.py-Werte (x, y, w, h)."""
    nx  = x1 / fw
    ny  = y1 / fh
    nw  = (x2 - x1) / fw
    nh  = (y2 - y1) / fh
    return round(nx, 3), round(ny, 3), round(nw, 3), round(nh, 3)


def read_boost_from_roi(frame, x1, y1, x2, y2, threshold=180, scale=4.0):
    roi  = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ratio = float(np.mean(gray > threshold))
    return round(min(ratio * scale * 100, 100.0), 1)


def main():
    global base_frame, roi_final

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DISPLAY_W, DISPLAY_H)
    cv2.setMouseCallback(WIN, on_mouse)

    # Schwellwert-Schieberegler
    cv2.createTrackbar("Threshold", WIN, 180, 255, lambda x: None)
    cv2.createTrackbar("Scale x10",  WIN,  40, 100, lambda x: None)

    print("\nBoost-ROI Kalibrierung")
    print("Ziehe ein Rechteck um die Boost-Anzeige im Spiel.")
    print("Der erkannte Boost-Wert wird live angezeigt.\n")

    while True:
        raw = grab()
        fh, fw = raw.shape[:2]
        base_frame = raw.copy()

        threshold = cv2.getTrackbarPos("Threshold", WIN)
        scale     = cv2.getTrackbarPos("Scale x10",  WIN) / 10.0

        # Frame auf Display-Größe skalieren
        disp = cv2.resize(raw, (DISPLAY_W, DISPLAY_H))
        sx   = DISPLAY_W / fw
        sy   = DISPLAY_H / fh

        # Aktuelles Rechteck zeichnen
        if drawing or roi_final:
            rx1, ry1 = (start_x, start_y) if drawing else (roi_final[0], roi_final[1])
            rx2, ry2 = (end_x,   end_y)   if drawing else (roi_final[2], roi_final[3])
            cv2.rectangle(disp, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

            if roi_final and not drawing:
                # Boost live berechnen
                # Pixel-Koordinaten zurück auf Original-Frame
                ox1 = int(roi_final[0] / sx)
                oy1 = int(roi_final[1] / sy)
                ox2 = int(roi_final[2] / sx)
                oy2 = int(roi_final[3] / sy)
                boost_val = read_boost_from_roi(raw, ox1, oy1, ox2, oy2,
                                                threshold, scale)

                # Boost-Wert groß anzeigen
                cv2.putText(disp, f"Boost: {boost_val:.0f}%",
                            (roi_final[0], roi_final[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                # ROI-Vorschau (Ausschnitt vergrößert)
                roi_crop = raw[oy1:oy2, ox1:ox2]
                if roi_crop.size > 0:
                    roi_big = cv2.resize(roi_crop, (200, 60))
                    gray_big = cv2.cvtColor(roi_big, cv2.COLOR_BGR2GRAY)
                    _, thresh_big = cv2.threshold(gray_big, threshold, 255, cv2.THRESH_BINARY)
                    thresh_bgr = cv2.cvtColor(thresh_big, cv2.COLOR_GRAY2BGR)
                    # Nebeneinander anzeigen
                    preview = np.hstack([roi_big, thresh_bgr])
                    ph, pw = preview.shape[:2]
                    disp[10:10+ph, DISPLAY_W-pw-10:DISPLAY_W-10] = preview
                    cv2.putText(disp, "ROI original / Schwellwert",
                                (DISPLAY_W-pw-10, 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Anleitung
        cv2.rectangle(disp, (0, 0), (600, 50), (20, 20, 20), -1)
        cv2.putText(disp, "Ziehe Rechteck um Boost-Anzeige  |  s=Speichern  r=Reset  q=Beenden",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(disp, "Ziel: Boost-% passt zum was du im Spiel siehst",
                    (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        cv2.imshow(WIN, disp)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r"):
            roi_final = None
            print("  Reset.")

        if key == ord("s") and roi_final:
            ox1 = int(roi_final[0] / sx)
            oy1 = int(roi_final[1] / sy)
            ox2 = int(roi_final[2] / sx)
            oy2 = int(roi_final[3] / sy)
            nx, ny, nw, nh = px_to_norm(ox1, oy1, ox2, oy2, fw, fh)
            boost_val = read_boost_from_roi(raw, ox1, oy1, ox2, oy2, threshold, scale)

            print("\n" + "="*55)
            print("Boost ROI kalibriert — in config.py eintragen:")
            print("="*55)
            print(f"  boost_roi:                  tuple = ({nx}, {ny}, {nw}, {nh})")
            print(f"  boost_brightness_threshold: int   = {threshold}")
            print(f"  boost_scale_factor:         float = {scale}")
            print(f"\n  Aktuell erkannter Boost: {boost_val:.0f}%")
            print("="*55 + "\n")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
