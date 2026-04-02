# detector.py — Robuste Objekterkennung mit Multi-Filter-Pipeline
#
# Strategie:
#   1. HSV-Farbfilter       → Kandidaten herausfiltern
#   2. ROI-Maske            → UI-Bereiche (Boost, Timer, Score) ausblenden
#   3. Tracker-ROI          → nur um letzte bekannte Position suchen
#   4. Form + Größe         → Kreisförmigkeit, Min/Max-Radius
#   5. Konfidenz-Scoring    → bestes Objekt wählen

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Konfiguration ─────────────────────────────────────────────────────────────

@dataclass
class DetectionConfig:
    # HSV-Bereiche — aus calibrate.py kalibrieren und hier eintragen
    ball_hsv_lower:        list = field(default_factory=lambda: [0,   0,   180])
    ball_hsv_upper:        list = field(default_factory=lambda: [179, 40,  255])

    boost_pad_hsv_lower:   list = field(default_factory=lambda: [15,  120, 150])
    boost_pad_hsv_upper:   list = field(default_factory=lambda: [35,  255, 255])

    goal_own_hsv_lower:    list = field(default_factory=lambda: [100, 80,  80])
    goal_own_hsv_upper:    list = field(default_factory=lambda: [130, 255, 255])

    goal_enemy_hsv_lower:  list = field(default_factory=lambda: [0,   80,  80])
    goal_enemy_hsv_upper:  list = field(default_factory=lambda: [10,  255, 255])

    enemy_hsv_lower:       list = field(default_factory=lambda: [0,   50,  50])
    enemy_hsv_upper:       list = field(default_factory=lambda: [179, 255, 255])

    # Spielfeld-ROI: Anteile des Bildes (0.0–1.0)
    # Obere ~15% = UI (Boost-Anzeige, Timer, Score) → ignorieren
    roi_top:    float = 0.15
    roi_bottom: float = 0.92
    roi_left:   float = 0.05
    roi_right:  float = 0.95

    # Ball-Filter
    ball_min_radius:      int   = 6
    ball_max_radius:      int   = 60
    ball_circularity_min: float = 0.70   # 1.0 = perfekter Kreis

    # Gegner-Auto-Filter (kein Kreis → Rechteck-basiert)
    enemy_min_area:        int  = 200
    enemy_max_area:        int  = 8000
    enemy_aspect_ratio_lo: float = 0.5
    enemy_aspect_ratio_hi: float = 2.5

    # Boostpad-Filter
    boost_min_radius: int = 4
    boost_max_radius: int = 30

    # Tracker
    tracker_search_radius: int = 120   # Pixel-Suchradius um letzte Position
    tracker_lost_frames:   int = 10    # Frames ohne Fund → Tracker reset


# ── Tracker-Objekt ────────────────────────────────────────────────────────────

@dataclass
class TrackedObject:
    x:           int   = 0
    y:           int   = 0
    radius:      int   = 0
    width:       int   = 0
    height:      int   = 0
    confidence:  float = 0.0
    lost_frames: int   = 0
    found:       bool  = False


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def build_roi_mask(frame_shape: tuple, cfg: DetectionConfig) -> np.ndarray:
    """
    Erstellt eine binäre Maske, die nur den Spielfeld-Bereich freigibt.
    Alles außerhalb (UI-Elemente oben/unten/seitlich) wird ausgeblendet.
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    y1 = int(h * cfg.roi_top)
    y2 = int(h * cfg.roi_bottom)
    x1 = int(w * cfg.roi_left)
    x2 = int(w * cfg.roi_right)
    mask[y1:y2, x1:x2] = 255
    return mask


def apply_tracker_roi(mask: np.ndarray, tracker: TrackedObject,
                      cfg: DetectionConfig) -> np.ndarray:
    """
    Wenn ein Objekt bereits verfolgt wird, suchen wir nur noch in einem
    Radius um die letzte bekannte Position. Das eliminiert fast alle
    verbleibenden False Positives aus anderen Bildbereichen.
    """
    if not tracker.found or tracker.lost_frames >= cfg.tracker_lost_frames:
        return mask
    h, w = mask.shape[:2]
    tracker_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(tracker_mask, (tracker.x, tracker.y),
               cfg.tracker_search_radius, 255, -1)
    return cv2.bitwise_and(mask, tracker_mask)


def contour_circularity(contour) -> float:
    """
    Berechnet die Kreisförmigkeit einer Kontur.
    Formel: 4π * Fläche / Umfang²
    Ergebnis: 1.0 = perfekter Kreis, <0.7 = kein Kreis
    """
    area = cv2.contourArea(contour)
    if area < 1:
        return 0.0
    perimeter = cv2.arcLength(contour, True)
    if perimeter < 1:
        return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)


def hsv_mask(frame_bgr: np.ndarray, lower: list, upper: list) -> np.ndarray:
    """Erzeugt eine HSV-Maske und wendet Morphologie zur Rauschreduktion an."""
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array(lower, np.uint8),
                       np.array(upper, np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ── Erkennungs-Funktionen ────────────────────────────────────────────────────

def detect_ball(frame_bgr: np.ndarray,
                cfg: DetectionConfig,
                tracker: TrackedObject,
                roi_mask: Optional[np.ndarray] = None) -> TrackedObject:
    """
    Erkennt den Ball mit 5-stufiger Filter-Pipeline:
      1. HSV-Farbfilter
      2. Spielfeld-ROI (UI ausblenden)
      3. Tracker-ROI (Suchbereich einengen)
      4. Kreisförmigkeit + Größe
      5. Konfidenz-Scoring → bester Kandidat

    Gibt den aktualisierten Tracker zurück.
    Wenn kein Ball gefunden: letzte Position bleibt, lost_frames steigt.
    """
    mask = hsv_mask(frame_bgr, cfg.ball_hsv_lower, cfg.ball_hsv_upper)

    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)

    mask = apply_tracker_roi(mask, tracker, cfg)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best       = TrackedObject()
    best_score = -1.0

    for c in contours:
        if cv2.contourArea(c) < 10:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)

        if not (cfg.ball_min_radius <= radius <= cfg.ball_max_radius):
            continue

        circ = contour_circularity(c)
        if circ < cfg.ball_circularity_min:
            continue

        # Score: Kreisförmigkeit × normalisierte Fläche
        area  = cv2.contourArea(c)
        score = circ * min(area / 500.0, 1.0)

        if score > best_score:
            best_score      = score
            best.x          = int(cx)
            best.y          = int(cy)
            best.radius     = radius
            best.confidence = round(score, 3)
            best.found      = True
            best.lost_frames = 0

    if not best.found:
        tracker.lost_frames += 1
        if tracker.lost_frames >= cfg.tracker_lost_frames:
            tracker.found = False
        return tracker

    return best


def detect_enemy(frame_bgr: np.ndarray,
                 cfg: DetectionConfig,
                 tracker: TrackedObject,
                 roi_mask: Optional[np.ndarray] = None) -> TrackedObject:
    """
    Erkennt das gegnerische Auto (kein Kreis → Rechteck-basiert).
    Filtert nach Fläche und Seitenverhältnis statt Kreisförmigkeit.
    """
    mask = hsv_mask(frame_bgr, cfg.enemy_hsv_lower, cfg.enemy_hsv_upper)

    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)

    mask = apply_tracker_roi(mask, tracker, cfg)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best       = TrackedObject()
    best_score = -1.0

    for c in contours:
        area = cv2.contourArea(c)
        if not (cfg.enemy_min_area <= area <= cfg.enemy_max_area):
            continue

        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h if h > 0 else 0
        if not (cfg.enemy_aspect_ratio_lo <= aspect <= cfg.enemy_aspect_ratio_hi):
            continue

        # Score: größere, kompaktere Objekte bevorzugen
        hull  = cv2.convexHull(c)
        if cv2.contourArea(hull) > 0:
            solidity = area / cv2.contourArea(hull)
        else:
            solidity = 0.0
        score = solidity * min(area / cfg.enemy_max_area, 1.0)

        if score > best_score:
            best_score      = score
            best.x          = x + w // 2
            best.y          = y + h // 2
            best.width      = w
            best.height     = h
            best.confidence = round(score, 3)
            best.found      = True
            best.lost_frames = 0

    if not best.found:
        tracker.lost_frames += 1
        if tracker.lost_frames >= cfg.tracker_lost_frames:
            tracker.found = False
        return tracker

    return best


def detect_boost_pads(frame_bgr: np.ndarray,
                      cfg: DetectionConfig,
                      roi_mask: Optional[np.ndarray] = None) -> list[TrackedObject]:
    """
    Erkennt ALLE sichtbaren Boostpads (gibt Liste zurück, kein Single-Tracker).
    Boostpads sind rund und relativ klein.
    """
    mask = hsv_mask(frame_bgr, cfg.boost_pad_hsv_lower, cfg.boost_pad_hsv_upper)

    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    results = []

    for c in contours:
        if cv2.contourArea(c) < 8:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)

        if not (cfg.boost_min_radius <= radius <= cfg.boost_max_radius):
            continue

        circ = contour_circularity(c)
        if circ < 0.55:   # Boostpads etwas weniger streng als Ball
            continue

        obj            = TrackedObject()
        obj.x          = int(cx)
        obj.y          = int(cy)
        obj.radius     = radius
        obj.confidence = round(circ, 3)
        obj.found      = True
        results.append(obj)

    return results


def detect_goals(frame_bgr: np.ndarray,
                 cfg: DetectionConfig,
                 roi_mask: Optional[np.ndarray] = None) -> dict:
    """
    Erkennt eigenes und gegnerisches Tor.
    Gibt dict {'own': TrackedObject, 'enemy': TrackedObject} zurück.
    """
    results = {}

    for key, lo, hi in [
        ("own",   cfg.goal_own_hsv_lower,   cfg.goal_own_hsv_upper),
        ("enemy", cfg.goal_enemy_hsv_lower,  cfg.goal_enemy_hsv_upper),
    ]:
        mask = hsv_mask(frame_bgr, lo, hi)
        if roi_mask is not None:
            mask = cv2.bitwise_and(mask, roi_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        obj = TrackedObject()

        if contours:
            # Größte Kontur = wahrscheinlichstes Tor
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 300:
                x, y, w, h = cv2.boundingRect(largest)
                obj.x      = x + w // 2
                obj.y      = y + h // 2
                obj.width  = w
                obj.height = h
                obj.found  = True
                obj.confidence = round(min(cv2.contourArea(largest) / 5000.0, 1.0), 3)

        results[key] = obj

    return results


# ── Visualisierung ────────────────────────────────────────────────────────────

# Farben für jedes Objekt (BGR)
COLORS = {
    "ball":        (0,   255, 100),
    "enemy":       (200,  0,  200),
    "boost":       (0,   220, 220),
    "goal_own":    (255, 100,   0),
    "goal_enemy":  (0,    50, 220),
}


def draw_ball(frame: np.ndarray, obj: TrackedObject) -> np.ndarray:
    if not obj.found:
        return frame
    c = COLORS["ball"]
    cv2.circle(frame, (obj.x, obj.y), obj.radius + 4, c, 2)
    cv2.circle(frame, (obj.x, obj.y), 3, (255, 255, 255), -1)
    cv2.putText(frame, f"Ball {obj.confidence:.2f}",
                (obj.x - 30, obj.y - obj.radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return frame


def draw_enemy(frame: np.ndarray, obj: TrackedObject) -> np.ndarray:
    if not obj.found:
        return frame
    c  = COLORS["enemy"]
    x1 = obj.x - obj.width  // 2
    y1 = obj.y - obj.height // 2
    cv2.rectangle(frame, (x1, y1), (x1 + obj.width, y1 + obj.height), c, 2)
    cv2.circle(frame, (obj.x, obj.y), 3, (255, 255, 255), -1)
    cv2.putText(frame, f"Gegner {obj.confidence:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return frame


def draw_boosts(frame: np.ndarray, boosts: list[TrackedObject]) -> np.ndarray:
    c = COLORS["boost"]
    for obj in boosts:
        if not obj.found:
            continue
        cv2.circle(frame, (obj.x, obj.y), obj.radius + 3, c, 2)
        cv2.putText(frame, "Boost",
                    (obj.x - 18, obj.y - obj.radius - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
    return frame


def draw_goals(frame: np.ndarray, goals: dict) -> np.ndarray:
    for key, color_key in [("own", "goal_own"), ("enemy", "goal_enemy")]:
        obj = goals.get(key)
        if obj is None or not obj.found:
            continue
        c  = COLORS[color_key]
        x1 = obj.x - obj.width  // 2
        y1 = obj.y - obj.height // 2
        cv2.rectangle(frame, (x1, y1), (x1 + obj.width, y1 + obj.height), c, 2)
        label = "Eigenes Tor" if key == "own" else "Gegner-Tor"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return frame


def draw_roi(frame: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    """Zeigt den aktiven Spielfeld-ROI als gestricheltes Rechteck."""
    h, w = frame.shape[:2]
    x1 = int(w * cfg.roi_left)
    x2 = int(w * cfg.roi_right)
    y1 = int(h * cfg.roi_top)
    y2 = int(h * cfg.roi_bottom)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
    cv2.putText(frame, "Spielfeld-ROI", (x1 + 4, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    return frame


# ── Haupt-Pipeline ────────────────────────────────────────────────────────────

class GameDetector:
    """
    Zentrale Klasse die alle Erkennungen zusammenfasst.
    Einfach instanziieren und pro Frame update() aufrufen.

    Beispiel:
        detector = GameDetector()
        while True:
            frame = grab_frame()
            state = detector.update(frame)
            vis   = detector.visualize(frame, state)
            cv2.imshow("Bot", vis)
    """

    def __init__(self, cfg: Optional[DetectionConfig] = None):
        self.cfg = cfg or DetectionConfig()
        self._roi_mask: Optional[np.ndarray] = None

        # Separate Tracker pro Objekt
        self.ball_tracker  = TrackedObject()
        self.enemy_tracker = TrackedObject()

    def _ensure_roi(self, frame: np.ndarray) -> None:
        if self._roi_mask is None:
            self._roi_mask = build_roi_mask(frame.shape, self.cfg)

    def update(self, frame_bgr: np.ndarray) -> dict:
        """
        Führt alle Erkennungen durch und gibt den aktuellen Spielzustand zurück.

        Rückgabe:
            {
                'ball':       TrackedObject,
                'enemy':      TrackedObject,
                'boosts':     list[TrackedObject],
                'goals':      {'own': TrackedObject, 'enemy': TrackedObject},
            }
        """
        self._ensure_roi(frame_bgr)

        self.ball_tracker  = detect_ball(
            frame_bgr, self.cfg, self.ball_tracker,  self._roi_mask)
        self.enemy_tracker = detect_enemy(
            frame_bgr, self.cfg, self.enemy_tracker, self._roi_mask)

        boosts = detect_boost_pads(frame_bgr, self.cfg, self._roi_mask)
        goals  = detect_goals(frame_bgr, self.cfg, self._roi_mask)

        return {
            "ball":   self.ball_tracker,
            "enemy":  self.enemy_tracker,
            "boosts": boosts,
            "goals":  goals,
        }

    def visualize(self, frame_bgr: np.ndarray, state: dict,
                  show_roi: bool = True) -> np.ndarray:
        """Zeichnet alle Erkennungen auf das Frame und gibt es zurück."""
        vis = frame_bgr.copy()
        if show_roi:
            vis = draw_roi(vis, self.cfg)
        vis = draw_ball(vis, state["ball"])
        vis = draw_enemy(vis, state["enemy"])
        vis = draw_boosts(vis, state["boosts"])
        vis = draw_goals(vis, state["goals"])
        return vis


# ── Standalone-Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Schnelltest: Rocket League im Fenster-Modus starten,
    dann dieses Script ausführen. Zeigt Live-Erkennungen.
    """
    try:
        import mss
        def grab():
            with mss.mss() as sct:
                mon = sct.monitors[1]
                img = sct.grab(mon)
                return np.array(img)[:, :, :3]
    except ImportError:
        from PIL import ImageGrab
        def grab():
            img = ImageGrab.grab()
            return np.array(img)[:, :, ::-1]

    detector = GameDetector()

    print("Detektor gestartet — q zum Beenden")
    while True:
        frame = grab()
        state = detector.update(frame)
        vis   = detector.visualize(frame, state)

        # Status in Konsole
        b = state["ball"]
        e = state["enemy"]
        if b.found:
            print(f"\rBall: ({b.x},{b.y}) r={b.radius} conf={b.confidence:.2f}  ", end="")
        else:
            print(f"\rBall: nicht gefunden (lost={b.lost_frames})          ", end="")

        # Auf 1280x720 skalieren für die Anzeige
        vis_small = cv2.resize(vis, (1280, 720))
        cv2.imshow("RL Detektor", vis_small)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("\nBeendet.")
