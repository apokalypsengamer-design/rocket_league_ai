from __future__ import annotations
from config import Config
from core.state import GameState, ObjectPosition, BoostPadType
from input.actions import Action


class RuleEngine:
    """
    Regelbasierte Entscheidungslogik — Prioritäten (hoch → niedrig):

      1. SCHUSS       Ball nah + Spieler hinter Ball + Winkel zum Tor ok
                      → Volle Fahrt + Boost auf gegnerisches Tor ausrichten
      2. ANGRIFF      Ball in Angriffshälfte
                      → Zum Ball fahren, dann auf gegnerisches Tor schießen
      3. VERTEIDIGUNG Ball in eigener Hälfte
                      → Zurück, Ball abfangen, Gegner berücksichtigen
      4. BOOST_HOLEN  Boost niedrig
                      → Nächsten (möglichst LARGE) Pad anfahren
      5. ROTATION     Ball unsichtbar / Mittelfeld
                      → Neutral positionieren
    """

    def __init__(self, config: Config):
        self._dz  = config.gameplay.steer_dead_zone
        self._cfg = config.gameplay

    def evaluate(self, state: GameState, frame: int) -> tuple[Action, str]:
        """
        Gibt (Action, reasoning_string) zurück.
        Der reasoning_string wird direkt im Brain genutzt.
        """
        # Höchste Priorität: Schuss-Opportunity
        if state.shot_opportunity:
            return self._shoot(state)

        if state.phase == "attack":
            return self._attack(state)
        if state.phase == "defense":
            return self._defense(state)
        if state.phase == "boost_collect":
            return self._boost_collect(state)
        return self._rotate(state)

    # ── Phase-Handler ────────────────────────────────────────────────────────

    def _shoot(self, state: GameState) -> tuple[Action, str]:
        """
        Schuss-Ausführung:
        Spieler steht hinter dem Ball → volle Fahrt + Boost
        Ausrichtung: nicht zum Ball, sondern durch den Ball aufs Tor.
        """
        # Schuss-Vektor: vom Spieler durch den Ball hindurch Richtung Tor
        # Wir steuern leicht Richtung Tor statt direkt zum Ball,
        # damit der Ball in die richtige Richtung fliegt.
        shoot_target = self._aim_through_ball(state)
        action = Action(forward=True, boost=True)
        action = self._steer_to(action, state.player_x, shoot_target)
        angle_str = f"{state.ball_to_goal_angle:.2f}"
        reason = (
            f"SCHUSS! Ball ({state.ball_x:.2f},{state.ball_y:.2f}) "
            f"→ Tor ({state.enemy_goal.x:.2f},{state.enemy_goal.y:.2f}) "
            f"| Winkel={angle_str}"
        )
        return action, reason

    def _attack(self, state: GameState) -> tuple[Action, str]:
        """
        Angriff: Zum Ball fahren, dabei aufs gegnerische Tor ausrichten.
        Wenn ein Gegner zwischen Ball und Tor steht → aggressiver anfahren.
        """
        if not state.ball_visible:
            # Ball nicht sichtbar → ans gegnerische Tor fahren
            action = Action(forward=True, boost=True)
            action = self._steer_to(action, state.player_x, state.enemy_goal)
            return action, "Ball unsichtbar → fahre zum gegnerischen Tor"

        # Ziel: nicht direkt zum Ball, sondern so positionieren dass
        # Ball zwischen Spieler und Tor liegt ("hinter den Ball kommen")
        approach_target = self._approach_behind_ball(state)
        use_boost = state.boost > 30 or state.enemy_between_ball_and_goal
        action = Action(forward=True, boost=use_boost)
        action = self._steer_to(action, state.player_x, approach_target)

        if state.enemy_between_ball_and_goal:
            reason = f"Angriff | Gegner im Weg → aggressiv zum Ball ({state.ball_x:.2f},{state.ball_y:.2f})"
        else:
            reason = f"Angriff | Ball ({state.ball_x:.2f},{state.ball_y:.2f}) → Tor anvisieren"
        return action, reason

    def _defense(self, state: GameState) -> tuple[Action, str]:
        """
        Verteidigung: Ball abfangen. Gegnerposition berücksichtigen.
        Wenn Gegner näher am Ball als wir → eher ans eigene Tor zurück.
        """
        en = state.nearest_enemy
        ball_obj = self._ball_as_obj(state)

        # Ist ein Gegner näher am Ball als wir?
        enemy_closer = False
        if en and en.visible and state.ball_visible:
            enemy_dist = en.distance_to(state.ball_x, state.ball_y)
            our_dist   = state.ball_dist_to_player
            enemy_closer = enemy_dist < our_dist * 0.8

        if enemy_closer:
            # Gegner hat den Ball → wir fahren ans eigene Tor (defensive Position)
            target = state.own_goal
            action = Action(forward=True, boost=True)
            action = self._steer_to(action, state.player_x, target)
            reason = f"Verteidigung | Gegner näher am Ball → zurück zum Tor"
        else:
            # Wir können den Ball noch erreichen → abfangen
            target = ball_obj if state.ball_visible else state.own_goal
            action = Action(forward=True, boost=True)
            action = self._steer_to(action, state.player_x, target)
            e_str = f"Gegner@({en.x:.2f},{en.y:.2f})" if en else "kein Gegner"
            reason = f"Verteidigung | Ball abfangen | {e_str}"

        return action, reason

    def _boost_collect(self, state: GameState) -> tuple[Action, str]:
        """
        Boost holen: Bevorzuge LARGE Pads (100 Boost).
        Wenn kein Pad sichtbar → zum Ball fahren.
        """
        # Bevorzuge großes Pad
        pad = state.nearest_large_boost or state.nearest_boost
        action = Action(forward=True)

        if pad and pad.visible:
            action = self._steer_to(action, state.player_x, pad)
            pad_type_str = "LARGE" if pad.pad_type == BoostPadType.LARGE else "small"
            reason = (
                f"Boost niedrig ({state.boost:.0f}%) → "
                f"{pad_type_str} Pad @ ({pad.x:.2f},{pad.y:.2f})"
            )
        elif state.ball_visible:
            action = self._steer_to(action, state.player_x, self._ball_as_obj(state))
            reason = f"Boost niedrig ({state.boost:.0f}%) | kein Pad sichtbar → zum Ball"
        else:
            reason = f"Boost niedrig ({state.boost:.0f}%) | nichts sichtbar → vorwärts"

        return action, reason

    def _rotate(self, state: GameState) -> tuple[Action, str]:
        """
        Rotation/Neupositionierung: Ball suchen, neutral positionieren.
        """
        action = Action(forward=True)
        if state.ball_visible:
            # Wir fahren nicht direkt zum Ball, sondern zu einer
            # neutralen Position zwischen Ball und eigenem Tor
            neutral = self._neutral_position(state)
            action  = self._steer_to(action, state.player_x, neutral)
            reason  = f"Rotation | Ball ({state.ball_x:.2f},{state.ball_y:.2f}) | neutral positionieren"
        else:
            reason = "Rotation | Ball unsichtbar → vorwärts"
        return action, reason

    # ── Steuerungs-Helfer ────────────────────────────────────────────────────

    def _steer_to(self, action: Action, player_x: float,
                  target: ObjectPosition) -> Action:
        """Lenkt den Bot zum Ziel basierend auf horizontalem Offset."""
        if not target or not target.visible:
            return action
        diff = target.x - player_x
        action.steer_left  = diff < -self._dz
        action.steer_right = diff >  self._dz
        return action

    def _approach_behind_ball(self, state: GameState) -> ObjectPosition:
        """
        Berechnet einen Anfahrtspunkt "hinter" dem Ball – also so dass
        der Ball zwischen Spieler und gegnerischem Tor liegt.

        Das Ziel liegt leicht hinter dem Ball (von der Torseite aus).
        """
        if not state.ball_visible:
            return state.enemy_goal

        # Vektor Ball → eigenes Tor (normiert)
        dx = state.own_goal.x - state.ball_x
        dy = state.own_goal.y - state.ball_y
        dist = (dx * dx + dy * dy) ** 0.5 + 1e-6

        # Offset: 0.08 hinter dem Ball in Richtung eigenes Tor
        offset = 0.10
        approach_x = state.ball_x + (dx / dist) * offset
        approach_y = state.ball_y + (dy / dist) * offset

        # Clamp auf Spielfeld
        approach_x = max(0.05, min(0.95, approach_x))
        approach_y = max(0.05, min(0.95, approach_y))

        return ObjectPosition(approach_x, approach_y, True)

    def _aim_through_ball(self, state: GameState) -> ObjectPosition:
        """
        Beim Schuss: Ziel ist nicht der Ball selbst, sondern der Punkt
        der den Ball optimal Richtung Tor schickt.
        Wir fahren leicht Richtung Tor-x ausgerichtet.
        """
        if not state.ball_visible:
            return state.enemy_goal

        # Mische Ball-Position und Tor-Position (70% Ball, 30% Tor-X)
        aim_x = state.ball_x * 0.5 + state.enemy_goal.x * 0.5
        aim_y = state.ball_y  # Y hauptsächlich zum Ball

        return ObjectPosition(aim_x, aim_y, True)

    def _neutral_position(self, state: GameState) -> ObjectPosition:
        """
        Neutrale Position: zwischen Ball und eigenem Tor,
        leicht zur Mitte hin – klassische Rocket-League-Rotation.
        """
        if not state.ball_visible:
            return ObjectPosition(0.5, 0.7, True)

        mid_x = (state.ball_x + 0.5) / 2.0   # Richtung Mitte
        mid_y = (state.ball_y + state.own_goal.y) / 2.0
        return ObjectPosition(
            max(0.1, min(0.9, mid_x)),
            max(0.1, min(0.9, mid_y)),
            True,
        )

    @staticmethod
    def _ball_as_obj(state: GameState) -> ObjectPosition:
        return ObjectPosition(state.ball_x, state.ball_y, state.ball_visible)
