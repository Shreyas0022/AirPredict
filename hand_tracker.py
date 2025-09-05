# hand_tracker.py (Gesture-only, clean version)

import cv2
import mediapipe as mp
import math


class HandTracker:
    """
    Hand tracker for gesture recognition only.
    Provides: DRAW (index up), MOVE (open palm), PINCH (thumb-index close).
    """

    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.results = None
        self.pinch_threshold = 30  # Pixel distance for pinch gesture

    @staticmethod
    def fingers_up(landmarks):
        """
        Returns list of booleans for [Index, Middle, Ring, Pinky].
        True = finger up, False = finger down.
        """
        fingers = []
        # Index finger
        fingers.append(landmarks[8].y < landmarks[6].y)
        # Middle
        fingers.append(landmarks[12].y < landmarks[10].y)
        # Ring
        fingers.append(landmarks[16].y < landmarks[14].y)
        # Pinky
        fingers.append(landmarks[20].y < landmarks[18].y)
        return fingers

    def process_frame(self, frame):
        """
        Processes a frame and detects gestures.
        Returns dict: { frame, gesture, cursor_coords, pinch_coords }
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)

        output = {
            "frame": frame,
            "gesture": "NONE",
            "cursor_coords": None,
            "pinch_coords": None,
        }

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                # Key points
                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                cursor_coords = (int(index_tip.x * w), int(index_tip.y * h))
                output["cursor_coords"] = cursor_coords

                # Finger states
                finger_states = self.fingers_up(hand_landmarks.landmark)

                # --- Gesture Logic ---
                # Pinch
                distance = math.hypot(index_tip.x - thumb_tip.x,
                                      index_tip.y - thumb_tip.y) * w
                if distance < self.pinch_threshold:
                    output["gesture"] = "PINCH"
                    pinch_coords = (
                        int((index_tip.x + thumb_tip.x) / 2 * w),
                        int((index_tip.y + thumb_tip.y) / 2 * h),
                    )
                    output["pinch_coords"] = pinch_coords

                # Open palm (all fingers up → MOVE)
                elif finger_states == [True, True, True, True]:
                    output["gesture"] = "MOVE"

                # Index only (→ DRAW)
                elif finger_states == [True, False, False, False]:
                    output["gesture"] = "DRAW"

        return output
