import cv2
import dlib
from fer import FER
from collections import defaultdict


class EmotionRecognizer:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.emotion_detector = FER()
        self.previous_emotions = defaultdict(lambda: {'emotions': {}, 'frames_left': 0, 'total_time': 0})
        self.smoothing_factor = 0.5
        self.emotion_display_time = 50

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        return faces

    def recognize_emotions(self, face_region, face_id):
        emotions = self.emotion_detector.detect_emotions(face_region)
        if emotions:
            current_emotion = emotions[0]['emotions']
            self._smooth_emotions(face_id, current_emotion)
            return True
        elif self.previous_emotions[face_id]['frames_left'] > 0:
            self.previous_emotions[face_id]['frames_left'] -= 1
            return True
        return False

    def _smooth_emotions(self, face_id, current_emotion):
        previous_emotion = self.previous_emotions[face_id]['emotions']
        for emotion in current_emotion:
            previous_emotion[emotion] = (
                    previous_emotion.get(emotion, current_emotion[emotion]) * self.smoothing_factor +
                    current_emotion[emotion] * (1 - self.smoothing_factor)
            )
        self.previous_emotions[face_id]['emotions'] = previous_emotion
        self.previous_emotions[face_id]['frames_left'] = self.emotion_display_time
        self.previous_emotions[face_id]['total_time'] += 1

    def get_dominant_emotion(self, face_id):
        emotions = self.previous_emotions[face_id]['emotions']
        if not emotions:
            return None, None
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions[dominant_emotion]
        return dominant_emotion, confidence
