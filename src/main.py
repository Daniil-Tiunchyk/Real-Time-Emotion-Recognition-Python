from src.camera import Camera
from src.emotion_recognizer import EmotionRecognizer
from src.interface_renderer import InterfaceRenderer
import cv2


def main():
    camera = Camera()
    emotion_recognizer = EmotionRecognizer()
    renderer = InterfaceRenderer()

    while True:
        ret, frame = camera.read_frame()
        if not ret:
            break

        faces = emotion_recognizer.detect_faces(frame)

        for i, face in enumerate(faces):
            x, y, w, h = face.left(), face.top(), face.right(), face.bottom()
            face_region = frame[y:h, x:w]

            if emotion_recognizer.recognize_emotions(face_region, i):
                dominant_emotion, confidence = emotion_recognizer.get_dominant_emotion(i)
                if dominant_emotion and confidence > 0.5:
                    renderer.render_face_rectangle(frame, x, y, w, h)
                    renderer.render_emotion_text(frame, x, y, dominant_emotion, confidence)

        renderer.render_info_panel(frame, len(faces), emotion_recognizer.previous_emotions)
        cv2.imshow('Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
