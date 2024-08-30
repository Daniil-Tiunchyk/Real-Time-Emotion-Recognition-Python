import cv2


class InterfaceRenderer:
    def __init__(self, text_color=(255, 255, 255), font_scale=0.7, font_thickness=2, line_height=30):
        self.text_color = text_color
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.line_height = line_height
        self.info_panel_width = 300

    def render_face_rectangle(self, frame, x, y, w, h):
        cv2.rectangle(frame, (x, y), (w, h), (36, 255, 12), 3)

    def render_emotion_text(self, frame, x, y, dominant_emotion, confidence):
        cv2.putText(frame, f'{dominant_emotion}: {confidence:.2f}', (x + 5, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.text_color, 3, cv2.LINE_AA)

    def render_info_panel(self, frame, num_faces, previous_emotions):
        start_y = 30
        cv2.putText(frame, f'Number of faces: {num_faces}', (frame.shape[1] - self.info_panel_width + 10, start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.text_color, self.font_thickness, cv2.LINE_AA)

        start_y += self.line_height

        for face_id, data in previous_emotions.items():
            if not data['emotions']:
                continue
            dominant_emotion = max(data['emotions'], key=data['emotions'].get)
            avg_confidence = sum(data['emotions'].values()) / len(data['emotions'])
            total_time = data['total_time']

            cv2.putText(frame, f'Emotion: {dominant_emotion}', (frame.shape[1] - self.info_panel_width + 10, start_y),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.text_color, self.font_thickness, cv2.LINE_AA)
            start_y += self.line_height

            cv2.putText(frame, f'Avg. Confidence: {avg_confidence:.2f}',
                        (frame.shape[1] - self.info_panel_width + 10, start_y),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.text_color, self.font_thickness, cv2.LINE_AA)
            start_y += self.line_height

            cv2.putText(frame, f'Time: {total_time} frames', (frame.shape[1] - self.info_panel_width + 10, start_y),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.text_color, self.font_thickness, cv2.LINE_AA)
            start_y += self.line_height
            break
