import cv2
import mediapipe as mp

def face_anonymizer():
    cap = cv2.VideoCapture(1)
    mp_face_detector = mp.solutions.face_detection
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with mp_face_detector.FaceDetection(model_selection=0, min_detection_confidence=0.2) as face_detection:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = face_detection.process(img_rgb)

            if out.detections is not None:
                for detection in out.detections:
                    location_data = detection.location_data
                    bbox = location_data.relative_bounding_box

                    x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                    x1 = int(x1 * frame.shape[1])
                    y1 = int(y1 * frame.shape[0])
                    w = int(w * frame.shape[1])
                    h = int(h * frame.shape[0])

                    img =  frame
                    img[y1:y1 + h, x1:x1 + w] = cv2.blur(img[y1:y1 + h, x1:x1 + w], (50, 50))
                    cv2.putText(img, 'Face', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imshow('frame', img)
            else:
                cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
