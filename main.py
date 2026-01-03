from deepface import DeepFace
import cv2

cam = cv2.VideoCapture(0)
detected_emotion = None

while True:
    output, frame = cam.read()
    if not output:
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        detected_emotion = result[0]['dominant_emotion']

        cv2.putText(frame,
                    "Emotion: " + detected_emotion,
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
        )

    except Exception as e:
        print("Error:", e)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

print("Detected Emotin", detected_emotion)