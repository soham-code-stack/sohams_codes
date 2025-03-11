import cv2
from deepface import DeepFace # type: ignore

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'ESC' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    # Resize frame for faster processing (optional, adjust if needed)
    small_frame = cv2.resize(frame, (640, 480))

    try:
        # Analyze emotions with DeepFace
        analysis = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False)

        for face in analysis:
            region = face.get('region', {})
            mood = face.get('dominant_emotion', 'Unknown')

            if 'x' in region and 'y' in region and 'w' in region and 'h' in region:
                x, y, w, h = region['x'], region['y'], region['w'], region['h']

                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Display detected emotion
                cv2.putText(frame, mood, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    except Exception as e:
        print(f"Error processing frame: {e}")

    # Show output
    cv2.imshow('Mood Detection', frame)

    # Exit on 'ESC' key press
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
