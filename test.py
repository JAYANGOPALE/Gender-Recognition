import cv2
import numpy as np
import tensorflow as tf
print(cv2.__version__)
# Load the trained models
gender_model = tf.keras.models.load_model('gender_classification_model.h5')
print("\n\nload gender detection model\n")

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
a=str(input("do you want to see video input (Y/N)"))
if a=="Y" or a=="y":
    # OpenCV video capture for real-time gender and mask detection
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream or file")
        exit()
    
    while True:
        ret, frame = cap.read()
    
        if not ret:
            print("Error: Failed to capture image")
            break
            
        # Convert the frame to grayscale (needed for Haar cascade)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) of the face
            face_roi = frame[y:y+h, x:x+w]

            '''# Preprocess the face ROI for mask detection
            mask_face = cv2.resize(face_roi, (128, 128))  # Resize to the input size of your mask model
            mask_face = mask_face.astype('float32') / 255.0
            mask_face = np.expand_dims(mask_face, axis=0)'''

            '''# Predict mask or no mask
            mask_prob = mask_model.predict(mask_face)[0][0]
            mask_status = 'Mask' if mask_prob > 0.7 else 'No Mask'''

            # Preprocess the face ROI for gender classification
            gender_face = cv2.resize(face_roi, (64, 64))  # Resize to the input size of your gender model
            gender_face = gender_face.astype('float32') / 255.0
            gender_face = np.expand_dims(gender_face, axis=0)

            # Predict gender
            gender_prob = gender_model.predict(gender_face)[0][0]
            gender = 'Male' if gender_prob > 0.5 else 'Female'

            # Display the results
            #mask_color = (0, 0, 255) if mask_status == 'Mask' else (0, 255, 0)
            #cv2.putText(frame, f"Mask: {mask_status}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mask_color, 2)
            cv2.putText(frame, f"Gender: {gender}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Gender Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
else:
    print("code")
