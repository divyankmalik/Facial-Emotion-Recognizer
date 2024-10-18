import cv2
from keras.models import model_from_json
import numpy as np

# Load the model architecture
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# Load the model weights
model.load_weights("facialemotionmodel.h5")

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    try:
        # Capture frame from webcam
        i, im = webcam.read()
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (p, q, r, s) in faces:
            # Extract the region of interest (the face)
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            
            # Resize the face to the required input size for the model
            image = cv2.resize(image, (48, 48))
            
            # Extract features and make a prediction
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            
            # Display the prediction on the frame
            cv2.putText(im, '%s' % prediction_label, (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        
        # Display the frame
        cv2.imshow("Output", im)
        
        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    except Exception as e:
        print("An error occurred: ", e)
        pass

# Release the webcam and close windows
webcam.release()
cv2.destroyAllWindows()
