import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model

# Camera resolution and brightness (adjust as needed)
frameWidth = 640
frameHeight = 480
brightness = 180
confidence_threshold = 0.8  # Minimum probability for displaying a prediction
font = cv2.FONT_HERSHEY_SIMPLEX

# Setup video capture
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# Load the trained model
try:
  model = load_model("model.h5")
except OSError as e:
  print("Error loading model:", e)
  exit()

def grayscale(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
  return cv2.equalizeHist(img)

def preprocessing(img):
  img = grayscale(img)
  img = equalize(img)
  img = img / 255
  return img

def getCalssName(classNo):
  # Replace with your actual class name mapping function (modify as needed)
  class_names = {
      0: 'Speed Limit 20 km/h',
        1: 'Speed Limit 30 km/h',
        2: 'Speed Limit 50 km/h',
        3: 'Speed Limit 60 km/h' ,
        4: 'Speed Limit 70 km/h',
        5: 'Speed Limit 80 km/h',
        6: 'Speed Limit 90 km/h',
        7: 'Speed Limit 100 km/h',
        8: 'Speed Limit 120 km/h',
        35: 'Go straight',
        37 : 'Compulsory ahead or turn left',
        36 : 'Compulsory ahead or turn right',
        34 : 'Compuklsory turn left',
        33 : 'Compuklsory turn Right',
        32 : 'Restirction End',
        31 : 'Wild zone',
        24 : 'Narrow ahead (right) ',
        20:   'Turn right ',
        19: 'Turn left ',
        18 : 'Avoid alert',

      # ... (add all your class names)
  }
  return class_names.get(classNo, "Speed Limit 80 km/h ")  # Handle unknown classes gracefully

while True:
  success, imgOrignal = cap.read()

  if not success:
    print("Error reading frame from camera")
    break

  img = np.asarray(imgOrignal)
  img = cv2.resize(img, (32, 32))
  img = preprocessing(img)
  cv2.imshow("Processed Image", img)
  img = img.reshape(1, 32, 32, 1)

  # Display placeholders
  cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
  cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

  # Make predictions
  predictions = model.predict(img)
  classIndex = np.argmax(predictions)
  probabilityValue = np.amax(predictions)

  # Filter out low-confidence predictions
  if probabilityValue > confidence_threshold:
    class_name = getCalssName(classIndex)
    cv2.putText(imgOrignal, str(class_name) + " " + str(round(probabilityValue * 100, 2)) + "%", (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

  # Show the final image with predictions
  cv2.imshow("Result", imgOrignal)

  k = cv2.waitKey(1)
  if k == ord('q'):
    break

# Cleanup
cv2.destroyAllWindows()
cap.release()
