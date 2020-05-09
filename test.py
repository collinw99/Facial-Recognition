import cv2

# capture from video device 2
CAP = cv2.VideoCapture(2)
# use the face classifier training data from cv2 install location
FACE_CASCADE = cv2.CascadeClassifier('/home/collinw99/.local/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    RET, frame = CAP.read()

    # Our operation on the frame come here
    GRAY = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # use the grayscale frame to find the faces with the classifier
    # parameters: (image, rejectLevels, levelWeights, ?scaleFactor, ?minNeighbors, ?flags, ?minSize, ?maxSize, ?outputRejectLevels)
    FACES = FACE_CASCADE.detectMultiScale(GRAY, 1.3, 5)
    # for each face found
    for i, (x, y, w, h) in enumerate(FACES):
        COLOR = (30*(i+2)%256, 20*(i+2)%256, 40*(i+2)%256)
        # draw a rectangle in the frame around the face found
        # parameters: (image, point1, point2, color, thickness, ?lineType, ?shift)
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR, 2)
        frame = cv2.putText(frame, 'face'+str(i), (x, y+h-2), cv2.FONT_HERSHEY_PLAIN, 1, COLOR)

    # Display the resulting frame
    cv2.imshow('Video Feed', frame)
    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
CAP.release()
cv2.destroyAllWindows()
