import cv2                                          # use OpenCV version 2 in this program
import sys                                          # this line helps to get the images from the computer's directory

# Get user supplied values
imagePath = '6.jpg'                            #get the abba image 
cascPath = "haarcascade_frontalface_default.xml"    # use the OpenCV face detection tool

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)       # starts/initializes the cascade process which will eventually detect face

# Read the image
image = cv2.imread(imagePath)                       #check the image and store in image variable
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      #convert the image into grayscale or gray image without color

# Detect faces in the image
faces = faceCascade.detectMultiScale(               #detectMultiScale is the function to detect face in the image
    gray,                                           # detect gray image
    scaleFactor=1.1,                                # some faces are closer to the image than faces in the back. Scale factor checks those info
    minNeighbors=5,                                 # The detection method uses a moving window to detect faces. minNeighbors defines how many
                                                    # faces are detected near the current one before it decales the face found
    minSize=(30, 30)                                # gives the size of each window                                       
)

print("I have Found {0} faces!".format(len(faces)))        # prints how many faces are found

# Draw a rectangle around the faces                 
for (x, y, w, h) in faces:                                       #this fuction returns a list of rectangles where it found faces and 
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)     #keeps on looping where it found something

cv2.imshow("Faces found", image)                    #diaplay the image
cv2.waitKey(0)                                      #wait for the user to enter a key
