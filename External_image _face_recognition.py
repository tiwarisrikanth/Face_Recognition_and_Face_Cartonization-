import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade.load("haarcascade_frontalface_default.xml")

imgage = cv2.imread("face4.jpg")
imgage = cv2.imread("face3.jpg")

print(imgage)

gray_image = cv2.cvtColor(imgage, cv2.COLOR_BGR2GRAY)
    
gray_image = cv2.medianBlur(gray_image, 5)

edges = cv2.adaptiveThreshold(gray_image, 250, cv2.ADAPTIVE_THRESH_MEAN_C,  
                                         cv2.THRESH_BINARY, 5, 5) 

# Cartoonization 
color = cv2.bilateralFilter(gray_image, 10, 250, 250) 
cartoon = cv2.bitwise_and(color, color, mask=edges) 

faces = face_cascade.detectMultiScale(gray_image, scaleFactor = 1.05, minNeighbors = 5)

for x,y,w,h in faces:
        img = cv2.rectangle(imgage, (x,y), (x+w, y+h), (0,255,0), 3)


cv2.imshow("Gray Capture", gray_image)
cv2.imshow("Representing edges", edges) 
cv2.imshow("Cartoonized", cartoon)
cv2.imshow("Detection", imgage)
cv2.waitKey(0)
cv2.destroyAllWindows()