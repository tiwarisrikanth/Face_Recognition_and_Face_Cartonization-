import cv2, time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade.load("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)

a=1

while True:
    
    a = a + 1

    check, frame = video.read() 
    
    print(frame)
    
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray_image = cv2.medianBlur(gray_image, 5)
    
    edges = cv2.adaptiveThreshold(gray_image, 250, cv2.ADAPTIVE_THRESH_MEAN_C,  
                                         cv2.THRESH_BINARY, 5, 5) 
    
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor = 1.05, minNeighbors = 5)
    
    
        
     # Cartoonization 
    color = cv2.bilateralFilter(gray_image, 10, 250, 250) 
    cartoon = cv2.bitwise_and(color, color, mask=edges) 
    
    for x,y,w,h in faces:
        img = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

    
    cv2.imshow("Gray Capture", gray_image)
    cv2.imshow("Detection", frame)
    cv2.imshow("Representing edges", edges) 
    cv2.imshow("Cartoonized", cartoon)


    key = cv2.waitKey(1)
    
    if key == ord("q"):
        
        break       
print(a)
video.release()
cv2.destroyAllWindows()