import cv2
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break