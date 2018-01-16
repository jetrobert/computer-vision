import cv2

cascade_fn = 'haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_fn)

img = cv2.imread("img/fe.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#gray = cv2.equalizeHist(gray)

def detect(img, cascade):
    rects = cascade.detectMultiScale(
			img, 
			scaleFactor=1.1,
            minNeighbors=5, 
			minSize=(30, 30)
			)
    print(rects)
    return rects

def draw_rects(img, rects, color):
    for x1, y1, w, h in rects:
        cv2.rectangle(img, (x1, y1), (x1+w, y1+h), color, 2)

rects = detect(gray, cascade)

vis = img.copy()
draw_rects(img, rects, (0, 255, 0))
cv2.imshow('facedetect', img)

cv2.waitKey(0)
