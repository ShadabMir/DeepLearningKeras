from ultralytics import YOLO
import cv2 as cv
model = YOLO('yolov8n.pt')
cam = cv.VideoCapture(0)
while cam.isOpened():
    ret,frame = cam.read()
    if ret:
        results = model(source=frame)
        anotate_frame = results[0].plot()
        cv.imshow("Yolov8",anotate_frame)
        if cv.waitKey(1) == ord('q'):
            break
    else:
        break
cam.release()
cv.destroyAllWindows()