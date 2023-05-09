import cv2

from yolo_segmentation import YOLOSegmentation

img = cv2.imread("images/rugby.jpg")
img = cv2.resize(img, None, fx=0.7, fy=0.7)

ys = YOLOSegmentation("yolov8m-seg.pt")

bboxes, classes, segmentations, scores = ys.detect(img)
for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
    print("bbox:", bbox, "class id:", class_id, "seg:", seg, "score:", score)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.polylines(img, [seg], True, (0, 0, 255), 4)

cv2.imwrite("res.jpg", img)
