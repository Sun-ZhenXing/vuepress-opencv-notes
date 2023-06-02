import cv2
from yolo_segmentation import YOLOSegmentation

PHONE_CLASS_ID = 67

ys = YOLOSegmentation("yolov8m-seg.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    bboxes, classes, segmentations, scores = ys.detect(frame)
    mask = frame.copy()
    for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        if class_id != PHONE_CLASS_ID:
            continue
        # print("bbox:", bbox, "class id:", class_id, "seg:", seg, "score:", score)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.polylines(mask, [seg], True, (0, 0, 255), 4)
        cv2.fillPoly(mask, [seg], (0, 255, 0))
        cv2.putText(
            mask,
            f"{score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36, 255, 12),
            2,
        )

    frame = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
