# https://pysource.com/2023/02/21/yolo-v8-segmentation
import numpy as np
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results

class YOLOSegmentation:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, img: np.ndarray):
        height, width, _ = img.shape
        results = self.model.predict(
            source=img.copy(),
            save=False,
            save_txt=False
        )
        result: Results = results[0]
        segmentation_contours_idx = []
        for seg in result.masks.segments:
            # contours
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)
            segmentation_contours_idx.append(segment)

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        # Get class ids
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        # Get scores
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
        return bboxes, class_ids, segmentation_contours_idx, scores
