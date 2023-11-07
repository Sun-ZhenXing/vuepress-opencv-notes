import os

import cv2
import numpy as np
import onnxruntime

_CLASS_COLOR_MAP = [
    (0, 0, 255),  # Person (blue).
    (255, 0, 0),  # Bear (red).
    (0, 255, 0),  # Tree (lime).
    (255, 0, 255),  # Bird (fuchsia).
    (0, 255, 255),  # Sky (aqua).bbbbbbb
    (255, 255, 0),  # Cat (yellow).
]
palette = np.array(
    [
        [255, 128, 0],
        [255, 153, 51],
        [255, 178, 102],
        [230, 230, 0],
        [255, 153, 255],
        [153, 204, 255],
        [255, 102, 255],
        [255, 51, 255],
        [102, 178, 255],
        [51, 153, 255],
        [255, 153, 153],
        [255, 102, 102],
        [255, 51, 51],
        [153, 255, 153],
        [102, 255, 102],
        [51, 255, 51],
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [255, 255, 255],
    ]
)

skeleton = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]

pose_limb_color = palette[
    [9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]
]
pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
radius = 5

_cache_session = None


def preprocess_image(img: np.ndarray, img_mean=0, img_scale=1 / 255):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    img = (img - img_mean) * img_scale
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img, 0)
    img = img.transpose(0, 3, 1, 2)
    return img


def model_inference(model_path="./yolov7-w6-pose.onnx", input=None):
    global _cache_session
    if _cache_session is None:
        _cache_session = onnxruntime.InferenceSession(model_path, None)
    input_name = _cache_session.get_inputs()[0].name
    output = _cache_session.run([], {input_name: input})
    return output


def post_process(img: np.ndarray, output: np.ndarray, score_threshold=0.3):
    h, w, _ = img.shape
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    det_bboxes, det_scores, det_labels, kpts = (
        output[:, 0:4],
        output[:, 4],
        output[:, 5],
        output[:, 6:],
    )
    for idx in range(len(det_bboxes)):
        det_bbox = det_bboxes[idx]
        kpt = kpts[idx]
        # print(det_labels[idx], kpt, det_bbox)
        if det_scores[idx] > score_threshold:
            color_map = _CLASS_COLOR_MAP[int(det_labels[idx])]
            img = cv2.rectangle(
                img,
                (int(det_bbox[0]), int(det_bbox[1])),
                (int(det_bbox[2]), int(det_bbox[3])),
                color_map[::-1],
                2,
            )
            cv2.putText(
                img,
                "id:{}".format(int(det_labels[idx])),
                (int(det_bbox[0] + 5), int(det_bbox[1]) + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_map[::-1],
                2,
            )
            cv2.putText(
                img,
                "score:{:2.1f}".format(det_scores[idx]),
                (int(det_bbox[0] + 5), int(det_bbox[1]) + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_map[::-1],
                2,
            )
            plot_skeleton_kpts(img, kpt)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img, kpts


def plot_skeleton_kpts(img: np.ndarray, kpts, steps=3):
    num_kpts = len(kpts) // steps
    # plot keypoints
    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        conf = kpts[steps * kid + 2]
        if conf > 0.5:  # Confidence of a keypoint has to be greater than 0.5
            cv2.circle(
                img, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1
            )
    # plot skeleton
    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0] - 1) * steps]), int(kpts[(sk[0] - 1) * steps + 1]))
        pos2 = (int(kpts[(sk[1] - 1) * steps]), int(kpts[(sk[1] - 1) * steps + 1]))
        conf1 = kpts[(sk[0] - 1) * steps + 2]
        conf2 = kpts[(sk[1] - 1) * steps + 2]
        if (
            conf1 > 0.5 and conf2 > 0.5
        ):  # For a limb, both the keypoint confidence must be greater than 0.5
            cv2.line(img, pos1, pos2, (int(r), int(g), int(b)), thickness=2)


def infer_video(video_path: str | int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img = preprocess_image(frame)
            output = model_inference(input=img)[0]
            res, kpts = post_process(frame, output)
            cv2.imshow("frame", res)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def build_train_data():
    import pandas as pd

    cols = []
    for p in range(1, 18):
        cols.append("x{}".format(p))
        cols.append("y{}".format(p))
        cols.append("c{}".format(p))
    data = pd.DataFrame(columns=cols)
    i = 0
    data_path = "train"
    for f in os.listdir(f"./data/{data_path}"):
        img_src = cv2.imread(f"./data/{data_path}/{f}")
        img = preprocess_image(img_src)
        output = model_inference(input=img)[0]
        res, kpts = post_process(img_src, output)
        if kpts.size > 0:
            data.loc[i] = kpts[0]  # type: ignore
            i += 1
        cv2.imshow("frame", res)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None
    data.to_csv(f"./data/{data_path}.csv", index=False)


def main():
    infer_video(0)
    # build_train_data()


if __name__ == "__main__":
    main()
