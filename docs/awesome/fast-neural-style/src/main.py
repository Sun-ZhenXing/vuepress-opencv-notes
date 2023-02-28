import cv2
import numpy as np


def process(image_path: str, model_path: str):
    net = cv2.dnn.readNetFromTorch(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False
    )
    net.setInput(blob)
    out: np.ndarray = net.forward()
    out = out.reshape(3, out.shape[2], out.shape[3])
    out[0] += 103.939
    out[1] += 116.779
    out[2] += 123.68
    out /= 255
    out = out.transpose(1, 2, 0)
    return out


model_list = [
    './models/eccv16/composition_vii.t7',
    './models/eccv16/la_muse.t7',
    './models/eccv16/starry_night.t7',
    './models/eccv16/the_wave.t7',
    './models/instance_norm/candy.t7',
    './models/instance_norm/feathers.t7',
    './models/instance_norm/la_muse.t7',
    './models/instance_norm/mosaic.t7',
    './models/instance_norm/the_scream.t7',
    './models/instance_norm/udnie.t7',
]

if __name__ == '__main__':
    image_path = 'test.jpg'
    for model_path in model_list:
        out = process(image_path, model_path)
        print(model_path)
        cv2.imshow('out', out)
        key = cv2.waitKey()
        if key == 27:
            break
