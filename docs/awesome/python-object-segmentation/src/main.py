import cv2
import numpy as np


def cluster(img: np.ndarray, k: int):
    """聚类实现目标分割
    @param `img`: 图像
    @param `k`: 聚类数
    """
    # 图像尺寸
    h, w = img.shape[:2]
    # 将图像转换为二维矩阵
    data = img.reshape((h * w, 3))
    # 转换为浮点型
    data = np.float32(data)
    # 定义停止条件
    criteria = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0
    # 聚类
    ret, label, center = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    # 转换回uint8
    center = np.uint8(center)
    # 分割图像
    res = center[label.flatten()]
    return res.reshape((img.shape))


def threshold(img: np.ndarray):
    """自适应阈值分割实现目标分割
    @param `img`: 图像
    """
    # 图像灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 阈值分割
    ret, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    return binary


def main():
    img = cv2.imread('test.png')
    cv2.imshow('img', img)
    # 聚类分割
    res_cluster = cluster(img, 3)
    # 显示图像
    cv2.imshow('res', res_cluster)
    # 阈值分割
    res_threshold = threshold(img)
    cv2.imshow('res2', res_threshold)
    # 等待显示
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
