# Python OpenCV 支持 Unicode

[[TOC]]

如果在 `cv2` 中使用 Unicode 字符串，在某些系统上会无法读取。

C++ OpenCV 可以通过对应平台的编码来解决这个问题，但是 Python OpenCV 没有这个功能。

## 1. 使用 NumPy 读取

```python
import cv2
import numpy as np
 
def cv_imread(path: str, flags=cv2.IMREAD_COLOR):
    cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    return cv_img

def cv_imwrite(path: str, img: np.ndarray):
    buffer: np.ndarray = cv2.imencode('.jpg', img)[1]
    buffer.tofile(path)
```

## 2. 使用 PIL 读取

```python
import cv2
from PIL import Image

def pli_to_cv2(img: Image.Image):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def cv2_to_pli(img: np.ndarray):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def cv_imread(path: str, **kwargs):
    return pli_to_cv2(Image.open(path, **kwargs))

def cv_imwrite(path: str, img: np.ndarray):
    cv2_to_pli(img).save(path)
```
