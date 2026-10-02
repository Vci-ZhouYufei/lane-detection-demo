import cv2
import numpy as np


# ==============================
# 1. 创建一张模拟道路图像
# ==============================

width = 1280
height = 720

image = np.zeros((height, width, 3), dtype=np.uint8)

# 道路背景
image[:] = (60, 60, 60)

# 左车道线
cv2.line(
    image,
    (300, 700),
    (560, 400),
    (255, 255, 255),
    10
)

# 右车道线
cv2.line(
    image,
    (980, 700),
    (720, 400),
    (255, 255, 255),
    10
)


# ==============================
# 2. 灰度化
# ==============================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ==============================
# 3. 高斯滤波
# ==============================

blur = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)


# ==============================
# 4. Canny边缘检测
# ==============================

edges = cv2.Canny(
    blur,
    50,
    150
)


# ==============================
# 5. 设置感兴趣区域 ROI
# ==============================

mask = np.zeros_like(edges)

polygon = np.array([[
    (100, height),
    (1180, height),
    (750, 350),
    (530, 350)
]], dtype=np.int32)

cv2.fillPoly(mask, polygon, 255)

roi = cv2.bitwise_and(
    edges,
    mask
)


# ==============================
# 6. 霍夫直线检测
# ==============================

lines = cv2.HoughLinesP(
    roi,
    rho=2,
    theta=np.pi / 180,
    threshold=50,
    minLineLength=80,
    maxLineGap=100
)


# ==============================
# 7. 将检测结果画到图像上
# ==============================

result = image.copy()

if lines is not None:

    for x1, y1, x2, y2 in lines.reshape(-1, 4):

        cv2.line(
            result,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            5
        )


# ==============================
# 8. 显示结果
# ==============================

cv2.imshow("Original Road", image)
cv2.imshow("Edge Detection", edges)
cv2.imshow("Lane Detection", result)

# 同时保存结果
cv2.imwrite("lane_result.png", result)

print("车道线检测完成")
print("结果已保存为 lane_result.png")

cv2.waitKey(0)
cv2.destroyAllWindows()