import cv2
import numpy as np
from pathlib import Path


# ==========================================
# 自动寻找与本程序同目录下的 road.mp4
# ==========================================

video_path = Path(__file__).parent / "road.mp4"

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("无法打开视频：", video_path)
    exit()


# 保存上一帧车道线，用于减少抖动
previous_left = None
previous_right = None


def region_of_interest(image):
    """
    只保留自车前方道路区域
    """
    height, width = image.shape[:2]

    mask = np.zeros_like(image)

    polygon = np.array([[
        (int(width * 0.08), height),
        (int(width * 0.42), int(height * 0.60)),
        (int(width * 0.58), int(height * 0.60)),
        (int(width * 0.95), height)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)

    return cv2.bitwise_and(image, mask)


def make_lane_line(points, height, width, side):
    """
    利用多个候选线段的端点拟合出一条完整车道线
    """

    if len(points) < 4:
        return None

    points = np.array(points)

    x = points[:, 0]
    y = points[:, 1]

    # 用 x = a*y + b 拟合
    a, b = np.polyfit(y, x, 1)

    y_bottom = height - 1
    y_top = int(height * 0.60)

    x_bottom = int(a * y_bottom + b)
    x_top = int(a * y_top + b)

    # 再做一次位置合理性判断
    if side == "left":

        if not (0 < x_bottom < width * 0.55):
            return None

    if side == "right":

        if not (width * 0.45 < x_bottom < width):
            return None

    return np.array([
        x_bottom,
        y_bottom,
        x_top,
        y_top
    ], dtype=np.float32)


def smooth_line(current, previous, alpha=0.25):
    """
    对连续视频帧进行平滑，减少车道线抖动
    """

    if current is None:
        return previous

    if previous is None:
        return current

    return previous * (1 - alpha) + current * alpha


def detect_lane(frame):

    global previous_left
    global previous_right

    height, width = frame.shape[:2]

    # ==========================================
    # 1. 提取白色车道线
    # ==========================================

    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    lower_white = np.array([0, 170, 0])
    upper_white = np.array([255, 255, 255])

    white_mask = cv2.inRange(
        hls,
        lower_white,
        upper_white
    )


    # ==========================================
    # 2. 灰度化 + 高斯滤波
    # ==========================================

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


    # ==========================================
    # 3. Canny边缘检测
    # ==========================================

    edges = cv2.Canny(
        blur,
        50,
        150
    )


    # 只保留白色区域中的边缘
    edges = cv2.bitwise_and(
        edges,
        white_mask
    )


    # ==========================================
    # 4. ROI
    # ==========================================

    roi = region_of_interest(edges)


    # ==========================================
    # 5. 霍夫直线检测
    # ==========================================

    lines = cv2.HoughLinesP(
        roi,
        rho=2,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=30,
        maxLineGap=80
    )


    left_points = []
    right_points = []


    # ==========================================
    # 6. 候选车道线筛选
    # ==========================================

    if lines is not None:

        for x1, y1, x2, y2 in lines.reshape(-1, 4):

            dx = x2 - x1
            dy = y2 - y1

            # 接近竖直线，例如灯杆，直接排除
            if abs(dx) < 5:
                continue

            slope = dy / dx

            # 排除水平线和过于竖直的线
            if abs(slope) < 0.5:
                continue

            if abs(slope) > 2.5:
                continue

            center_x = (x1 + x2) / 2


            # 左车道
            if slope < 0 and center_x < width * 0.55:

                left_points.append((x1, y1))
                left_points.append((x2, y2))


            # 右车道
            elif slope > 0 and center_x > width * 0.45:

                right_points.append((x1, y1))
                right_points.append((x2, y2))


    # ==========================================
    # 7. 左右车道拟合
    # ==========================================

    left_line = make_lane_line(
        left_points,
        height,
        width,
        "left"
    )

    right_line = make_lane_line(
        right_points,
        height,
        width,
        "right"
    )


    # ==========================================
    # 8. 连续帧平滑
    # ==========================================

    previous_left = smooth_line(
        left_line,
        previous_left
    )

    previous_right = smooth_line(
        right_line,
        previous_right
    )


    # ==========================================
    # 9. 绘制
    # ==========================================

    overlay = np.zeros_like(frame)


    if previous_left is not None:

        x1, y1, x2, y2 = previous_left.astype(int)

        cv2.line(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            8
        )


    if previous_right is not None:

        x1, y1, x2, y2 = previous_right.astype(int)

        cv2.line(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            8
        )


    result = cv2.addWeighted(
        frame,
        1.0,
        overlay,
        0.9,
        0
    )

    return result, edges, roi


# ==========================================
# 主循环
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("视频播放结束")
        break


    result, edges, roi = detect_lane(frame)


    # 缩小显示
    display_width = 960

    scale = display_width / result.shape[1]

    display_result = cv2.resize(
        result,
        None,
        fx=scale,
        fy=scale
    )


    cv2.imshow(
        "Lane Detection - Q or ESC to exit",
        display_result
    )


    key = cv2.waitKey(25) & 0xFF


    # q / Q / ESC 均退出
    if key == ord("q"):
        break

    if key == ord("Q"):
        break

    if key == 27:
        break


    # 点击窗口 X 后退出
    if cv2.getWindowProperty(
        "Lane Detection - Q or ESC to exit",
        cv2.WND_PROP_VISIBLE
    ) < 1:
        break


cap.release()
cv2.destroyAllWindows()

print("程序已退出")