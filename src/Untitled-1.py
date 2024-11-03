import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def _classify_lines(lines):
    horizontal = []
    vertical = []
    highest_vertical_y = np.inf
    lowest_vertical_y = 0
    lowest_vertical_y1 = np.inf
    lowest_vertical_x = np.inf
    xs = 0
    ys = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        if dx > 2 * dy:
            horizontal.append(line[0])
        else:
            vertical.append(line[0])
            highest_vertical_y = min(highest_vertical_y, y1, y2)
            lowest_vertical_y = max(lowest_vertical_y, y1, y2)
            if lowest_vertical_x > x1 and ys < y1:
                leftmost_vertical = line[0]
                lowest_vertical_x = x1
                ys = y1
            if lowest_vertical_y1 >= y1 and xs < x1:
                rightmost_vertical = line[0]
                lowest_vertical_y1 = y1
                xs = x1
    clean_horizontal = []
    h = lowest_vertical_y - highest_vertical_y
    lowest_vertical_y += h / 15
    highest_vertical_y -= h * 2 / 15
    for line in horizontal:
        x1, y1, x2, y2 = line
        if lowest_vertical_y > y1 > highest_vertical_y and lowest_vertical_y > y1 > highest_vertical_y:
            clean_horizontal.append(line)
    return clean_horizontal, vertical, leftmost_vertical, rightmost_vertical

def location(line):
    x1, y1, x2, y2 = line
    d = 50
    direction = np.array([x2 - x1, y2 - y1])
    length = np.linalg.norm(direction)
    unit_direction = direction / length
    perpendicular_direction = np.array([-unit_direction[1], unit_direction[0]])
    offset = d * perpendicular_direction
    pts_src = np.array([
        [x1 - offset[0], y1 + offset[1] + 50],
        [x2 - offset[0], y2 + offset[1] + 50],
        [x2 - offset[0] * 2, y2 - offset[1] * 2],
        [x1 - offset[0] * 2, y1 - offset[1] * 2]
    ])
    pts_src = pts_src.astype(int)
    return pts_src

def cam_pose(img_pts, real_pts, int_para=None, dist_coef=None, frame_size=None):
    x1, y1, x2, y2 = left
    x3, y3, x4, y4 = right
    image_pts = np.array([
        [x4, y4],
        [x3, y3],
        [x2, y2],
        [x1, y1]
    ], dtype=np.float32)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(real_pts, img_pts, int_para, dist_coef, iterationsCount=100, reprojectionError=1.0, confidence=0.99)
    image_pts2, _ = cv2.projectPoints(real_pts, rvec, tvec, intrinsic_par, dist_coeffs)
    image_pts2 = np.reshape(image_pts2.astype(int), [-1, 2])
    errors = np.linalg.norm(img_pts - image_pts2.squeeze(), axis=1)
    mean_error = np.mean(errors)
    return rvec, tvec

previous1 = None
previous2 = None
cap = cv2.VideoCapture('../input/tennis_1.mp4')
dist_tau = 3
intensity_threshold = 40
minLineLength = 300
maxLineGap = 60
ad_image = cv2.imread('../input/download.jpeg')
angle = 60

# Get the dimensions of the banner image
(h, w) = ad_image.shape[:2]
center = (w // 2, h // 2)

# Compute the rotation matrix for slanting the image
M = cv2.getRotationMatrix2D(center, angle, 1.0)
ad_image = cv2.warpAffine(ad_image, M, (w, h))

# Create a shadow effect
shadow = np.full_like(ad_image, (50, 50, 50))  # Gray shadow
shadow = cv2.warpAffine(shadow, M, (w, h))

# Get the height and width of the advertisement image
ad_height, ad_width = ad_image.shape[:2]
cam_matrix = np.load('../input/camera_matrix.npz')
intrinsic_par = cam_matrix['camera_mat']
dist_coeffs = cam_matrix['dist_coeffs']
c_lw_th = 50
c_hg_th = 80
smoothed_homography = 0
frame_counter = 0
smoothed_rvec = smoothed_tvec = 0
alpha = 0.1  # Smoothing factor
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

pts_dst = np.array([
    [0, 0],
    [0, ad_height],
    [ad_width, ad_height],
    [ad_width, 0]
])
ad_3D_coordinates = np.array([
    [12.77, 3, 0],
    [11.27, 3, 0],
    [11.27, 8, 0],
    [12.77, 8, 0]
], dtype=np.float32)

obj_points = np.array([
    [0, 0, 0],
    [0, 23.77, 0],
    [10.97, 23.77, 0],
    [10.97, 0, 0]
], dtype=np.float32)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, frame = cap.read()
gray = cv2.GaussianBlur(frame, (5, 5), 0)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
kernel = np.ones((4, 4), np.uint8)
gray = cv2.dilate(gray, kernel, iterations=1)
kernel = np.ones((3, 3), np.uint8)
gray = cv2.erode(gray, kernel, iterations=1)
lines = cv2.HoughLinesP(gray, 1, np.pi / 180, threshold=120, minLineLength=minLineLength, maxLineGap=maxLineGap)
horizontal, vertical, left, right = _classify_lines(lines)

x1, y1, x2, y2 = left
x3, y3, x4, y4 = right
p0 = np.array([
    [x4, y4],
    [x3, y3],
    [x2, y2],
    [x1, y1]
], dtype=np.float32)
p0 = p0.reshape(-1, 1, 2)
old_gray = gray
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('Wrong Path')
        break

    gray = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((4, 4), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.erode(gray, kernel, iterations=1)
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, threshold=120, minLineLength=minLineLength, maxLineGap=maxLineGap)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    old_gray = gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    img_pts = good_new
    print(img_pts)
    horizontal, vertical, left, right = _classify_lines(lines)
    for line in horizontal:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for line in vertical:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    new_rvec, new_tvec = cam_pose(img_pts, obj_points, intrinsic_par, dist_coeffs)
    smoothed_rvec = (1 - alpha) * smoothed_rvec + alpha * new_rvec
    smoothed_tvec = (1 - alpha) * smoothed_tvec + alpha * new_tvec
    ad_corners, _ = cv2.projectPoints(ad_3D_coordinates, smoothed_rvec, smoothed_tvec, intrinsic_par, dist_coeffs)
    pts_dst = np.reshape(ad_corners, (4, 2)).astype(int)

    # for i in range(len(ad_corners)):
    #     cv2.circle(frame, tuple(ad_corners[i][0]), 10, (0, 255, 255), -1)

    
    h, status = cv2.findHomography(pts_src, pts_dst)

    pts_src_shadow = pts_src + np.array([20, 20])
    h_shadow, _ = cv2.findHomography(pts_src_shadow, pts_dst)
    shadow_image = cv2.warpPerspective(shadow, h_shadow, (frame.shape[1], frame.shape[0]))
    shadow_mask = np.zeros_like(frame)
    cv2.fillPoly(shadow_mask, [pts_dst], (255, 255, 255))
    shadow_mask = cv2.cvtColor(shadow_mask, cv2.COLOR_BGR2GRAY)
    shadow_image = cv2.bitwise_and(shadow_image, shadow_image, mask=shadow_mask)
    shadow_image = cv2.GaussianBlur(shadow_image, (25, 25), 0)

    img_warped = cv2.warpPerspective(ad_image, h, (frame.shape[1], frame.shape[0]))

    for i in range(len(img_warped)):
        for j in range(len(img_warped[i])):
            if not np.array_equal(img_warped[i][j], np.zeros(3)):
                frame[i][j] = img_warped[i][j]
            if not np.array_equal(shadow_image[i][j], np.zeros(3)):
                frame[i][j] = shadow_image[i][j]
    
    cv2.polylines(frame, [pts_src], isClosed=True, color=(0, 255, 0), thickness=3)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
