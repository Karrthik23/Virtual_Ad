#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
#%%
cap = cv2.VideoCapture('../input/tennis_1.mp4')
dist_tau =3
intensity_threshold = 40
minLineLength = 200
maxLineGap = 50
ad_image = cv2.imread('../input/num.png')
ad_height, ad_width = ad_image.shape[:2]
pts_dst = np.array([
    [0, 0],                  # Top-left corner
    [ad_width, 0],           # Top-right corner
    [ad_width, ad_height],   # Bottom-right corner
    [0, ad_height]           # Bottom-left corner
])
def _classify_lines(lines):
        """
        Classify line to vertical and horizontal lines
        """
        horizontal = []
        vertical = []
        highest_vertical_y = np.inf
        lowest_vertical_y = 0
        lowest_vertical_x = np.inf
        i=0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            if dx > 2* dy:
                # print('horizontal')
                horizontal.append(line[0])
            else:
                # print('vertical')
                vertical.append(line[0])
                highest_vertical_y = min(highest_vertical_y, y1, y2)
                lowest_vertical_y = max(lowest_vertical_y, y1, y2)
                if lowest_vertical_x > x1:
                     leftmost_vertical = line[0]
                     lowest_vertical_x = x1
        clean_horizontal = []
        h = lowest_vertical_y - highest_vertical_y
        lowest_vertical_y += h / 15
        highest_vertical_y -= h * 2 / 15
        for line in horizontal:
            x1, y1, x2, y2 = line
            if lowest_vertical_y > y1 > highest_vertical_y and lowest_vertical_y > y1 > highest_vertical_y:
                clean_horizontal.append(line)
        return clean_horizontal, vertical,leftmost_vertical
        # return horizontal, vertical
def location(line):
    x1,y1,x2,y2 = line
    x1, y1, x2, y2 = 339, 874, 471, 578

    # Distance from the line where you want to place the advertisement
    d = 50  # Example distance in pixels

    # Calculate the direction vector of the line
    direction = np.array([x2 - x1, y2 - y1])  # (dx, dy)

    # Calculate the length of the direction vector
    length = np.linalg.norm(direction)

    # Normalize the direction vector to get the unit direction
    unit_direction = direction / length

    # Calculate the perpendicular direction
    perpendicular_direction = np.array([-unit_direction[1], unit_direction[0]])  # Swap and negate y

    # Calculate the offset for the desired distance
    offset = d * perpendicular_direction

    # Calculate new source points (pts_src)
    pts_src = np.array([
        [x1 - offset[0], y1 + offset[1]+50],  # Adjusted point 1
        [x2 - offset[0], y2 + offset[1]+50],  # Adjusted point 2
        [x2 - offset[0]*2, y2 - offset[1]*2],  # Adjusted point 3
        [x1 - offset[0]*2, y1 - offset[1]*2]   # Adjusted point 4
    ])
    pts_src = pts_src.astype(int)
    return pts_src

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        print('Wrong Path')
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('before',gray)
    edges = cv2.Canny(gray,100,150,apertureSize=3)
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 80, minLineLength=minLineLength, maxLineGap=maxLineGap)
    
    horizontal, vertical,leftmost_vertical = _classify_lines(lines)
    for line in horizontal:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) 

    for line in vertical:
        x1, y1, x2, y2 = leftmost_vertical
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) 
    a = location(leftmost_vertical)
    pts_src =a
    print(a)
    x_1,y_1 =a[0]
    x_2,y_2 =a[1]
    x_3,y_3 =a[2]
    x_4,y_4 =a[3]
    cv2.line(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 2) 
    cv2.line(frame, (x_2, y_2), (x_3, y_3), (0, 255, 0), 2) 
    cv2.line(frame, (x_3, y_3), (x_4, y_4), (0, 255, 0), 2) 
    cv2.line(frame, (x_4, y_4), (x_1, y_1), (0, 255, 0), 2) 
    h, status = cv2.findHomography(pts_dst, pts_src)
    frame_height, frame_width = frame.shape[:2]
    ad_warped = cv2.warpPerspective(ad_image, h, (frame_width, frame_height))
    
    # Create a mask for the advertisement
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts_src.astype(int), 255)

    # Mask the frame to remove the area where the ad will go
    masked_frame = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

    # Combine the warped advertisement with the masked frame
    result = cv2.add(masked_frame, ad_warped)

    cv2.imshow('before',result)
    cv2.imshow('after',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
# %%
cv2.destroyAllWindows()
# %%
