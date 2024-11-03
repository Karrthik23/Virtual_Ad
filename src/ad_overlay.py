#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
#%%

def _classify_lines(lines):
        """
        Classify line to vertical and horizontal lines
        """
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
            if dx > 2* dy:
                horizontal.append(line[0])
            else:
                vertical.append(line[0])
                highest_vertical_y = min(highest_vertical_y, y1, y2)
                lowest_vertical_y = max(lowest_vertical_y, y1, y2)
                if lowest_vertical_x > x1 and ys<y1:
                     leftmost_vertical = line[0]
                     lowest_vertical_x = x1
                     ys =y1
                if lowest_vertical_y1 >= y1 and xs<x1:
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
        return clean_horizontal, vertical,leftmost_vertical, rightmost_vertical
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

def cam_pose(img_pts,real_pts,int_para=None,dist_coef=None,frame_size=None):
    x1,y1,x2,y2 = left
    x3,y3,x4,y4 = right
    #origin for cam matrix calculations start with bottom left
    image_pts = np.array([
         [x4,y4],           # bottom left
         [x3,y3],           # top left
         [x2,y2],           # bottom right
         [x1,y1]  
         
    ],dtype=np.float32)
    
    success, rvec, tvec, inliers = cv2.solvePnPRansac(real_pts, img_pts, int_para, dist_coef,
                                                iterationsCount=100, reprojectionError=1.0, confidence=0.99)
    image_pts2, _ = cv2.projectPoints(real_pts, rvec, tvec, intrinsic_par, dist_coeffs)
    image_pts2 = np.reshape(image_pts2.astype(int),[-1,2])
    errors = np.linalg.norm(img_pts - image_pts2.squeeze(), axis=1)
    mean_error = np.mean(errors)
    return rvec,tvec

def draw_banner_with_shadow(image_pts, shadow_length, light_direction):
    # Calculate shadow points based on the light direction
    shadow_points = []
    for point in image_pts:
        shadow_point = point + light_direction * shadow_length
        shadow_points.append(shadow_point)

    # Convert to numpy array for drawing
    shadow_points = np.array(shadow_points, dtype=np.int32)
    return shadow_points
    # Draw the shadow (filled polygon)
    cv2.fillPoly(frame, [shadow_points], (0, 0, 0))  # Black shadow

    # Draw the banner (can be a different color)
    cv2.polylines(frame, [image_pts], isClosed=True, color=(0, 255, 0), thickness=2)  # Green banner

previous1 = None
previous2 = None
cap = cv2.VideoCapture('../input/tennis_1.mp4')
dist_tau =3
intensity_threshold = 40
minLineLength = 200
maxLineGap = 60
ad_image = cv2.imread('../input/download.jpeg')
ad_image1 = cv2.imread('../input/num.png')
ad_height, ad_width = ad_image.shape[:2]
cam_matirx = np.load('../input/camera_matrix.npz')
intrinsic_par = cam_matirx['camera_mat']
dist_coeffs = cam_matirx['dist_coeffs']
c_lw_th = 50
c_hg_th = 80
smoothed_homography = 0
frame_counter = 0
smoothed_rvec=smoothed_tvec=0
alpha = 0.1  # Smoothing factor
lk_params = dict(winSize=(21, 21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

pts_dst = np.array([
    [0, 0],                  # Top-left corner
    [0, ad_height],           # Top-right corner
    [ad_width, ad_height],   # Bottom-right corner
    [ad_width,0]           # Bottom-left corner
])
ad_3D_coordinates = np.array([
     [13.77,6,0],   # top left
     [12.77,6,4],       # top right
     [12.77,12,4],     # bottom right
     [13.77,12,0]      # bottom left
],dtype=np.float32)
offset = 0.5
shadow_offset_x = 0.5  # Horizontal offset to the right
shadow_offset_y = 0.5  # Horizontal offset forward/backward
shadow_height = -0.1    # Slightly below the ground level

# Calculate shadow vertices by applying offsets to the original coordinates
shadow_vertices = np.array([
    [ad_3D_coordinates[0][0] + shadow_offset_x, ad_3D_coordinates[0][1] + shadow_offset_y, shadow_height],  # top left shadow
    [ad_3D_coordinates[1][0] + shadow_offset_x, ad_3D_coordinates[1][1] + shadow_offset_y, shadow_height],  # top right shadow
    [ad_3D_coordinates[2][0] + shadow_offset_x, ad_3D_coordinates[2][1] + shadow_offset_y, shadow_height],  # bottom right shadow
    [ad_3D_coordinates[3][0] + shadow_offset_x, ad_3D_coordinates[3][1] + shadow_offset_y, shadow_height]   # bottom left shadow
], dtype=np.float32)
shadow_3D = np.array([
     [13.77,6,1],   # top left
     [12.77,6,0],       # top right
     [12.77,12,0],     # bottom right
     [13.77,12,1]      # bottom left
],dtype=np.float32)
#court lines 
obj_points = np.array([
    [0,0,0],            #1  bottom left
    [0,23.77,0],        #2  top left
    ## top points ##
    [10.97,23.77,0],    #4   top right
    [10.97,0,0]       #3   bottom right   
],dtype=np.float32)

    # Define light direction and shadow length
light_direction = np.array([1, -1,0])  # Direction of light (e.g., to the right and downwards)
shadow_length = 50  # Adjust based on desired shadow length

# Draw the banner with shadow
# print(shadow_3D_coordinates)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret,frame = cap.read()
gray = cv2.GaussianBlur(frame,(5,5), 0)
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
kernel = np.ones((4, 4), np.uint8)
gray = cv2.dilate(gray, kernel, iterations=1)
kernel = np.ones((3, 3), np.uint8)
gray = cv2.erode(gray, kernel, iterations=1)
lines = cv2.HoughLinesP(gray, 1, np.pi / 180, threshold=120, minLineLength=minLineLength, maxLineGap=maxLineGap)
horizontal, vertical,left,right= _classify_lines(lines)

x1,y1,x2,y2 = left
x3,y3,x4,y4 = right
p0 = np.array([
         [x4,y4],           # bottom left
         [x3,y3],           # top left
         [x2,y2],           # bottom right
         [x1,y1]  
        ],dtype=np.float32)
p0 = p0.reshape(-1, 1, 2)
old_gray = gray
while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        print('Wrong Path')
        break
    
    gray = cv2.GaussianBlur(frame,(5,5), 0)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
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
    horizontal, vertical,left,right= _classify_lines(lines)
    for line in horizontal:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) 

    for line in vertical:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) 
    

    
    new_rvec,new_tvec = cam_pose(img_pts,obj_points,intrinsic_par,dist_coeffs)
    
    image_pts, _ = cv2.projectPoints(ad_3D_coordinates, new_rvec, new_tvec, intrinsic_par, dist_coeffs)
    image_pts = np.reshape(image_pts.astype(int),[-1,2])
    h, status = cv2.findHomography(pts_dst, image_pts,cv2.RANSAC,5)
    
    smoothed_homography = alpha * h + (1 - alpha) * smoothed_homography
    frame_height, frame_width = frame.shape[:2]
    ad_warped = cv2.warpPerspective(ad_image, h, (frame_width, frame_height))
    
    print('image points',image_pts)
    shadow_offset_x = 100  # Horizontal offset to the right
    shadow_offset_y = 100   # Vertical offset downward (this creates the shadow depth)
    shadow_depth = 20      # Additional depth behind the banner

    # Calculate shadow points by applying offsets
    shadow_points = image_pts.copy()
    print(shadow_points[0,0])
    shadow_points[0] = shadow_points[1]
    shadow_points[3] = shadow_points[2]
    shadow_points[0,0] = shadow_points[0,0]-shadow_offset_x
    shadow_points[3,0] = shadow_points[2,0]-shadow_offset_x
    print('shadow_pts',shadow_points)
    shadow_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(shadow_mask, shadow_points.astype(np.int32), 255)  # Create shadow mask

    # Apply Gaussian blur to the shadow mask to create a soft edge
    shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)

    # Create a color shadow using the blurred mask
    shadow_color = (0, 0, 0)  # Black shadow
    shadow_bgr = np.zeros_like(frame)
    shadow_bgr[:, :] = shadow_color
    shadow_bgr = cv2.bitwise_and(shadow_bgr, shadow_bgr, mask=shadow_mask)
    # Blend the shadow onto the frame using the mask
    frame = cv2.addWeighted(frame, 1.0, shadow_bgr, 0.5, 0)  # Adjust alpha for shadow intensity
    frame = cv2.add(frame, cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR))
    # cv2.fillConvexPoly(frame, shadow_points, (0, 0, 0), lineType=cv2.LINE_AA)  # Black shadow
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillConvexPoly(mask, image_pts, (255, 255, 255))
    frame = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
    frame = cv2.bitwise_or(frame, ad_warped)
    # image_pts, _ = cv2.projectPoints(shadow_3D, new_rvec, new_tvec, intrinsic_par, dist_coeffs)
    # image_pts = np.reshape(image_pts.astype(int),[-1,2])
    # h, status = cv2.findHomography(pts_dst, image_pts,cv2.RANSAC,5)
    # ad_warped1 = cv2.warpPerspective(ad_image, h, (frame_width, frame_height))
    # cv2.fillConvexPoly(frame, image_pts, (0, 0, 0), lineType=cv2.LINE_AA) 
    # cv2.fillConvexPoly(mask, image_pts, (255, 255, 255))
    # frame = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
    # frame = cv2.bitwise_or(frame, ad_warped1)
    


    cv2.imshow('before',gray)
    cv2.imshow('after',frame)
    key = cv2.waitKey(25)
    if key == ord('p'):
        print("Paused. Press any key to continue...")
        cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        break
# %%
cv2.destroyAllWindows()
# %%
