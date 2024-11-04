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
             # to classify horizontal and vertical lines 
             # dx>2*dy is used to ensure that only lines with very high dx
             # are classified as horizontal
            if dx > 2* dy:
                horizontal.append(line[0])
            else:
                vertical.append(line[0])
                #based on the identified vertical lines store the min and max y-pixel
                highest_vertical_y = min(highest_vertical_y, y1, y2)
                lowest_vertical_y = max(lowest_vertical_y, y1, y2)
                #   find the left vertical court line
                if lowest_vertical_x > x1 and ys<y1:
                     leftmost_vertical = line[0]
                     lowest_vertical_x = x1
                     ys =y1
                #   find the right vertical court line
                if lowest_vertical_y1 >= y1 and xs<x1:
                    rightmost_vertical = line[0]
                    lowest_vertical_y1 = y1
                    xs = x1
        clean_horizontal = []
        #since the court layout is known to us, we ensure only horizontal lines
        # between min and max vertical y-pixel [found earlier] are stored
        h = lowest_vertical_y - highest_vertical_y
        lowest_vertical_y += h / 15
        highest_vertical_y -= h * 2 / 15
        for line in horizontal:
            x1, y1, x2, y2 = line
            if lowest_vertical_y > y1 > highest_vertical_y and lowest_vertical_y > y1 > highest_vertical_y:
                clean_horizontal.append(line)
        return clean_horizontal, vertical,leftmost_vertical, rightmost_vertical
        # return horizontal, vertical, extreme right and left court lines

def detect_lines(ip_frame,minlen,maxlen,threshold):
    # GaussianBlur to reduce noise
    gray = cv2.GaussianBlur(ip_frame,(5,5), 0)
    #converting to grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # thresholding  to highlight the white court lines
    gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    # applying closing morphological operation -> dilation followed by erode
    kernel = np.ones((4, 4), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.erode(gray, kernel, iterations=1)
    # Hough Line Transform to get coordinates for lines
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, threshold=threshold, minLineLength=minlen, maxLineGap=maxlen)
    return lines

def cam_pose(img_pts,real_pts,int_para=None,dist_coef=None):   
    #img_pts: 2D image coordinates
    #real_pts: 3D world coordinates for the same points
    #int_para: camera matrix-> intrensic parameters
    #dist_coef: distortino coefficients
    #fr
    success, rvec, tvec, inliers = cv2.solvePnPRansac(real_pts, img_pts, int_para, dist_coef,
                                                iterationsCount=100, reprojectionError=1.0, confidence=0.99)
    # reprojection error calculations
    # image_pts2, _ = cv2.projectPoints(real_pts, rvec, tvec, intrinsic_par, dist_coeffs)
    # image_pts2 = np.reshape(image_pts2.astype(int),[-1,2])
    # errors = np.linalg.norm(img_pts - image_pts2.squeeze(), axis=1)
    # mean_error = np.mean(errors)
    return rvec,tvec

def shadow_plot(ip_frame,shadow_points,height=None,width = None):
    shadow_offset_x = 150   # horizontal offset 
    shadow_offset_y = 20    # vertical offset 
    shadow_offset_z = 10    # additional depth behind the banner

    # calculating the shadow pts based on image points
    shadow_points[0] = shadow_points[1]
    shadow_points[3] = shadow_points[2]
    shadow_points[0,0] = shadow_points[0,0]-shadow_offset_x
    shadow_points[3,0] = shadow_points[2,0]-shadow_offset_x
    shadow_points[0,1] = shadow_points[0,1]+shadow_offset_y
    shadow_points[3,1] = shadow_points[2,1]+shadow_offset_y
    shadow_points[1,1] = shadow_points[1,1]+shadow_offset_z
    shadow_points[2,1] = shadow_points[2,1]+shadow_offset_z
    # creating 2 shadow_masks
    # shadow_mask   --> used to remove the region of interest from original frame
    # shadow_mask2  --> to add the shadow to the video frame
    shadow_mask = np.zeros((height, width, 3), dtype=np.uint8)
    shadow_mask1 = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.fillConvexPoly(shadow_mask, shadow_points.astype(np.int32), (50, 50,50))
    cv2.fillConvexPoly(shadow_mask1, shadow_points.astype(np.int32), (5, 5,5))  
    # blur to blend the shadow with the background and round the edges
    shadow_mask = cv2.GaussianBlur(shadow_mask, (35, 35), 0)
    # mask for blending
    # Normalize to [0, 1] for later bitwise operations
    shadow_alpha = shadow_mask[..., 0] / 255.0  
    # to make it 3-channel as video frame is 3 channel
    shadow_alpha = shadow_alpha[..., np.newaxis]
    # blending the shadow with the frame
    blended = frame * (1 - shadow_alpha) + shadow_mask1 * shadow_alpha  
    # converting back to uint8 to match frame format
    out_frame = np.clip(blended, 0, 255).astype(np.uint8)

    return out_frame
# loading  video and ad banner
cap = cv2.VideoCapture('../input/tennis_1.mp4')
ad_image = cv2.imread('../input/download.jpeg')
# extracting the height and width of ad banner 
# used later
ad_height, ad_width = ad_image.shape[:2]
cam_matirx = np.load('../input/camera_matrix.npz')
# cam intrinsic parameters
intrinsic_par = cam_matirx['camera_mat']
# distortion coefficients
dist_coeffs = cam_matirx['dist_coeffs']
# extracting frame height and width
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# parameters for hough line transform
minLineLength = 200
maxLineGap = 60
# smoothed homography
smoothed_homography = 0
# Smoothing factor for homography
alpha = 0.1  
# dimensions/2D coordinates of the banner 
ad_2d_pts = np.array([
    [0, 0],                    # Top-left corner
    [0, ad_height],            # Top-right corner
    [ad_width, ad_height],     # Bottom-right corner
    [ad_width,0]               # Bottom-left corner
])
# 3D coordinates of banner w.r.t origin
# origin of world frame is located at the bottm right corner of court
ad_3D_coordinates = np.array([
     [13.77,6,0],       # top left
     [12.77,6,4],       # top right
     [12.77,12,4],      # bottom right
     [13.77,12,0]       # bottom left
],dtype=np.float32)
# 3D coordinates of court lines [outer court lines]
court_3D_pts = np.array([
    [0,0,0],            #1  bottom right [origin of world frame]
    [0,23.77,0],        #2  top right
    [10.97,23.77,0],    #4  top left
    [10.97,0,0]         #3  bottom left   
],dtype=np.float32)
# read the first frame
ret,frame = cap.read()
# detect lines in the frame
lines = detect_lines(frame,minLineLength, maxLineGap,threshold=120)
# classify lines 
horizontal, vertical,left,right= _classify_lines(lines)
# extract pixel coordinates from the identified court edges
x1,y1,x2,y2 = left
x3,y3,x4,y4 = right
# store to track them later with optic Flow method
p0 = np.array([
         [x4,y4],           # bottom left
         [x3,y3],           # top left
         [x2,y2],           # bottom right
         [x1,y1]  
        ],dtype=np.float32)
p0 = p0.reshape(-1, 1, 2)
# store the current greyscale version of frame: Optic Flow
old_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# parameters for Lukas Kenede Optic flow 
lk_params = dict(winSize=(21, 21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        print('Wrong Path')
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ################## Points tracking ###################
    # optic flow to track the corner of outer court dimensions
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    old_gray = gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    img_pts = good_new
    #!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ############### Court Line Detection #################
    # detecting lines
    lines = detect_lines(frame,minLineLength, maxLineGap,threshold=120)
    #classify lines
    horizontal, vertical,left,right= _classify_lines(lines)
    #plotting lines on the frame
    for line in horizontal:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 4) 
    for line in vertical:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 4) 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ############### Ad Banner Overlay #####################
    new_rvec,new_tvec = cam_pose(img_pts,court_3D_pts,intrinsic_par,dist_coeffs)
    # projecting world frame to 2D frame
    image_pts, _ = cv2.projectPoints(ad_3D_coordinates, new_rvec, new_tvec, intrinsic_par, dist_coeffs)
    image_pts = np.reshape(image_pts.astype(int),[-1,2])
    # finding Homography matrix H --> used later for transforming banner edge points to projected points
    # ad_2d_pts   --> 2D dimensions of the corner of ad banner
    # image_pts --> projected points of the banner based on camera pose
    h, status = cv2.findHomography(ad_2d_pts, image_pts,cv2.RANSAC,5)
    # smoothing the homography matrix to reduce noise and distortions between frames
    smoothed_homography = alpha * h + (1 - alpha) * smoothed_homography
    frame_height, frame_width = frame.shape[:2]
    # warping the banner to image coordinates [2D]
    ad_warped = cv2.warpPerspective(ad_image, h, (frame_width, frame_height))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ############### SHADOW FOR AD BANNER ##################

    frame = shadow_plot(frame,image_pts.copy(),height,width)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ############### DISPLAYING AD BANNER ##################
    # creating mask
    mask = np.zeros_like(frame, dtype=np.uint8)
    # to create a region of interest from projected img_pts
    cv2.fillConvexPoly(mask, image_pts, (255, 255, 255))
    # removing the region of interest from frame
    frame = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
    # adding banner overlay to the frame    --> gets added in previously removed region 
    frame = cv2.bitwise_or(frame, ad_warped)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # showing output of frame
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
