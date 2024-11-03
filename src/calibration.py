import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
print(os.getcwd())
cap = cv2.VideoCapture('input/tennis.mp4')
print('adsa')
image = cv2.imread('../frame_1.png')
## image pts ##
image_size = (image.shape[1], image.shape[0])

cv2.imshow("image",image)
img_points = np.array([
    [357,848],          #1
    [515,845],          #2
    [581,640],          #3
    [969,638],          #4
    [1356,637],         #5
    [1428,842],         #6
    [1585,843],         #7
    [1311,252],         #8
    [1224,255],         #9
    [1254,340],         #10
    [967,340],          #11
    [680,342],          #12
    [709,256],          #13
    [620,255]           #14
],dtype=np.float32)
## real world coordinates ##
obj_points = np.array([
    [0,0,0],            #1
    [1.37,0,0],         #2
    [1.37,5.48,0],      #3
    [5.48,5.48,0],      #4
    [9.59,5.48,0],      #5
    [9.59,0,0],         #6
    [10.97,0,0],        #7
    ## top points ##
    [10.97,23.77,0],    #8
    [9.59,23.77,0],     #9
    [9.59,18.29,0],     #10
    [5.48,18.29,0],     #11
    [1.37,18.29,0],     #12
    [1.37,23.77,0],     #13
    [0,23.77,0]         #14
],dtype=np.float32)
cv2.imshow("image",image)
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                    [obj_points], [img_points], image_size, None, None
                                                                    )
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)
np.savez('../input/camera_matrix',camera_mat = camera_matrix,
                                    dist_coeffs = dist_coeffs)

# json_object = json.dumps(intrensic_parameters, indent = 4) 
projected_img_points, _ = cv2.projectPoints(obj_points, rvecs[0], tvecs[0], camera_matrix, dist_coeffs)

errors = np.linalg.norm(img_points - projected_img_points.squeeze(), axis=1)
mean_error = np.mean(errors)
print(f"Reprojection error: {mean_error:.4f} pixels")

for i, point in enumerate(projected_img_points):
    x, y = int(point[0][0]), int(point[0][1])
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    cv2.circle(image, (int(img_points[i][0]), int(img_points[i][1])), 5, (0, 0, 255), -1)


cv2.imshow("Projected vs Original Points", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
