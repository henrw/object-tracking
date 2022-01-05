import cv2
import numpy as np
import matplotlib.pyplot as plt
from common import *
from kalman import my_kalman
import os

out_dir = "./output/"
in_dir = "./source/"

cap = cv2.VideoCapture(os.path.join(in_dir, 'dog.mp4'))

# x_left = 500
# x_right = 900
# y_high = 100
# y_low = 820

# (800, 640, 3)
x_left = 220
x_right = 380
y_high = 220
y_low = 400

ret, frame = cap.read()
cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
result = cv2.VideoWriter(os.path.join(out_dir, 'result.avi'), fourcc,
                         20, (frame.shape[1], frame.shape[0]))
sift = cv2.SIFT_create()
my_km = my_kalman(x_left, y_high, x_right, y_low)
img_filtered = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
black_frame = rectangular_mask(
    img_filtered, x_left, x_right, y_low, y_high)
keypoints_base, descriptors_base = sift.detectAndCompute(black_frame, None)
frame = cv2.rectangle(frame, (x_left, y_high),
                      (x_right, y_low), (0, 255, 0), 2)
out = cv2.drawKeypoints(black_frame, keypoints_base, black_frame)
save_img(out, os.path.join(out_dir, 'frame_0.png'))
cnt = 0
while (cap.isOpened()):
    cnt += 1
    ret, frame = cap.read()
    if not ret:
        break
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    keypoints = match_keypoints(descriptors_base, frame)
    my_km.update(keypoints, frame.shape[0], frame.shape[1])
    x, y, h, w = my_km.predict()
    frame = cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
    out = cv2.drawKeypoints(frame, keypoints, frame)
    print(out)
    result.write(out)
    save_img(out, os.path.join(out_dir, 'frame_'+str(cnt)+'.png'))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
result.release()
cv2.destroyAllWindows()
