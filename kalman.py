import numpy as np


class my_kalman(object):
    def __init__(self, x_left, y_up, x_right, y_low, dt=0.5):
        self.dt = dt
        self.x = x_left
        self.y = y_up
        self.h = y_low-y_up
        self.w = x_right-x_left
        self.x_ = 0
        self.y_ = 0
        self.h_ = 0
        self.w_ = 0

    def predict(self):
        predict_matrix = np.array([[1, 0, self.dt, 0, 0, 0, 0, 0], [0, 1, 0, self.dt, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [
            0, 0, 0, 0, 1, 0, self.dt, 0, ], [0, 0, 0, 0, 0, 1, 0, self.dt], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]])
        result = np.dot(predict_matrix, self.state_matrix())
        self.x, self.y, self.x_, self.y_, self.h, self.w,  self.h_, self.w_ = result.flatten()
        # self.extract = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [
        #                         0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0]])
        return self.x, self.y, self.h, self.w

    def update(self, keypoints, H, W):
        x_A = np.empty((0, 2))
        x_b = np.empty((0, 1))
        y_A = np.empty((0, 2))
        y_b = np.empty((0, 1))

        inlier_cnt = 0
        outlier_x_left_cnt = 0
        outlier_x_right_cnt = 0
        outlier_y_high_cnt = 0
        outlier_y_low_cnt = 0
        padding_x=self.w*0.05
        padding_y=self.h*0.05
        inlier_cnt = 0

        total=0
        min_W = W
        max_W = 0
        max_H = 0
        min_H = H
        sum_x = 0
        sum_y = 0
        for keypoint in keypoints:
            total+=1
            sum_x += keypoint.pt[0]
            sum_y += keypoint.pt[1]
            if self.x< keypoint.pt[0] < self.x+self.w and self.y< keypoint.pt[1] < self.y+self.h:
                if keypoint.pt[0] > max_W:
                    max_W = keypoint.pt[0]
                if keypoint.pt[0] < min_W:
                    min_W = keypoint.pt[0]
                if keypoint.pt[1] > max_H:
                    max_H = keypoint.pt[1]
                if keypoint.pt[1] < min_H:
                    min_H = keypoint.pt[1]
                inlier_cnt += 1
            elif self.x-padding_x < keypoint.pt[0]<self.x and self.y < keypoint.pt[1] < self.y+self.h:
                outlier_x_left_cnt+=1
            elif self.x+self.w<keypoint.pt[0]<self.x+self.w+padding_x and self.y< keypoint.pt[1] < self.y+self.h:
                outlier_x_right_cnt+=1
            elif self.y-padding_y  < keypoint.pt[1]<self.y and self.x < keypoint.pt[0] < self.x+self.w:
                outlier_y_high_cnt+=1
            elif self.y+self.h<keypoint.pt[1]<self.y+self.h+padding_y and self.x < keypoint.pt[0] < self.x+self.w:
                outlier_y_low_cnt+=1
        W = max_W-min_W
        H = max_H-min_H
        for keypoint in keypoints:
            if min_W < keypoint.pt[0] < max_W and min_H < keypoint.pt[1] < max_H:
                x_A = np.vstack((x_A, [1, (keypoint.pt[0]-min_W)/W]))
                x_b = np.vstack((x_b, keypoint.pt[0]))
                y_A = np.vstack((y_A, [1, (keypoint.pt[1]-min_H)/H]))
                y_b = np.vstack((y_b, keypoint.pt[1]))
        result_x = np.dot(np.linalg.inv(
            np.dot(x_A.T, x_A)), np.dot(x_A.T, x_b)).flatten()
        result_y = np.dot(np.linalg.inv(
            np.dot(y_A.T, y_A)), np.dot(y_A.T, y_b)).flatten()
        self.x_ = (sum_x/total-self.x-self.w/2)/self.x-self.w*outlier_x_left_cnt/inlier_cnt
        self.y_ = (sum_y/total-self.y-self.h/2)/self.y-self.h*outlier_y_high_cnt/inlier_cnt
        self.w_ = self.w*(outlier_x_left_cnt+outlier_y_high_cnt)/inlier_cnt
        self.h_ = self.h*(outlier_y_high_cnt+outlier_y_low_cnt)/inlier_cnt

    def state_matrix(self):
        return np.array([[self.x, self.y, self.x_, self.y_, self.h, self.w,  self.h_, self.w_]]
                        ).T  # x,y,x_,y_,w,h,w_,h_
