import cv2
import numpy as np

# 创建对象点数组
base_path = 'C:\\Users\\FMS\\Pictures\\Camera Roll/'
objpoints = []
objp = np.zeros((5*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2) * 3

# 存储图像点数组和对象点数组
imgpoints = []
# images = ['image1.jpg', 'image2.jpg', 'image3.jpg']  # 替换为你的图像路径
images = [base_path+str(i)+'.jpg' for i in range(1, 24)]

# 检测角点并添加到图像点数组和对象点数组
for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, (1280, 720))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (8, 5), None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

# 标定相机
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 输出相机内部参数
print("相机矩阵:")
print(mtx)
print("\n畸变系数:")
print(dist)
print("\n旋转矩阵:")
print(rvecs[0])
print("\n平移向量:")
print(tvecs[0])
if __name__ == '__main__':
    pass
    # import cv2
    # import numpy as np
    #
    # # 读取图像
    # image = cv2.imread(base_path+'15'+'.jpg' )  # 替换为你的棋盘格图像路径
    # # cv2.imshow('image', image)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # # 定义棋盘格尺寸
    # pattern_size = (8, 5)  # 替换为你的棋盘格尺寸
    #
    # # 检测棋盘格角点
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    #
    # if ret:
    #     # 在图像上绘制角点
    #     cv2.drawChessboardCorners(image, pattern_size, corners, ret)
    #
    #     # 显示绘制角点后的图像
    #     cv2.imshow('Chessboard Corners', image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # else:
    #     print("未能检测到棋盘格角点")