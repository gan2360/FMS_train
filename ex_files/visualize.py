import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
import pickle

BODY_22_color = np.array([[255, 0, 0], [255, 51, 0], [255, 102, 0], [255, 153, 0], [255, 204, 0], [255, 255, 0], [204, 255, 0]
                         , [153, 255, 0], [102, 255, 0], [51, 255, 0], [0, 255, 0], [0, 255, 51], [0, 255, 102], [0,255,153]
                         , [0, 255, 204], [0, 255, 255], [0, 204, 255], [0, 153, 255], [0, 102, 255], [0, 53, 255], [0, 0, 255]
                         , [53, 0, 255], [102, 0, 255], [153, 0, 255], [204, 0, 255], [255, 0, 255]])
# BODY_22_color = np.array([[239, 43, 3], [3, 216, 239], [3, 216, 239], [3, 216, 239], [239, 219, 3], [239, 219, 3], [239, 219, 3]
#                          , [239, 3, 217], [239, 3, 217], [239, 3, 217], [37, 239, 3], [37, 239, 3], [37, 239, 3], [239, 43, 3]
#                          , [239, 43, 3], [239, 118, 3], [3, 239, 173], [3, 239, 173], [3, 239, 173], [125, 3, 239], [125, 3, 239]
#                          , [125, 3, 239]]) 

BODY_22_pairs = np.array([[14, 13], [13, 0], [14, 19], [14, 16], [19, 20], [20, 21], [16, 17], [17, 18], [0, 1], [1, 2], [2, 3], [0, 7],
                        [7, 8], [8, 9], [14, 15], [9, 11], [11, 12], [9, 10],
                         [3, 5], [5, 6], [3, 4]])

def rotate(keypoint, degree):
    r = R.from_euler('z', degree, degrees=True)
    r = r.as_matrix()
    b = np.array([0.5, 0.5, 0 ])
    keypoint_r = np.copy(keypoint)

    for frame in range(keypoint.shape[0]):
        for i in range(keypoint.shape[1]):
            keypoint_r[frame,i,:] = np.dot(r,(np.reshape(keypoint[frame,i,:], (3)) - b)) + b

    return keypoint_r


def plotKeypoint(keypoints_pred, keypoints_gt, gt_comp=False):
    b = [-800,-800,0]
    resolution = 100
    scale = 19
    fig = plt.figure(figsize=(4.8,4.8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-80,80)
    ax.set_ylim(-80,80)
    ax.set_zlim(0,200)
    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)
    ax.view_init(20,-70)
    keypoints_pred = keypoints_pred * scale # 20 as max
    keypoints_pred = keypoints_pred * resolution + b
    xs = keypoints_pred[:, 0]*1/10
    ys = keypoints_pred[:, 1]*1/10
    zs = keypoints_pred[:, 2]*1/10
    for i in range(BODY_22_pairs.shape[0]):
        index_1 = BODY_22_pairs[i, 0]
        index_2 = BODY_22_pairs[i, 1]
        xs_line = [xs[index_1],xs[index_2]]
        ys_line = [ys[index_1],ys[index_2]]
        zs_line = [zs[index_1],zs[index_2]]
        ax.plot(xs_line,ys_line,zs_line, color = BODY_22_color[i]/255.0)
    ax.scatter(xs, ys, zs, s=20, c=BODY_22_color[:22]/255.0)

    if gt_comp:
        keypoints_gt = keypoints_gt * scale # 20 as max
        keypoints_gt = keypoints_gt * resolution + b

        xs = keypoints_gt[:, 0]/10
        ys = keypoints_gt[:, 1]/10
        zs = keypoints_gt[:, 2]/10

        for i in range(BODY_22_pairs.shape[0]):
            index_1 = BODY_22_pairs[i, 0]
            index_2 = BODY_22_pairs[i, 1]
            xs_line = [xs[index_1],xs[index_2]]
            ys_line = [ys[index_1],ys[index_2]]
            zs_line = [zs[index_1],zs[index_2]]
            ax.plot(xs_line,ys_line,zs_line, color = (0,0,0))
        ax.scatter(xs, ys, zs, s=20, color=(0,0,0))

    fig.canvas.draw()
    frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1]
    return img

def generateKeypointImage(keypoints_pred, keypoints_gt, path):  # pred代表预测出来的，gt代表真实的
    keypoints_pred = rotate(keypoints_pred, 90)
    keypoints_gt = rotate(keypoints_gt, 90)
    for i in range(keypoints_pred.shape[0]): #keypoint_GT.shape[0]):
        print (i)
        img = plotKeypoint(np.reshape(keypoints_pred[i], (22,3)), np.reshape(keypoints_gt[i], (22,3)), gt_comp=True)
        cv2.imwrite(path + 'pred%06d.jpg' % i, img)

def generateKeypointVideo(keypoints_pred, keypoints_gt, path):
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(path + '.avi', fourcc, 30, (480,480))
    print ('Video streaming')
    for i in range(keypoints_pred.shape[0]): #keypoint_GT.shape[0]):
        img = plotKeypoint(np.reshape(keypoints_pred[i], (22,3)), np.reshape(keypoints_gt[i], (22,3)), gt_comp=True)
        out.write(img)

if __name__=='__main__':
    data = pickle.load(open('./test_output/predictions/data/singlePeople_0.0001_0_best55.p', "rb"))
    # print(data[2].shape)
    kp_gt = data[2][:]
    kp_pred = data[3][:]
    # print(kp_pred[0])
    generateKeypointVideo(kp_pred, kp_gt, './test_output/predictions/video/vis' + str(55))