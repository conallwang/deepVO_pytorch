from glob import glob

import numpy as np
import cv2

for seq in range(11):
    poses = np.loadtxt(f'/home/wangcong/workspace/ICRA 2021/DeepVO/poses/{seq:02}.txt')

    num = poses.shape[0]
    res = []
    for i in range(num - 1):
        # 齐次项
        c = np.array([[0, 0, 0, 1]])

        # get pose
        pose = np.concatenate((poses[i].reshape(3, 4), c), axis=0)
        pose_1 = np.concatenate((poses[i + 1].reshape(3, 4), c), axis=0)

        # relative_pose * pose_0(pose) = pose_1
        relative_pose = np.matmul(pose_1, np.linalg.inv(pose))

        # get R and t
        relative_R = relative_pose[:3, :3]
        relative_t = relative_pose[:3, -1]

        relative_r, _ = cv2.Rodrigues(relative_R)

        relative_r = relative_r.reshape(3)
        relative_t = relative_t.reshape(3)

        relative_p = np.concatenate((relative_r, relative_t), axis=0)

        res.append(relative_p)

    res = np.array(res)
    print(res.shape)
    np.savetxt(f'/home/wangcong/workspace/ICRA 2021/DeepVO/poses/{seq:02}_rp.txt', res)