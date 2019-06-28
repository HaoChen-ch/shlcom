# 第二步，输入文件是已经打好label的文件
import numpy as np
import pandas as pd
import os


# -------------------------------------------------------------------------------------------------
# 把加速度传感器的值从对手机坐标系转换成对地坐标系，具体转换公式可以参考Android的getRotationMatrix源码

# 需要重力和磁感应的三轴数值

def getRotationMatrix(R, gravity, geomagnetic):
    Ax = gravity[:, 0].T
    Ay = gravity[:, 1].T
    Az = gravity[:, 2].T
    Ex = geomagnetic[:, 0].T
    Ey = geomagnetic[:, 1].T
    Ez = geomagnetic[:, 2].T
    Hx = Ey * Az - Ez * Ay
    Hy = Ez * Ax - Ex * Az
    Hz = Ex * Ay - Ey * Ax
    normH = np.sqrt(Hx * Hx + Hy * Hy + Hz * Hz)
    invH = 1.0 / normH
    Hx *= invH
    Hy *= invH
    Hz *= invH
    invA = 1.0 / np.sqrt(Ax * Ax + Ay * Ay + Az * Az)
    Ax *= invA
    Ay *= invA
    Az *= invA
    Mx = Ay * Hz - Az * Hy
    My = Az * Hx - Ax * Hz
    Mz = Ax * Hy - Ay * Hx
    R[:, 0] = Hx
    R[:, 1] = Hy
    R[:, 2] = Hz
    R[:, 3] = Mx
    R[:, 4] = My
    R[:, 5] = Mz
    R[:, 6] = Ax
    R[:, 7] = Ay
    R[:, 8] = Az


def dev():
    os.chdir('test')
    data = pd.read_csv("dirty.csv", nrows=5)
    print(data)

    gravity = np.asarray([data['g_x'], data['g_y'], data['g_z']]).T
    magnetic = np.asarray([data['m_x'], data['m_y'], data['m_z']]).T

    rotate = np.zeros(shape=(len(gravity), 9))

    acc_o = np.asarray([data['acc_x'], data['acc_y'], data['acc_z']]).T
    acc = np.zeros(shape=(len(acc_o), 3))
    getRotationMatrix(rotate, gravity, magnetic)
    acc[:, 0] = rotate[:, 0] * acc_o[:, 0] + rotate[:, 1] * acc_o[:, 1] + rotate[:, 2] * acc_o[:, 2]
    acc[:, 1] = rotate[:, 3] * acc_o[:, 0] + rotate[:, 4] * acc_o[:, 1] + rotate[:, 5] * acc_o[:, 2]
    acc[:, 2] = rotate[:, 6] * acc_o[:, 0] + rotate[:, 7] * acc_o[:, 1] + rotate[:, 8] * acc_o[:, 2]
    label = np.asarray(data['label'])
    time = np.asarray(data['time'])
    label = label.reshape(label.shape[0], 1)
    time = time.reshape(time.shape[0], 1)
    label = pd.DataFrame(label)
    # acc = np.hstack((time, acc))
    acc = pd.DataFrame(acc)
    print("acc*Matrix: ", acc)
    # 输出格式['time', 'acc_x', 'acc_y', 'acc_z']
    # --------------------------------------------------------------------------------------------------
    # # 对o_w,o_x,o_y,o_z进行坐标系的转换，具体公式可以参考ubicomp2018第一名的公式

    orientation = np.asarray([data['o_w'], data['o_x'], data['o_y'], data['o_z']])
    orien = orientation.T
    rn0 = np.asarray(1 - 2 * (np.square(orien[:, 2]) + np.square(orien[:, 3])))
    rn1 = 2 * (orien[:, 1] * orien[:, 2] - orien[:, 0] * orien[:, 3])
    rn2 = 2 * (orien[:, 1] * orien[:, 3] + orien[:, 0] * orien[:, 2])
    rn3 = 2 * (orien[:, 1] * orien[:, 2] + orien[:, 0] * orien[:, 3])
    rn4 = 1 - 2 * (np.square(orien[:, 1]) + np.square(orien[:, 3]))
    rn5 = 2 * (orien[:, 2] * orien[:, 3] - orien[:, 0] * orien[:, 1])
    rn6 = 2 * (orien[:, 1] * orien[:, 3] - orien[:, 0] * orien[:, 2])
    rn7 = 2 * (orien[:, 2] * orien[:, 3] + orien[:, 0] * orien[:, 1])
    rn8 = 1 - 2 * (np.square(orien[:, 1]) + np.square(orien[:, 2]))

    o1 = np.asarray([data['acc_x'], data['acc_y'], data['acc_z']])
    o_x = pd.DataFrame(rn0 * o1[0] + rn1 * o1[1] + rn2 * o1[2])
    o_y = pd.DataFrame(rn3 * o1[0] + rn4 * o1[1] + rn5 * o1[2])
    o_z = pd.DataFrame(rn6 * o1[0] + rn7 * o1[1] + rn8 * o1[2])
    pitch = pd.DataFrame(np.arctan(rn7 / rn8))
    roll = pd.DataFrame(np.arcsin(-rn6))
    yaw = pd.DataFrame(np.arctan(rn3 / rn0))
    ori = pd.concat((o_x, o_y, o_z), axis=1)
    print("acc论文方法 :", ori)


if __name__ == '__main__':
    dev()
