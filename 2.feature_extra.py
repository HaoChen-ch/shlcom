# 第二步，输入文件是已经打好label的文件
import numpy as np
import pandas as pd
import os


# -------------------------------------------------------------------------------------------------
# 把加速度传感器的值从对手机坐标系转换成对地坐标系，具体转换公式可以参考Android的getRotationMatrix源码

# 需要重力和磁感应的三轴数值
#计算错误弃用了
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
    data = pd.read_csv("dirty.csv")
    print(data.shape)

    gravity = np.asarray([data['g_x'], data['g_y'], data['g_z']]).T
    magnetic = np.asarray([data['m_x'], data['m_y'], data['m_z']]).T

    rotate = np.zeros(shape=(len(gravity), 9))

    acc_o = np.asarray([data['acc_x'], data['acc_y'], data['acc_z']]).T
    acc = np.zeros(shape=(len(acc_o), 3))
    getRotationMatrix(rotate, gravity, magnetic)
    print("data", data.shape)
    print("acc", acc.shape)
    acc[:, 0] = rotate[:, 0] * acc_o[:, 0] + rotate[:, 1] * acc_o[:, 1] + rotate[:, 2] * acc_o[:, 2]
    acc[:, 1] = rotate[:, 3] * acc_o[:, 0] + rotate[:, 4] * acc_o[:, 1] + rotate[:, 5] * acc_o[:, 2]
    acc[:, 2] = rotate[:, 6] * acc_o[:, 0] + rotate[:, 7] * acc_o[:, 1] + rotate[:, 8] * acc_o[:, 2] - 9.807
    label = np.asarray(data['label'])
    time = np.asarray(data['time'])
    print(label.shape)
    label = label.reshape(label.shape[0], 1)
    time = time.reshape(time.shape[0], 1)
    print(label.shape)
    label = pd.DataFrame(label)
    acc = np.hstack((time, acc))
    acc_xy = pd.DataFrame(np.sqrt(np.square(acc[:, 1]) + np.square(acc[:, 2])))
    acc_xyz = pd.DataFrame(np.sqrt(np.square(acc[:, 1]) + np.square(acc[:, 2]) + np.square(acc[:, 3])))
    acc = pd.DataFrame(acc)
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

    o1 = np.asarray([data['o_x'], data['o_y'], data['o_z']])
    o_x = pd.DataFrame(rn0 * o1[0] + rn1 * o1[1] + rn2 * o1[2])
    o_y = pd.DataFrame(rn3 * o1[0] + rn4 * o1[1] + rn5 * o1[2])
    o_z = pd.DataFrame(rn6 * o1[0] + rn7 * o1[1] + rn8 * o1[2])
    pitch = pd.DataFrame(np.arctan(rn7 / rn8))
    roll = pd.DataFrame(np.arcsin(-rn6))
    yaw = pd.DataFrame(np.arctan(rn3 / rn0))
    ori = pd.concat((o_x, o_y, o_z, pitch, roll, yaw), axis=1)
    print(ori.shape)
    # 输出格式为['o_x', 'o_y', 'o_z', 'pitch', 'roll', 'yaw']
    # -----------------------------------------------------------------------------------------------
    # 对m_x,m_y,m_z取平方和之后开根号，作为新的列值
    magnetic = np.asarray([data['m_x'], data['m_y'], data['m_z']]).T

    ma = np.sqrt(np.square(magnetic[:, 0]) + np.square(magnetic[:, 1]) + np.square(magnetic[:, 2]))
    magnetic = pd.DataFrame(magnetic)
    print("magnetic", magnetic.shape)
    ma_t = pd.DataFrame(ma)
    print(ma_t.shape)
    ma = pd.concat((ma_t, magnetic), axis=1)

    # 输出格式为['ma','m_x', 'm_y', 'm_z']
    # -----------------------------------------------------------------------------------------------
    remain = pd.DataFrame(np.asarray([data['gy_x'], data['gy_y'], data['gy_z'],
                                      data['g_x'], data['g_x'], data['g_x'],
                                      data['l_x'], data['l_x'], data['l_x'],
                                      data['pressure']
                                      ]).T)

    fin = pd.concat((acc, acc_xy, acc_xyz, ori, ma, remain, label), axis=1)
    print(fin.shape)

    fin.to_csv("raw_data.csv", index=False, header=['time', 'acc_x', 'acc_y', 'acc_z', 'acc_xy', 'acc_xyz',
                                                    'o_x', 'o_y', 'o_z', 'pitch', 'roll', 'yaw',
                                                    'magnetic', 'm_x', 'm_y', 'm_z',
                                                    'gy_x', 'gy_y', 'gy_z',
                                                    'g_x', 'g_y', 'g_z',
                                                    'l_x', 'l_y', 'l_z',
                                                    'pressure',
                                                    'label'])


def train():
    os.chdir('data')
    data = pd.read_csv("dirty.csv")
    print(data.shape)

    gravity = np.asarray([data['g_x'], data['g_y'], data['g_z']]).T
    magnetic = np.asarray([data['m_x'], data['m_y'], data['m_z']]).T

    rotate = np.zeros(shape=(len(gravity), 9))

    acc_o = np.asarray([data['acc_x'], data['acc_y'], data['acc_z']]).T
    acc = np.zeros(shape=(len(acc_o), 3))
    getRotationMatrix(rotate, gravity, magnetic)
    print("data", data.shape)
    print("acc", acc.shape)
    acc[:, 0] = rotate[:, 0] * acc_o[:, 0] + rotate[:, 1] * acc_o[:, 1] + rotate[:, 2] * acc_o[:, 2]
    acc[:, 1] = rotate[:, 3] * acc_o[:, 0] + rotate[:, 4] * acc_o[:, 1] + rotate[:, 5] * acc_o[:, 2]
    acc[:, 2] = rotate[:, 6] * acc_o[:, 0] + rotate[:, 7] * acc_o[:, 1] + rotate[:, 8] * acc_o[:, 2] - 9.807
    label = np.asarray(data['label'])
    time = np.asarray(data['time'])
    print(label.shape)
    label = label.reshape(label.shape[0], 1)
    time = time.reshape(time.shape[0], 1)
    print(label.shape)
    label = pd.DataFrame(label)
    acc = np.hstack((time, acc))
    # acc = pd.DataFrame(acc)
    acc_xy = pd.DataFrame(np.sqrt(np.square(acc[:, 1]) + np.square(acc[:, 2])))
    acc_xyz = pd.DataFrame(np.sqrt(np.square(acc[:, 1]) + np.square(acc[:, 2]) + np.square(acc[:, 3])))
    acc = pd.DataFrame(acc)
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

    o1 = np.asarray([data['o_x'], data['o_y'], data['o_z']])
    o_x = pd.DataFrame(rn0 * o1[0] + rn1 * o1[1] + rn2 * o1[2])
    o_y = pd.DataFrame(rn3 * o1[0] + rn4 * o1[1] + rn5 * o1[2])
    o_z = pd.DataFrame(rn6 * o1[0] + rn7 * o1[1] + rn8 * o1[2])
    pitch = pd.DataFrame(np.arctan(rn7 / rn8))
    roll = pd.DataFrame(np.arcsin(-rn6))
    yaw = pd.DataFrame(np.arctan(rn3 / rn0))
    ori = pd.concat((o_x, o_y, o_z, pitch, roll, yaw), axis=1)
    print(ori.shape)
    # 输出格式为['o_x', 'o_y', 'o_z', 'pitch', 'roll', 'yaw']
    # -----------------------------------------------------------------------------------------------
    # 对m_x,m_y,m_z取平方和之后开根号，作为新的列值
    magnetic = np.asarray([data['m_x'], data['m_y'], data['m_z']]).T

    ma = np.sqrt(np.square(magnetic[:, 0]) + np.square(magnetic[:, 1]) + np.square(magnetic[:, 2]))
    magnetic = pd.DataFrame(magnetic)
    print("magnetic", magnetic.shape)
    ma_t = pd.DataFrame(ma)
    print(ma_t.shape)
    ma = pd.concat((ma_t, magnetic), axis=1)

    # 输出格式为['ma','m_x', 'm_y', 'm_z']
    # -----------------------------------------------------------------------------------------------
    remain = pd.DataFrame(np.asarray([data['gy_x'], data['gy_y'], data['gy_z'],
                                      data['g_x'], data['g_x'], data['g_x'],
                                      data['l_x'], data['l_x'], data['l_x'],
                                      data['pressure']
                                      ]).T)

    fin = pd.concat((acc, acc_xy, acc_xyz, ori, ma, remain, label), axis=1)
    print(fin.shape)

    fin.to_csv("raw_data.csv", index=False, header=['time', 'acc_x', 'acc_y', 'acc_z', 'acc_xy', 'acc_xyz',
                                                    'o_x', 'o_y', 'o_z', 'pitch', 'roll', 'yaw',
                                                    'magnetic', 'm_x', 'm_y', 'm_z',
                                                    'gy_x', 'gy_y', 'gy_z',
                                                    'g_x', 'g_y', 'g_z',
                                                    'l_x', 'l_y', 'l_z',
                                                    'pressure',
                                                    'label'])


if __name__ == '__main__':
    train()
