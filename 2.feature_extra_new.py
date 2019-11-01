import numpy as np
import pandas as pd
import os


# 第二步，输入文件是已经打好label的文件，对每条记录中的acc和magnetic分量做对手机坐标系到对地坐标系的转化

def train():
    os.chdir('data')
    data = pd.read_csv("dirty.csv")
    print(data.shape)

    label = np.asarray(data['label'])
    time = np.asarray(data['time'])

    label = label.reshape(label.shape[0], 1)
    time = time.reshape(time.shape[0], 1)
    label = pd.DataFrame(label)
    time = pd.DataFrame(time)

    # 输出格式['time', 'acc_x', 'acc_y', 'acc_z']
    # --------------------------------------------------------------------------------------------------
    # # 对acc和magnetic进行坐标系的转换，具体公式可以参考ubicomp2018第一名的公式

    orientation = np.asarray([data['o_w'], data['o_x'], data['o_y'], data['o_z']])
    orien = orientation.T
    rn0 = 1 - 2 * (np.square(orien[:, 2]) + np.square(orien[:, 3]))
    rn1 = 2 * (orien[:, 1] * orien[:, 2] - orien[:, 0] * orien[:, 3])
    rn2 = 2 * (orien[:, 1] * orien[:, 3] + orien[:, 0] * orien[:, 2])

    rn3 = 2 * (orien[:, 1] * orien[:, 2] + orien[:, 0] * orien[:, 3])
    rn4 = 1 - 2 * (np.square(orien[:, 1]) + np.square(orien[:, 3]))
    rn5 = 2 * (orien[:, 2] * orien[:, 3] - orien[:, 0] * orien[:, 1])

    rn6 = 2 * (orien[:, 1] * orien[:, 3] - orien[:, 0] * orien[:, 2])
    rn7 = 2 * (orien[:, 2] * orien[:, 3] + orien[:, 0] * orien[:, 1])
    rn8 = 1 - 2 * (np.square(orien[:, 1]) + np.square(orien[:, 2]))

    acc = np.asarray([data['acc_x'], data['acc_y'], data['acc_z']])
    acc_x = pd.DataFrame(rn0 * acc[0] + rn1 * acc[1] + rn2 * acc[2])
    acc_y = pd.DataFrame(rn3 * acc[0] + rn4 * acc[1] + rn5 * acc[2])
    acc_z = pd.DataFrame(rn6 * acc[0] + rn7 * acc[1] + rn8 * acc[2])
    acc = np.hstack((acc_x, acc_y, acc_z))
    acc_xy = pd.DataFrame(np.sqrt(np.square(acc[:, 0]) + np.square(acc[:, 1])))
    acc_xyz = pd.DataFrame(np.sqrt(np.square(acc[:, 0]) + np.square(acc[:, 1]) + np.square(acc[:, 2])))
    acc = pd.DataFrame(acc)
    print("acc.shape", acc.shape)

    pitch = pd.DataFrame(np.arctan(rn7 / rn8))
    roll = pd.DataFrame(np.arcsin(-rn6))
    yaw = pd.DataFrame(np.arctan(rn3 / rn0))
    orien = pd.DataFrame(orien)
    ori = pd.concat((orien, pitch, roll, yaw), axis=1)
    print("ori.shape: ", ori.shape)
    # 输出格式为['o_w','o_x', 'o_y', 'o_z', 'pitch', 'roll', 'yaw']
    # -----------------------------------------------------------------------------------------------
    # 对m_x,m_y,m_z取平方和之后开根号，作为新的列值，并且对magnetic做坐标转化

    mag = np.asarray([data['m_x'], data['m_y'], data['m_z']])
    mag_x = pd.DataFrame(rn0 * mag[0] + rn1 * mag[1] + rn2 * mag[2])
    mag_y = pd.DataFrame(rn3 * mag[0] + rn4 * mag[1] + rn5 * mag[2])
    mag_z = pd.DataFrame(rn6 * mag[0] + rn7 * mag[1] + rn8 * mag[2])
    mag = mag.T
    ma = np.sqrt(np.square(mag[:, 0]) + np.square(mag[:, 1]) + np.square(mag[:, 2])).reshape(-1, 1)
    magnetic = pd.DataFrame(np.hstack((ma, mag_x, mag_y, mag_z)))
    print(magnetic.shape)
    # 输出格式为['ma','m_x', 'm_y', 'm_z']
    # -----------------------------------------------------------------------------------------------
    remain = pd.DataFrame(np.asarray([data['gy_x'], data['gy_y'], data['gy_z'],
                                      data['g_x'], data['g_x'], data['g_x'],
                                      data['l_x'], data['l_x'], data['l_x'],
                                      data['pressure']
                                      ]).T)

    fin = pd.concat((time, acc, acc_xy, acc_xyz, ori, magnetic, remain, label), axis=1)
    print("fin.shape", fin.shape)

    fin.to_csv("raw_data.csv", index=False, header=['time', 'acc_x', 'acc_y', 'acc_z', 'acc_xy', 'acc_xyz',
                                                    'o_w', 'o_x', 'o_y', 'o_z', 'pitch', 'roll', 'yaw',
                                                    'magnetic', 'm_x', 'm_y', 'm_z',
                                                    'gy_x', 'gy_y', 'gy_z',
                                                    'g_x', 'g_y', 'g_z',
                                                    'l_x', 'l_y', 'l_z',
                                                    'pressure',
                                                    'label'])


def dev():
    os.chdir('test')
    data = pd.read_csv("dirty.csv")
    print(data.shape)

    label = np.asarray(data['label'])
    time = np.asarray(data['time'])

    label = label.reshape(label.shape[0], 1)
    time = time.reshape(time.shape[0], 1)
    label = pd.DataFrame(label)
    time = pd.DataFrame(time)

    # 输出格式['time', 'acc_x', 'acc_y', 'acc_z']
    # --------------------------------------------------------------------------------------------------
    # # 对acc和magnetic进行坐标系的转换，具体公式可以参考ubicomp2018第一名的公式

    orientation = np.asarray([data['o_w'], data['o_x'], data['o_y'], data['o_z']])
    orien = orientation.T
    rn0 = 1 - 2 * (np.square(orien[:, 2]) + np.square(orien[:, 3]))
    rn1 = 2 * (orien[:, 1] * orien[:, 2] - orien[:, 0] * orien[:, 3])
    rn2 = 2 * (orien[:, 1] * orien[:, 3] + orien[:, 0] * orien[:, 2])

    rn3 = 2 * (orien[:, 1] * orien[:, 2] + orien[:, 0] * orien[:, 3])
    rn4 = 1 - 2 * (np.square(orien[:, 1]) + np.square(orien[:, 3]))
    rn5 = 2 * (orien[:, 2] * orien[:, 3] - orien[:, 0] * orien[:, 1])

    rn6 = 2 * (orien[:, 1] * orien[:, 3] - orien[:, 0] * orien[:, 2])
    rn7 = 2 * (orien[:, 2] * orien[:, 3] + orien[:, 0] * orien[:, 1])
    rn8 = 1 - 2 * (np.square(orien[:, 1]) + np.square(orien[:, 2]))

    acc = np.asarray([data['acc_x'], data['acc_y'], data['acc_z']])
    acc_x = pd.DataFrame(rn0 * acc[0] + rn1 * acc[1] + rn2 * acc[2])
    acc_y = pd.DataFrame(rn3 * acc[0] + rn4 * acc[1] + rn5 * acc[2])
    acc_z = pd.DataFrame(rn6 * acc[0] + rn7 * acc[1] + rn8 * acc[2])
    acc = np.hstack((acc_x, acc_y, acc_z))
    acc_xy = pd.DataFrame(np.sqrt(np.square(acc[:, 0]) + np.square(acc[:, 1])))
    acc_xyz = pd.DataFrame(np.sqrt(np.square(acc[:, 0]) + np.square(acc[:, 1]) + np.square(acc[:, 2])))
    acc = pd.DataFrame(acc)
    print("acc.shape", acc.shape)

    pitch = pd.DataFrame(np.arctan(rn7 / rn8))
    roll = pd.DataFrame(np.arcsin(-rn6))
    yaw = pd.DataFrame(np.arctan(rn3 / rn0))
    orien = pd.DataFrame(orien)
    ori = pd.concat((orien, pitch, roll, yaw), axis=1)
    print("ori.shape: ", ori.shape)
    # 输出格式为['o_w','o_x', 'o_y', 'o_z', 'pitch', 'roll', 'yaw']
    # -----------------------------------------------------------------------------------------------
    # 对m_x,m_y,m_z取平方和之后开根号，作为新的列值，并且对magnetic做坐标转化

    mag = np.asarray([data['m_x'], data['m_y'], data['m_z']])
    mag_x = pd.DataFrame(rn0 * mag[0] + rn1 * mag[1] + rn2 * mag[2])
    mag_y = pd.DataFrame(rn3 * mag[0] + rn4 * mag[1] + rn5 * mag[2])
    mag_z = pd.DataFrame(rn6 * mag[0] + rn7 * mag[1] + rn8 * mag[2])
    mag = mag.T
    ma = np.sqrt(np.square(mag[:, 0]) + np.square(mag[:, 1]) + np.square(mag[:, 2])).reshape(-1, 1)
    magnetic = pd.DataFrame(np.hstack((ma, mag_x, mag_y, mag_z)))
    print(magnetic.shape)
    # 输出格式为['ma','m_x', 'm_y', 'm_z']
    # -----------------------------------------------------------------------------------------------
    remain = pd.DataFrame(np.asarray([data['gy_x'], data['gy_y'], data['gy_z'],
                                      data['g_x'], data['g_x'], data['g_x'],
                                      data['l_x'], data['l_x'], data['l_x'],
                                      data['pressure']
                                      ]).T)

    fin = pd.concat((time, acc, acc_xy, acc_xyz, ori, magnetic, remain, label), axis=1)
    print("fin.shape", fin.shape)

    fin.to_csv("raw_data.csv", index=False, header=['time', 'acc_x', 'acc_y', 'acc_z', 'acc_xy', 'acc_xyz',
                                                    'o_w', 'o_x', 'o_y', 'o_z', 'pitch', 'roll', 'yaw',
                                                    'magnetic', 'm_x', 'm_y', 'm_z',
                                                    'gy_x', 'gy_y', 'gy_z',
                                                    'g_x', 'g_y', 'g_z',
                                                    'l_x', 'l_y', 'l_z',
                                                    'pressure',
                                                    'label'])


if __name__ == '__main__':
    train()
    # dev()
