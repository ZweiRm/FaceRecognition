import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd

def learn_dataset(location):
    # 导入数据集
    data = open(location, 'rb')
    dict = sio.loadmat(data)

    # 查看数据集标签
    print("数据集属性：%s" % dict.keys())

    # 将 MATLAB 转换为 dataframe
    feature = dict['fea']
    label = dict['gnd']

    feature_df = pd.DataFrame(feature)
    label_df = pd.DataFrame(label)
    data_df = pd.concat([feature_df, label_df], axis=1)

    # 查看有多少类
    print("当前数据集有 %d 类" % label_df.value_counts().count())
    print(data_df.head)

    # 打印第一张图片
    image = feature_df.iloc[0, :].values.reshape((int(feature_df.shape[1] ** 0.5), int(feature_df.shape[1] ** 0.5))).T
    plt.title(label_df.iloc[0].values)
    plt.imshow(image, cmap="gray")
    plt.show()

learn_dataset('./datasets/Yale_64x64.mat')
learn_dataset('./datasets/YaleB_32x32.mat')
learn_dataset('./datasets/ORL_64x64.mat')

