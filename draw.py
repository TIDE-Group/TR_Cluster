import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

F1s = [0.64788732, 0.42105263,  0.54545455, 0.,         0.96103896, 0.92783505,  0.57142857, 0.84210526, 0.85714286, 0.58333333, 0.92592593, 0.5,  0.8,        1.,         0.92307692, 0.57142857, 1.,         0.5, 0.55555556, 0.,         0.5,        1.,         0.,         1., 0.,         0.54166667, 0. ,        0.5,        0.72727273, 0.71428571, 0.,         0.,         1.,         1. ,        0. ,        1., 0.,         1.,         0.,        ];

F1s = np.array(F1s)

plt.figure(figsize=(8, 6))
plt.hist(F1s, bins=20)
plt.title('各个话题的类内F1分布')
plt.xlabel('F1')
plt.savefig('./f1s.png', dpi=400)
plt.show()