import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

epochs = list(range(1, 55))
train_loss_multi = [0.3854,0.3029,0.2451,0.1801,0.1914,0.1543,0.1753,0.1421,0.1387,0.1659,0.1288,0.1169,0.1337,0.0962,0.1553,0.1305,0.1036,0.0895,0.0958,0.1078,0.0936,0.0785,0.0792,0.0634,0.0767,0.0576,0.0582,0.0485,0.0479,0.0616,0.0899,0.0427,0.0375,0.0406,0.0336,0.0379,0.0696,0.0240,0.0311,0.0212,0.0285,0.0310,0.0245,0.0173,0.0158,0.0159,0.0093,0.0143,0.0082,0.0077,0.0532,0.0076,0.0045,0.0081]
val_loss_multi = [0.4452,0.3302,0.1888,0.2942,0.6836,0.2829,0.2798,0.2112,0.2141,0.2320,0.2621,0.3912,0.2669,0.2708,0.2849,0.3031,0.3068,0.3231,0.3053,0.3394,0.3429,0.3678,0.3682,0.3267,0.3476,0.3970,0.4150,0.3936,0.4317,0.4450,0.4575,0.4557,0.4787,0.4931,0.4819,0.4855,0.4752,0.5228,0.5369,0.5581,0.5605,0.5579,0.5773,0.5855,0.5783,0.5903,0.5862,0.5787,0.6061,0.5961,0.6183,0.6314,0.6727,0.6517]

plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss_multi, 'b-', linewidth=2, label='训练损失')
plt.plot(epochs, val_loss_multi, 'r-', linewidth=2, label='验证损失')
plt.xlabel('训练周期')
plt.ylabel('损失值')
plt.title('多分类模型训练与验证损失曲线')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('multi_loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()



epochs_bin = list(range(1, 59))
train_loss_bin = [
    0.5590,0.5027,0.4823,0.4633,0.4401,0.4306,0.4301,0.4455,0.4047,0.3520,
    0.3178,0.3344,0.2934,0.2814,0.2652,0.2386,0.2563,0.2172,0.2270,0.1835,
    0.1392,0.1794,0.1597,0.1119,0.1072,0.1005,0.0783,0.0949,0.0730,0.0835,
    0.0680,0.0289,0.0441,0.0240,0.0187,0.0164,0.0136,0.0182,0.0138,0.0140,
    0.0162,0.0053,0.0180,0.0122,0.0143,0.0116,0.0067,0.0029,0.0004,0.0003,
    0.0010,0.0015,0.0004,0.0033,0.0006,0.0013,0.0043,0.0005
]
val_loss_bin = [
    0.2742,0.1148,0.1090,0.0991,0.0764,0.1904,0.0956,0.0733,0.0696,0.0778,
    0.1206,0.0892,0.1047,0.0971,0.1064,0.1369,0.1017,0.1131,0.1038,0.1090,
    0.1404,0.1463,0.1326,0.1555,0.1640,0.1701,0.1646,0.1649,0.1694,0.1784,
    0.2057,0.2062,0.1808,0.1888,0.1876,0.1785,0.1858,0.1852,0.1926,0.1856,
    0.1892,0.1932,0.1899,0.1874,0.1850,0.1847,0.1903,0.1975,0.1991,0.1975,
    0.1956,0.1905,0.1953,0.1900,0.1891,0.1936,0.1925,0.1870
]

plt.figure(figsize=(8,5))
plt.plot(epochs_bin, train_loss_bin, 'b-', linewidth=2, label='训练损失')
plt.plot(epochs_bin, val_loss_bin, 'r-', linewidth=2, label='验证损失')
plt.xlabel('训练周期', fontsize=12)
plt.ylabel('损失值', fontsize=12)
plt.title('二分类模型训练与验证损失曲线', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('binary_loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()




cm_multi = np.array([[37,0,16,0],
                     [12,2,15,0],
                     [8,0,30,0],
                     [0,0,9,0]])
classes = ['健康', '轻度', '中度', '重度']
plt.figure(figsize=(6,5))
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes,
            linewidths=0.5, linecolor='white')
plt.xlabel('预测类别', fontsize=12)
plt.ylabel('真实类别', fontsize=12)
plt.title('多分类混淆矩阵', fontsize=14)
plt.tight_layout()
plt.savefig('multi_cm.png', dpi=300, bbox_inches='tight')
plt.show()




cm_binary = np.array([[37,11],
                      [6,75]])
plt.figure(figsize=(6,5))
sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues',
            xticklabels=['健康', '患病'], yticklabels=['健康', '患病'],
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 12})
plt.xlabel('预测类别', fontsize=12)
plt.ylabel('真实类别', fontsize=12)
plt.title('二分类混淆矩阵', fontsize=14)
plt.tight_layout()
plt.savefig('binary_cm.png', dpi=300, bbox_inches='tight')
plt.show()