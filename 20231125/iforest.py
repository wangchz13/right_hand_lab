
# import csv
# import pandas as pd
# file = r"dataset/GAS/new_gas_drop_null.csv"

# df = pd.read_csv(file)
# for column_name, column_data in df.iloc[:, 1:].items():
#     # 创建新文件名，去除可能存在的非法字符
#     new_file_name = f"dataset/GAS/{column_name}.csv"

#     # 将列数据保存到新文件
#     column_data.to_csv(new_file_name, index=False,header=False)

#     print(f"Saved column {column_name} to {new_file_name}")



from cProfile import label
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from matplotlib.figure import FigureBase


data = pd.read_csv(r"\\wsl.localhost\Ubuntu\home\wcz\github\iTransformer\dataset\GAS\1345.csv",header=None)

# 提取每天每半小时的用气量作为特征
# X = df['gas_usage']
X = data[0].to_numpy().reshape((-1,48))
# 使用孤立森林（Isolation Forest）模型检测异常值
model = IsolationForest(contamination=0.002)  # 设置异常值比例
ans = model.fit_predict(X)



# 设置窗口状态为最大化
# 打印异常值
show = [130,174]

fig, axs = plt.subplots(2, 4, figsize=(12, 6))
fig.tight_layout(pad=3.0)


# for i in range(len(ans)):
#     if ans[i]==1:
#         if i >=show[0] and i <=show[1]:
#             row = (i%7) // 4
#             col = (i%7) % 4
#             axs[row, col].plot(X[i],label=i)
#             axs[row, col].legend()

flag = -1
for i in range(len(ans)):
    if ans[i]==-1:continue
    if i >=show[0] and i <=show[1]:
        flag += 1
        row = (flag//7) // 4
        col = (flag//7) % 4
        axs[row, col].plot(X[i],label=i)
        axs[row, col].legend()

plt.show()
   
