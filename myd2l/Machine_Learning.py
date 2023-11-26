"""
看到的一篇博文，学习记录一下：https://mp.weixin.qq.com/s/Kf-PwSH7v51WtgOK1eHPdw
    1.PCA主成分分析
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load iris dataset as an example
iris = load_iris()
X = iris.data       # (150,4)
y = iris.target     # (150,)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (important for PCA)
scaler = StandardScaler()           # 去均值和方差归一化，是针对feature维度执行的，不是针对样本
# 对部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），然后对该X_train进行转换transform，从而实现数据的标准化、归一化等
# 根据对之前部分X_train进行fit的整体指标，对剩余的数据（X_test）使用同样的均值、方差、最大最小值等指标进行转换transform(X_test)，从而保证train、test处理方式相同
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Apply PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)        # 对归一化后的X_train_std进行PCA

# Calculate the cumulative explained variance 计算累计解释方差
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)        # np.cumsum()沿指定维度计算累积和；每个被选择的组成部分解释的方差百分比。

# Determine the number of components to keep for 85% variance explained     # 要解释85%的方差时要保留的分量数，降至多少维
n_components = np.argmax(cumulative_variance_ratio >= 0.85) + 1

# Apply PCA with the selected number of components
pca = PCA(n_components=n_components)                # 要保留的组件数[降至多少维]
X_train_pca = pca.fit_transform(X_train_std)        # [120,4]->[120,2]
X_test_pca = pca.transform(X_test_std)              # [30,4]  ->[30,2]

# Display the results
print("Original Training Data Shape:", X_train.shape)
print("Reduced Training Data Shape (PCA):", X_train_pca.shape)
print("Number of Components Selected:", n_components)