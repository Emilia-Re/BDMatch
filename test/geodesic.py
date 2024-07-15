import numpy as np
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances

# 创建示例数据点（例如，这里是二维的）
data = np.array([
    [0.0, 0.0],
    [1.0, 1.0],
    [2.0, 2.0],
    [3.0, 3.0],
    [4.0, 4.0]
])

# 定义Isomap模型，设置邻居数和降维后的维度
n_neighbors = 3
n_components = 2
isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)

# 拟合数据并进行降维
isomap.fit(data)

# 获取在流形上的嵌入
embedding = isomap.transform(data)

# 计算流形上的距离矩阵
manifold_distances = pairwise_distances(embedding, metric='euclidean')

# 打印嵌入和流形上的距离矩阵
print("流形上的嵌入：")
print(embedding)
print("\n流形上的距离矩阵：")
print(manifold_distances)

# 示例：计算第一个点和第二个点在流形上的距离
distance = manifold_distances[0, 1]
print(f"\n第一个点和第二个点在流形上的距离是 {distance:.2f}")
