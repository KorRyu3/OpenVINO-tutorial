import numpy as np

a = np.array([1, 2, 3, 4])
print(a)
print(a.ndim)
print(a.shape)
print("----------")

b = np.array([[1, 2, 3, 4]])
print(b)
print(b.ndim)
print(b.shape)
print("----------")

c = np.array([[[1, 2, 3, 4]]])
print(c)
print(c.ndim)
print(c.shape)

# 実行結果
# [1 2 3 4]
# 1
# (4,)
# ----------
# [[1 2 3 4]]
# 2
# (1, 4)
# ----------
# [[[1 2 3 4]]]
# 3
# (1, 1, 4)


# squeezeは大きさが1である次元を削除する関数
# 余分な[]を取り除き、次元を減らす

a = np.array([1, 2, 3, 4])
a = np.squeeze(a)
print(a)
print(a.ndim)
print(a.shape)
print("----------")

b = np.array([[1, 2, 3, 4]])
b = np.squeeze(b)
print(b)
print(b.ndim)
print(b.shape)
print("----------")

c = np.array([[[1, 2, 3, 4]]])
c = np.squeeze(c)
print(c)
print(c.ndim)
print(c.shape)

# 実行結果
# [1 2 3 4]
# 1
# (4,)
# ----------
# [1 2 3 4]
# 1
# (4,)
# ----------
# [1 2 3 4]
# 1
# (4,)


# expand_dimsは逆に大きさ１の次元を追加することができます。
# 次元数のフォーマットを合わせる際に使う場合があります。

a = np.array([1, 2, 3, 4])
a = np.expand_dims(a, axis=0)
print(a)
print(a.ndim)
print(a.shape)
print("----------")

b = np.array([[1, 2, 3, 4]])
b = np.expand_dims(b, axis=0)
print(b)
print(b.ndim)
print(b.shape)
print("----------")

c = np.array([[[1, 2, 3, 4]]])
c = np.expand_dims(c, axis=0)
print(c)
print(c.ndim)
print(c.shape)

# 実行結果
# [[1 2 3 4]]
# 2
# (1, 4)
# ----------
# [[[1 2 3 4]]]
# 3
# (1, 1, 4)
# ----------
# [[[[1 2 3 4]]]]
# 4
# (1, 1, 1, 4)
