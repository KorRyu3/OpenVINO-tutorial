import cv2
import numpy as np

a = np.array([10, 50, 40, 30, 20])
# np.argmax は 配列（リストやndarray）の中で最大値の要素のインデックスを値で返します。
b = np.argmax(a)
print(b)

# 実行結果
# 1

image_file = "images/cat.png"
img = cv2.imread(image_file)

print(type(img))
print(img.shape)

cv2.imshow("image", img[200:400, 300:600])

cv2.waitKey(500)
cv2.destroyAllWindows()

# 実行結果
# <class 'numpy.ndarray'>
# (高さ, 幅, カラーチャンネル)
# カラーチャンネルは 青と緑と赤の3色
# (600, 800, 3)


# ndimはndarrayの次元数を参照
a = np.array([1, 2, 3, 4])
print(a.ndim)

b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(b.ndim)

c = np.array(
    [
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
    ]
)
print(c.ndim)

# 実行結果
# 1
# 2
# 3


a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print(a.shape)

print(a.transpose(0, 1))  # 変わらない
print("----------")

# axisで指定する
b = a.transpose(1, 0)
print(b)

# 実行結果
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
# ----------
# [[ 1  5  9]
#  [ 2  6 10]
#  [ 3  7 11]
#  [ 4  8 12]]


image_file = "images/cat.png"
img = cv2.imread(image_file)

cv2.imshow("image", img.transpose(1, 0, 2))
# axisの変化
# 0 -> 1
# 1 -> 0
# 2 -> 2

cv2.waitKey(750)
cv2.destroyAllWindows()

# (Height, Width, Channel)になります。この頭文字を取ってHWCと略します
# 今回取り扱う推論エンジンの画像フォーマットはCHW

img = img.transpose((2, 0, 1))  # HWC > CHW
# axisの変化
# 0 -> 2
# 1 -> 0
# 2 -> 1

print(img.shape)
print(img.ndim)
