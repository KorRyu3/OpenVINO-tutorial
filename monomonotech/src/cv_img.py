import cv2

image_file = 'images/cat.png'
img = cv2.imread(image_file)
# img = cv2.resize(img, (400, 300))

# cv2.putText(画像, 文字列, 位置(X, Y), フォント, 倍率, 色(B, G, R), 線の太さ)
cv2.putText(img, "Please Don't Disturb.", (80, 530), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 8)

# cv2.rectangle(画像, 左上座標(X, Y), 右下座標(X, Y), 色(B, G, R), 線の太さ)
cv2.rectangle(img, (360, 240), (550, 280), (20, 20, 20), -1)

# cv2.lineで直線描画
# cv2.circleで円描画
# cv2.ellipseで楕円描画

cv2.imshow('image', img)  # 上のウィンドウ名

# 画像が表示ウィンドウにフォーカスがあるとき、何かキーを押すとプログラムが終了
cv2.waitKey(3000)
cv2.destroyAllWindows()
