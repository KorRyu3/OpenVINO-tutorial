import cv2

cap = cv2.VideoCapture(1)  # 内部カメラは0, 外部カメラは1
# MacBook Airの内部カメラは1らしい


while True:
    # frameには1F毎の静止画像が格納されている
    ret, frame = cap.read()
    cv2.imshow('image', frame)
    key = cv2.waitKey(1)
    # key=-1の時は、キー入力がないとき
    if key != -1:
        break

cap.release()
cv2.destroyAllWindows()
