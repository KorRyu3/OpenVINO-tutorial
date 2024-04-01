import cv2
import numpy as np

# モジュール読み込み
import openvino as ov

# IEコアの初期化
core = ov.Core()

# モデルの準備（顔検出）
file_path_face = "intel/face-detection-retail-0004/FP32/face-detection-retail-0004"
model_face = file_path_face + ".xml"
weights_face = file_path_face + ".bin"

# モデルの読み込み（顔検出）
model = core.read_model(model=model_face, weights=weights_face)
compiled_model = core.compile_model(model=model, device_name="CPU")

# 入出力データのキー取得
input_layer_ir = compiled_model.input(0)
out_layer_ir = compiled_model.output(0)

# カメラ準備
cap = cv2.VideoCapture(1)
# cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
# 入力データフォーマットへ変換
    img = cv2.resize(frame, (300, 300))  # サイズ変更
    img = img.transpose((2, 0, 1))  # HWC > CHW
    img = np.expand_dims(img, axis=0)  # 次元合せ

    # 推論実行
    result = compiled_model([img])
    out = result[out_layer_ir]
    out = np.squeeze(out)  # サイズ1の次元を全て削除

    # 検出されたすべての顔領域に対して１つずつ処理
    for detection in out:
        # conf値の取得
        confidence = float(detection[2])

        # conf値が0.5より大きい場合のみ感情推論とバウンディングボックス表示
        if confidence > 0.5:

            # バウンディングボックス座標を入力画像のスケールに変換
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])
            cv2.rectangle(
                frame, (xmin, ymin), (xmax, ymax), color=(240, 180, 0), thickness=3
            )

            # 画像表示
            cv2.imshow("frame", frame)

    # キーが押されたら終了
    # 何ms待機するか
    wait_time = 10
    key = cv2.waitKey(wait_time)
    if key != -1:
        break
