import cv2
import numpy as np

# モジュール読み込み
import openvino as ov

# IEコアの初期化
core = ov.Core()

available_devices = core.available_devices
print(available_devices)  # 使えるデバイスの確認

cpu_device_name = core.get_property("CPU", "FULL_DEVICE_NAME")  # デバイス名の取得
print(cpu_device_name)

# モデルの準備（感情分類）
file_path_emotion = (
    "intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003"
)
model_emotion = file_path_emotion + ".xml"
weights_emotion = file_path_emotion + ".bin"

# モデルの読み込み（感情分類）
model = core.read_model(model=model_emotion, weights=weights_emotion)
compiled_model = core.compile_model(model=model, device_name="CPU")

# 入出力データのキー取得
input_layer_ir = compiled_model.input(0)
output_layer_ir = compiled_model.output(0)

# 入力画像読み込み
img_emotion = cv2.imread("images/face.png")

IMG_WIDTH = img_emotion.shape[1]
IMG_HEIGHT = img_emotion.shape[0]

# 入力データフォーマットへ変換
img = cv2.resize(img_emotion, (64, 64))  # サイズ変更
img = img.transpose((2, 0, 1))  # HWC > CHW
img = np.expand_dims(img, axis=0)  # 次元合せ

# 推論実行
result = compiled_model([img])
# 出力値の取得
result = result[output_layer_ir]
out = np.squeeze(result)  # サイズ1の次元を全て削除

print(out)


# 出力値が最大のインデックスを得る
index_max = np.argmax(out)

# 各感情の文字列をリスト化
list_emotion = ["neutral", "happy", "sad", "surprise", "anger"]

# 文字列描画
cv2.putText(
    img_emotion,
    list_emotion[index_max],
    (IMG_WIDTH // 4, IMG_HEIGHT - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 30, 100),  # BGR
    2,
)

# 画像表示
cv2.imshow("image", img_emotion)

# 終了処理
cv2.waitKey(0)
cv2.destroyAllWindows()
