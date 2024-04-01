import cv2
import numpy as np
import openvino as ov

from pngoverlay import PNGOverlay

# アイコン画像のインスタンス生成
icon_neutral = PNGOverlay("icon_emotion/icon_neutral.png")
icon_happy = PNGOverlay("icon_emotion/icon_happy.png")
icon_sad = PNGOverlay("icon_emotion/icon_sad.png")
icon_surprise = PNGOverlay("icon_emotion/icon_surprise.png")
icon_anger = PNGOverlay("icon_emotion/icon_anger.png")

# インスタンス変数をリストにまとめる
icon_emotion = [icon_neutral, icon_happy, icon_sad, icon_surprise, icon_anger]

# IEコアの初期化
core = ov.Core()

# モデルの準備（顔検出）
file_path_face = "intel/face-detection-retail-0004/FP32/face-detection-retail-0004"
model_face = file_path_face + ".xml"
weights_face = file_path_face + ".bin"

# モデルの準備（感情分類）
file_path_emotion = (
    "intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003"
)
model_emotion = file_path_emotion + ".xml"
weights_emotion = file_path_emotion + ".bin"

# モデルの読み込み（顔検出）
model = core.read_model(model=model_face, weights=weights_face)
compiled_model_face = core.compile_model(model=model, device_name="CPU")
del model

# モデルの読み込み（感情分類）
model = core.read_model(model=model_emotion, weights=weights_emotion)
compiled_model_emotion = core.compile_model(model=model, device_name="CPU")
del model

# 入出力データのキー取得
input_layer_ir_face = compiled_model_face.input(0)
output_layer_ir_face = compiled_model_face.output(0)

input_layer_ir_emotion = compiled_model_emotion.input(0)
output_layer_ir_emotion = compiled_model_emotion.output(0)

# PCの内蔵カメラやUSBカメラ・Webカメラの映像を読み込む場合はVideoCapture()の引数にカメラの番号（ID）を指定する。
# カメラの番号は、内蔵カメラが0、さらにUSBで追加のカメラを接続すると1のように基本的には0から順番に割り当てられているはず（公式ドキュメントのチュートリアルによると-1の場合もあるとのこと）。
# とりあえず順番に試してみればよい。
# また、動画ファイルを指定したいなら、そこまでのpathを指定
cap = cv2.VideoCapture(1)  # macbook Airの内部カメラは1だった
# cap = cv2.VideoCapture(0)

print("任意のキーを押すと終了します")
while True:
    ret, frame = cap.read()
    # 入力データフォーマットへ変換
    img = cv2.resize(frame, (300, 300))  # サイズ変更
    img = img.transpose((2, 0, 1))  # HWC > CHW
    img = np.expand_dims(img, axis=0)  # 次元合せ

    # 推論実行
    out = compiled_model_face([img])

    # 出力から必要なデータのみ取り出し
    out = out[output_layer_ir_face]
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

            # 顔検出領域は入力画像範囲内に補正する。特にminは補正しないとエラーになる
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > frame.shape[1]:
                xmax = frame.shape[1]
            if ymax > frame.shape[0]:
                ymax = frame.shape[0]

            # 顔領域のみ切り出し
            frame_face = frame[ymin:ymax, xmin:xmax]

            # 入力データフォーマットへ変換
            img = cv2.resize(frame_face, (64, 64))  # サイズ変更
            img = img.transpose((2, 0, 1))  # HWC > CHW
            img = np.expand_dims(img, axis=0)  # 次元合せ

            # 推論実行
            out = compiled_model_emotion([img])

            # 出力から必要なデータのみ取り出し
            out = out[output_layer_ir_emotion]
            out = np.squeeze(out)  # 不要な次元の削減

            # 出力値が最大のインデックスを得る
            index_max = np.argmax(out)

            # 各感情の文字列をリスト化
            list_emotion = ["neutral", "happy", "sad", "surprise", "anger"]

            # 文字列描画
            # cv2.putText(
            #     frame,
            #     list_emotion[index_max],
            #     (20, 60),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     2,
            #     (240, 180, 0),
            #     4,
            # )

            # バウンディングボックス表示
            cv2.rectangle(
                frame, (xmin, ymin), (xmax, ymax), color=(240, 180, 0), thickness=3
            )

            # 棒グラフ表示
            str_emotion = ["neu", "hap", "sad", "sur", "ang"]
            text_x = 10
            text_y = frame.shape[0] - 180
            rect_x = 80
            rect_y = frame.shape[0] - 200
            for i in range(5):
                cv2.putText(
                    frame,
                    str_emotion[i],
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (240, 180, 0),
                    2,
                )
                cv2.rectangle(
                    frame,
                    (rect_x, rect_y),
                    (rect_x + int(300 * out[i]), rect_y + 20),
                    color=(240, 180, 0),
                    thickness=-1,
                )
                text_y += 40
                rect_y += 40

            # 顔アイコン表示
            icon_emotion[index_max].show(
                frame,
                frame.shape[1] - 110,
                frame.shape[0] - 110
            )

            # １つの顔で終了
            break

    # 画像表示
    cv2.imshow("frame", frame)

    # キーが押されたら終了
    key = cv2.waitKey(10)
    if key != -1:
        break
