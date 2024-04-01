
## チュートリアルサイト
[OpenVINOでゼロから学ぶディープラーニング推論](https://monomonotech.jp/learning/contents/openvino_emotion_recognition/openvino_emotion_recognition/index.html)

## 注意点
OpenVINOは`2024.0.0`Ver.からIE周りが仕様変更されたため、上記のサイトとプログラムが違う箇所が多々あります。  
2024.0.0以前のバージョンを使う際はご注意ください。

## 環境
- Python@~3.10.14
- CMake@^3.29.0 (いらないかも？)

### ランタイムバージョン管理：
- mise@^2024.3.9

### Pythonパッケージ管理：
- Poetry@^1.8.2 

## Setup

```bash
git clone git@github.com:KorRyu3/OpenVINO-tutorial.git
cd OpenVINO-tutorial/monomonotech/
```

```bash
mise i
```

```bash
poetry install
```

## Usage
### 内部カメラを使用した、人の顔の物体検出
```bash
# エラーが出たら、video_face_detection.py の 25行目のコメントアウトを外してみてください。
poetry run python3 src/video_face_detection.py
```

### 内部カメラを使用した、人の顔の物体検出と感情検出
```bash
# エラーが出たら、video_face_detection_and_emotion.py の 54行目のコメントアウトを外してみてください。
poetry run python3 src/video_face_detection_and_emotion.py
```
