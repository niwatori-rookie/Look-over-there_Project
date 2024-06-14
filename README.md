

# あっち向いてほい（以下再編集したもの）


・利用したプログラミング言語やフレームワーク
----------------------------------------------------
<br>

**プログラミング言語：Python**

<br>

**フレームワーク：pygame , OpenCV , media pipe , dilb, imutils , numpy , itertools , collections等**
<br>
<br>


・ポートフォリオの概要と目的
----------------------------------------------------
<br>

**・目的**
<br>
-------------------------------------------------------

<br>
次の「ジェスチャーゲーム」の作成に向けてプログラムの組み方を学び、機械学習でできる範囲を知るため。
<br>
<br>
<br>

**・概要**
<br>
----------------------------------------------------
<br>
「最初はグー、じゃんけんポン」の合図で、ユーザーの手の形をコンピュータ側が（グー or チョキ or パーを）認識し、その結果（勝ち　or 負け）かによって以下のようなあっち向いてほいの処理を行う。

<br>
<br>
<br>
1:勝ち　→　ユーザー：手、コンピュータ側：ランダム値を出して、ユーザーの手の上下左右を認識。
<br>

2:負け　→　ユーザー：顔、コンピュータ側：ランダム値を出して、ユーザーの手の上下左右を認識。






・プログラムの流れ
----------------------------------------------------
<br>

**・プログラムの特徴と苦労した点**
<br>
**・じゃんけんのテストケース**

![image](https://github.com/niwatori-rookie/Look-over-there_Project/assets/138978518/0d136b7f-7ed8-41c2-ab16-1e052562001c)

<br>

### hand_pose()の処理について
----------------------------------------------------
<br>

**MLP（多層パーセプトロン）を使って、手のランドマークデータを処理。**
<br>
<br>

![image](https://github.com/niwatori-rookie/Look-over-there_Project/assets/138978518/35baa9b1-1e4a-4e48-a4ba-3ac4c1ef0751)
<br>
<br>
![image](https://github.com/niwatori-rookie/Look-over-there_Project/assets/138978518/493e1876-9ecd-4536-b04d-ce8c9c29f79c)
<br>
<br>

※以下、a.py（モデルの定義と推論）
<br>
<br>
このソースコードは、手のランドマークデータを用いた手のジェスチャー認識モデルをTensorFlowを使用して訓練し、保存、評価、推論、TensorFlow Lite形式への変換を行う一連の処理を行っています。具体的な処理内容を以下に説明します。

### ライブラリのインポート
```python
import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
```
- `csv`: CSVファイルの操作用ライブラリ。
- `numpy`: 数値計算ライブラリ。
- `tensorflow`: TensorFlowライブラリ。
- `sklearn.model_selection.train_test_split`: データセットを訓練用とテスト用に分割するための関数。

### 定数の設定
```python
RANDOM_SEED = 42

dataset = 'C:/Users/p-user/project/hand-gesture-recognition-using-mediapipe-main/hand-gesture-recognition-using-mediapipe-main/model/keypoint_classifier/keypoint.csv'
model_save_path = 'C:/Users/p-user/project/hand-gesture-recognition-using-mediapipe-main/hand-gesture-recognition-using-mediapipe-main/model/keypoint_classifier/keypoint_classifier.keras'

NUM_CLASSES = 4
```
- `RANDOM_SEED`: 乱数のシード値を設定し、再現性を確保。
- `dataset`: ランドマークデータが保存されているCSVファイルのパス。
- `model_save_path`: 訓練後のモデルを保存するパス。
- `NUM_CLASSES`: クラス数を設定（ここでは4クラス）。

### データの読み込みと分割
```python
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)
```
- `X_dataset`: ランドマークデータ（入力データ）を読み込み。
- `y_dataset`: ラベルデータ（出力データ）を読み込み。
- `train_test_split`: データを訓練用とテスト用に分割。

### モデルの定義
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2, )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
```
- シーケンシャルモデルを定義。
- 入力層は42次元（21個のランドマーク×2次元）。
- 隠れ層にはドロップアウト層と全結合層を配置。

### モデルの概要表示
```python
model.summary()  # tf.keras.utils.plot_model(model, show_shapes=True)
```
- モデルの構造を表示。

### コールバックの設定
```python
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)
```
- モデルチェックポイント: 訓練中にモデルを保存。
- 早期停止: 訓練の早期停止を設定。

### モデルのコンパイル
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```
- モデルをコンパイル。
- 最適化関数はAdam、損失関数は`sparse_categorical_crossentropy`。

### モデルの訓練
```python
model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback]
)
```
- モデルを訓練。
- 訓練データと検証データを設定し、コールバックを使用。

### モデルの評価
```python
val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)
```
- テストデータを使用してモデルを評価。

### モデルの保存とロード
```python
model.save(model_save_path)
model = tf.keras.models.load_model(model_save_path)
```
- 訓練後のモデルを保存し、再度ロード。

### 推論テスト
```python
predict_result = model.predict(np.array([X_test[0]]))
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))
```
- テストデータの一部を使って推論を行い、結果を表示。

### TensorFlow Lite形式への変換
```python
tflite_save_path = 'C:/Users/p-user/project/hand-gesture-recognition-using-mediapipe-main/hand-gesture-recognition-using-mediapipe-main/model/keypoint_classifier/keypoint_classifier.tflite'

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

open(tflite_save_path, 'wb').write(tflite_quantized_model)
```
- モデルをTensorFlow Lite形式に変換し、量子化。
- 変換したモデルを保存。

### TensorFlow Liteモデルのロードと推論
```python
interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()

# 入出力テンソルを取得
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))

# 推論実施
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])

print(np.squeeze(tflite_results))
print(np.argmax(np.squeeze(tflite_results)))
```
- TensorFlow Liteインタープリターを初期化し、テンソルを割り当て。
- 入力テンソルにデータをセットし、推論を実行。
- 結果を表示。

この一連の処理により、手のランドマークデータを使用して手のジェスチャーを分類するモデルを訓練し、TensorFlow Lite形式に変換して軽量化されたモデルを使って推論を行うことができます。

<br>
<br>
<br>
<br>


model/point_history_classifier
<br>
フィンガージェスチャー認識に関わるファイルを格納するディレクトリです。
<br>



<br>
<br>


また以下のファイルが格納されます。

学習用データ(point_history.csv)

学習済モデル(point_history_classifier.tflite)

ラベルデータ(point_history_classifier_label.csv)

推論用クラス(point_history_classifier.py)


### face_poseの処理について
----------------------------------------------------

**dlibを使って顔の向き（yaw、pitch、roll）を推定。**

<br>


・動画等



・参照サイト
----------------------------------------------------

https://tpsxai.com/raspberrypi_tensorflow_lite/

https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe?tab=readme-ov-file

｛最適化関数　Adamについて｝
<br>
https://qiita.com/Fumio-eisan/items/798351e4915e4ba396c2
<br>
<br>
｛正規化と標準化について｝
<br>
https://qiita.com/yShig/items/dbeb98598abcc98e1a57
<br>
｛顔の向き、dlibについて｝
<br>
https://qiita.com/oozzZZZZ/items/1e68a7572bc5736d474e
<br>
https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
