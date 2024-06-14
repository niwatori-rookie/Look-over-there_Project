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
MLP（多層パーセプトロン）を使って、手のランドマークデータを処理。
<br>
<br>

![image](https://github.com/niwatori-rookie/Look-over-there_Project/assets/138978518/35baa9b1-1e4a-4e48-a4ba-3ac4c1ef0751)
<br>
<br>
![image](https://github.com/niwatori-rookie/Look-over-there_Project/assets/138978518/493e1876-9ecd-4536-b04d-ce8c9c29f79c)
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

<br>


・動画等



・参照サイト
----------------------------------------------------

https://tpsxai.com/raspberrypi_tensorflow_lite/

https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe?tab=readme-ov-file

