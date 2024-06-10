import pygame
import time
import cv2 as cv
import random
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from keypoint_classifier import KeyPointClassifier
from point_history_classifier import PointHistoryClassifier
import csv
from utils import CvFpsCalc
import numpy as np


import argparse
import itertools
from collections import Counter
from collections import deque
import copy

import dlib #機械学習系ライブラリ
import imutils #OpenCVの補助
from imutils import face_utils



class Look_over_there(object):
    def __init__(self,font,black,width,height,screen):
        self.font = font
        self.black = black
        self.width = width
        self.height = height
        self.screen = screen
        self.white = (255, 255, 255)
        self.window_rect = screen.get_rect(center=(width // 2, height // 2))
        # 画像の読み込み
        self.up_img = pygame.image.load("img_file/point_up.png")
        self.down_img = pygame.image.load("img_file/point_down.png")
        self.right_img = pygame.image.load("img_file/point_right.png")
        self.left_img = pygame.image.load("img_file/point_left.png")
        

    def atti(self):
        # 画面に表示する文字列を作成
        text = "あっちむいて......."
        text_render = self.font.render(text, True, self.black)
        text_rect = text_render.get_rect(center=(self.width // 2, self.height // 2 - 200))
        self.screen.blit(text_render, text_rect)
        pygame.display.flip()
        pygame.time.delay(1000)
        self.screen.fill(self.white)
        pygame.display.flip()
        pygame.time.delay(1000)
        text = "ほい！"
        text_render = self.font.render(text, True, self.black)
        text_rect = text_render.get_rect(center=(self.width // 2, self.height // 2 - 200))
        self.screen.blit(text_render, text_rect)
        pygame.display.flip()
        pygame.time.delay(500)
    
    

    def calc_bounding_rect(self,image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x   , y, w, h = cv.boundingRect(landmark_array)
    
        return [x, y, x + w, y + h]
    

    def calc_landmark_list(self,image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # キーポイント
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point
    



    def pre_process_landmark(self,landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # 相対座標に変換
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # 1次元リストに変換
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # 正規化
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list
    



    def pre_process_point_history(self,image, point_history):
        image_width, image_height = image.shape[1], image.shape[0]

        temp_point_history = copy.deepcopy(point_history)

        # 相対座標に変換
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]

            temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

        # 1次元リストに変換
        temp_point_history = list(
            itertools.chain.from_iterable(temp_point_history))

        return temp_point_history



    def hand_pose(self):
        # ゲームステートの定義
        parser = argparse.ArgumentParser()

        parser.add_argument("--device", type=int, default=0)
        parser.add_argument("--width", help='cap width', type=int, default=960)
        parser.add_argument("--height", help='cap height', type=int, default=540)

        parser.add_argument('--use_static_image_mode', action='store_true')
        parser.add_argument("--min_detection_confidence",
                    help='min_detection_confidence',
                    type=float,
                    default=0.7)
        parser.add_argument("--min_tracking_confidence",
                    help='min_tracking_confidence',
                    type=float,  # typeをfloatに修正
                    default=0.5)

        args = parser.parse_args()
        # 引数解析 #################################################################

    

        use_static_image_mode = args.use_static_image_mode
        min_detection_confidence = args.min_detection_confidence
        min_tracking_confidence = args.min_tracking_confidence

        use_brect = True  # 全角スペースを半角スペースに修正
        # カメラ準備 ###############################################################
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

        # モデルロード #############################################################
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            )
        
        keypoint_classifier = KeyPointClassifier()
        point_history_classifier = PointHistoryClassifier()

        # ラベル読み込み ###########################################################
        with open('C:/Users/p-user/project/hand-gesture-recognition-using-mediapipe-main/hand-gesture-recognition-using-mediapipe-main/model/keypoint_classifier/keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
            ]
        
        # FPS計測モジュール ########################################################
        cvFpsCalc = CvFpsCalc(buffer_len=10)

        # 座標履歴 #################################################################
        history_length = 16
        point_history = deque(maxlen=history_length)

        # フィンガージェスチャー履歴 ################################################
        finger_gesture_history = deque(maxlen=history_length)

        a = random.choice([0, 1, 2, 3])
        check = 0
        
        while check <=0:
            # カメラキャプチャ #####################################################
            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)  # ミラー表示
            debug_image = copy.deepcopy(image)

            # 検出実施 #############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # BGR から RGB

            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

                #  ####################################################################
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                results.multi_handedness):
                    # 外接矩形の計算
                    brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                    # ランドマークの計算
                    landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)

                    # 相対座標・正規化座標への変換
                    pre_processed_landmark_list = self.pre_process_landmark(
                        landmark_list)
                    pre_processed_point_history_list = self.pre_process_point_history(
                        debug_image, point_history)
                    


                    # ハンドサイン分類
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    

                    if(a == 0):
                        current_image = self.up_img
                        img_rect = current_image.get_rect(center=(self.width // 2, self.height // 2 + 50))
                        self.screen.blit(current_image, img_rect)
                        pygame.display.flip()
                        if(hand_sign_id == 0):
                            pygame.time.delay(2000)
                            self.screen.fill(self.white)
                            text = "あなたの勝ち！"
                            text_render = self.font.render(text, True, self.black)
                            text_rect = text_render.get_rect(center=self.window_rect.center)
                            self.screen.blit(text_render, text_rect)
                            pygame.display.flip()
                            pygame.time.delay(2000)
                            check = 1
                        else:
                            pygame.time.delay(2000)
                            self.screen.fill(self.white)
                            text = "失敗 (-_-):ユーザー"
                            text_render = self.font.render(text, True, self.black)
                            text_rect = text_render.get_rect(center=self.window_rect.center)
                            self.screen.blit(text_render, text_rect)
                            pygame.display.flip()
                            pygame.time.delay(2000)
                            check = 1

                    elif(a == 1):
                        current_image = self.down_img
                        img_rect = current_image.get_rect(center=(self.width // 2, self.height // 2 + 50))
                        self.screen.blit(current_image, img_rect)
                        pygame.display.flip()
                        if(hand_sign_id == 1):
                            pygame.time.delay(2000)
                            self.screen.fill(self.white)
                            text = "あなたの勝ち！"
                            text_render = self.font.render(text, True, self.black)
                            text_rect = text_render.get_rect(center=self.window_rect.center)
                            self.screen.blit(text_render, text_rect)
                            pygame.display.flip()
                            pygame.time.delay(2000)
                            check = 1
                        else:
                            pygame.time.delay(2000)
                            self.screen.fill(self.white)
                            text = "失敗 (-_-):ユーザー"
                            text_render = self.font.render(text, True, self.black)
                            text_rect = text_render.get_rect(center=self.window_rect.center)
                            self.screen.blit(text_render, text_rect)
                            pygame.display.flip()
                            pygame.time.delay(2000)
                            check = 1
                    elif(a == 2):
                        current_image = self.right_img
                        img_rect = current_image.get_rect(center=(self.width // 2, self.height // 2 + 50))
                        self.screen.blit(current_image, img_rect)
                        pygame.display.flip()
                        if(hand_sign_id == 2):
                            pygame.time.delay(2000)
                            self.screen.fill(self.white)
                            text = "あなたの勝ち！"
                            text_render = self.font.render(text, True, self.black)
                            text_rect = text_render.get_rect(center=self.window_rect.center)
                            self.screen.blit(text_render, text_rect)
                            pygame.display.flip()
                            pygame.time.delay(2000)
                            check = 1
                        else:
                            pygame.time.delay(2000)
                            self.screen.fill(self.white)
                            text = "失敗 (-_-):ユーザー"
                            text_render = self.font.render(text, True, self.black)
                            text_rect = text_render.get_rect(center=self.window_rect.center)
                            self.screen.blit(text_render, text_rect)
                            pygame.display.flip()
                            pygame.time.delay(2000)
                            check = 1
                    elif(a == 3):
                        current_image = self.left_img
                        img_rect = current_image.get_rect(center=(self.width // 2, self.height // 2 + 50))
                        self.screen.blit(current_image, img_rect)
                        pygame.display.flip()
                        if(hand_sign_id == 3):
                            pygame.time.delay(2000)
                            self.screen.fill(self.white)
                            text = "あなたの勝ち！"
                            text_render = self.font.render(text, True, self.black)
                            text_rect = text_render.get_rect(center=self.window_rect.center)
                            self.screen.blit(text_render, text_rect)
                            pygame.display.flip()
                            pygame.time.delay(2000)
                            check = 1
                        else:
                            pygame.time.delay(2000)
                            self.screen.fill(self.white)
                            text = "失敗 (-_-):ユーザー"
                            text_render = self.font.render(text, True, self.black)
                            text_rect = text_render.get_rect(center=self.window_rect.center)
                            self.screen.blit(text_render, text_rect)
                            pygame.display.flip()
                            pygame.time.delay(2000)
                            check = 1
    
        cap.release()
        cv.destroyAllWindows()





    #頭部姿勢推定
    def face_pose(self):
        DEVICE_ID = 0 #　使用するカメラのID 0は標準webカメラ
        capture = cv.VideoCapture(DEVICE_ID)#dlibの学習済みデータの読み込み
        predictor_path = "C:/Users/p-user/project/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat"
        #学習済みdatファイルのパスをコピペ

        detector = dlib.get_frontal_face_detector() #顔検出器の呼び出し。ただ顔だけを検出する。
        predictor = dlib.shape_predictor(predictor_path) #顔から目鼻などランドマークを出力する
        check_2 = 0
        # 顔の向きを分類する
        direction = -1
        b = random.choice([0, 1, 2, 3])
        while check_2 <=0:
            ret, frame = capture.read() #カメラからキャプチャしてframeに１コマ分の画像データを入れる
            if not ret:
                break
            frame = cv.flip(frame, 1)  # ミラー表示
            frame = imutils.resize(frame, width=1000) #frameの画像の表示サイズを整える
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #gray scaleに変換する
            rects = detector(gray, 0) #grayから顔を検出
            image_points = None




        



            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                for (x, y) in shape: #顔全体の68箇所のランドマークをプロット
                    cv.circle(frame, (x, y), 1, (255, 255, 255), -1)

                image_points = np.array([
                    tuple(shape[30]),#鼻頭
                    tuple(shape[21]),
                    tuple(shape[22]),
                    tuple(shape[39]),
                    tuple(shape[42]),
                    tuple(shape[31]),
                    tuple(shape[35]),
                    tuple(shape[48]),
                    tuple(shape[54]),
                    tuple(shape[57]),
                    tuple(shape[8]),
                    ],dtype='double')

            if len(rects) > 0:
                cv.FONT_HERSHEY_PLAIN,( 0.7, (0, 0, 255), 2)
                model_points = np.array([
                (0.0,0.0,0.0), # 30
                (-30.0,-125.0,-30.0), # 21
                (30.0,-125.0,-30.0), # 22
                (-60.0,-70.0,-60.0), # 39
                (60.0,-70.0,-60.0), # 42
                (-40.0,40.0,-50.0), # 31
                (40.0,40.0,-50.0), # 35
                (-70.0,130.0,-100.0), # 48
                (70.0,130.0,-100.0), # 54
                (0.0,158.0,-10.0), # 57
                (0.0,250.0,-50.0) # 8
                ])

                size = frame.shape

                focal_length = size[1]
                center = (size[1] // 2, size[0] // 2) #顔の中心座標

                camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
                ], dtype='double')

                dist_coeffs = np.zeros((4, 1))

                (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix,
                                                                    dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
                #回転行列とヤコビアン
                (rotation_matrix, jacobian) = cv.Rodrigues(rotation_vector)
                mat = np.hstack((rotation_matrix, translation_vector))

                #yaw,pitch,rollの取り出し
                (_, _, _, _, _, _, eulerAngles) = cv.decomposeProjectionMatrix(mat)
                yaw = eulerAngles[1]
                pitch = eulerAngles[0]
                roll = eulerAngles[2]

                
                if yaw > 15:
                    direction = 2 #右
                elif yaw < -15:
                    direction = 3 #左
                if pitch > 15:
                    direction = 0 #上
                elif pitch < -15:
                    direction = 1 #下
            

            
            if(direction>=0):
                if(b == 0):
                    current_image = self.up_img
                    img_rect = current_image.get_rect(center=(self.width // 2, self.height // 2 + 50))
                    self.screen.blit(current_image, img_rect)
                    pygame.display.flip()
                    if(direction == 0):
                        pygame.time.delay(2000)
                        self.screen.fill(self.white)
                        text = "あなたの負け！"
                        text_render = self.font.render(text, True, self.black)
                        text_rect = text_render.get_rect(center=self.window_rect.center)
                        self.screen.blit(text_render, text_rect)
                        pygame.display.flip()
                        pygame.time.delay(2000)
                        check_2 = 1
                    else:
                        pygame.time.delay(2000)
                        self.screen.fill(self.white)
                        text = "失敗 (-_-):pc"
                        text_render = self.font.render(text, True, self.black)
                        text_rect = text_render.get_rect(center=self.window_rect.center)
                        self.screen.blit(text_render, text_rect)
                        pygame.display.flip()
                        pygame.time.delay(2000)
                        check_2 = 1

                elif(b == 1):
                    current_image = self.down_img
                    img_rect = current_image.get_rect(center=(self.width // 2, self.height // 2 + 50))
                    self.screen.blit(current_image, img_rect)
                    pygame.display.flip()
                    if(direction == 1):
                        pygame.time.delay(2000)
                        self.screen.fill(self.white)
                        text = "あなたの負け！"
                        text_render = self.font.render(text, True, self.black)
                        text_rect = text_render.get_rect(center=self.window_rect.center)
                        self.screen.blit(text_render, text_rect)
                        pygame.display.flip()
                        pygame.time.delay(2000)
                        check_2 = 1
                    else:
                        pygame.time.delay(2000)
                        self.screen.fill(self.white)
                        text = "失敗 (-_-):pc"
                        text_render = self.font.render(text, True, self.black)
                        text_rect = text_render.get_rect(center=self.window_rect.center)
                        self.screen.blit(text_render, text_rect)
                        pygame.display.flip()
                        pygame.time.delay(2000)
                        check_2 = 1
                elif(b == 2):
                    current_image = self.right_img
                    img_rect = current_image.get_rect(center=(self.width // 2, self.height // 2 + 50))
                    self.screen.blit(current_image, img_rect)
                    pygame.display.flip()
                    if(direction == 2):
                        pygame.time.delay(2000)
                        self.screen.fill(self.white)
                        text = "あなたの負け！"
                        text_render = self.font.render(text, True, self.black)
                        text_rect = text_render.get_rect(center=self.window_rect.center)
                        self.screen.blit(text_render, text_rect)
                        pygame.display.flip()
                        pygame.time.delay(2000)
                        check_2 = 1
                    else:
                        pygame.time.delay(2000)
                        self.screen.fill(self.white)
                        text = "失敗 (-_-):pc"
                        text_render = self.font.render(text, True, self.black)
                        text_rect = text_render.get_rect(center=self.window_rect.center)
                        self.screen.blit(text_render, text_rect)
                        pygame.display.flip()
                        pygame.time.delay(2000)
                        check_2 = 1
                elif(b == 3):
                    current_image = self.left_img
                    img_rect = current_image.get_rect(center=(self.width // 2, self.height // 2 + 50))
                    self.screen.blit(current_image, img_rect)
                    pygame.display.flip()
                    if(direction == 3):
                        pygame.time.delay(2000)
                        self.screen.fill(self.white)
                        text = "あなたの負け！"
                        text_render = self.font.render(text, True, self.black)
                        text_rect = text_render.get_rect(center=self.window_rect.center)
                        self.screen.blit(text_render, text_rect)
                        pygame.display.flip()
                        pygame.time.delay(2000)
                        check_2 = 1
                    else:
                        pygame.time.delay(2000)
                        self.screen.fill(self.white)
                        text = "失敗 (-_-) :pc"
                        text_render = self.font.render(text, True, self.black)
                        text_rect = text_render.get_rect(center=self.window_rect.center)
                        self.screen.blit(text_render, text_rect)
                        pygame.display.flip()
                        pygame.time.delay(2000)
                        check_2 = 1

        capture.release()
        cv.destroyAllWindows()