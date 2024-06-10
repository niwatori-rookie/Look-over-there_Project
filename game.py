import pygame
import time
import cv2
import random
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from Look_over_there import Look_over_there


# Pygameの初期化
pygame.init()


# 画面のサイズ設定
width, height = 1200, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Look over there!")


# 色の定義
white = (255, 255, 255)
black = (0, 0, 0)

# 日本語対応フォントの読み込み
font_path = 'azuki.ttf'  # フォントファイルへのパス
font = pygame.font.Font(font_path, 100)


# 画像の読み込み
rock_img = pygame.image.load("img_file/rock.png")
paper_img = pygame.image.load("img_file/paper.png")
scissors_img = pygame.image.load("img_file/scissors.png")


# ゲームステートの定義
STATE_START = 0
STATE_PLAYING = 1
game_state = STATE_START

# カメラの初期化
cap = cv2.VideoCapture(0)

# Pygameウィンドウを中央に配置
window_rect = screen.get_rect(center=(width // 2, height // 2))


model_path = 'gesture_recognizer.task'


base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)


running = True
show_text = True
start_time = None
current_image = None
choice = None

#あいこの場合か判断する処理｛フラグ｝(0:あいこでない, 1:あいこである)
aiko = 0

#あっち向いてほいのフラグ（1:ユーザ側が勝った｛指｝2:ユーザ側が負けた｛顔｝）
Look_check = 0

Look = Look_over_there(font,black,width,height,screen)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            if game_state == STATE_START:
                game_state = STATE_PLAYING


    screen.fill(white)

    if game_state == STATE_START:
        start_text = font.render("Press SPACE to start", True, black)
        start_rect = start_text.get_rect(center=window_rect.center)
        screen.blit(start_text, start_rect)
    elif game_state == STATE_PLAYING:
        if show_text:
            if start_time is None:
                start_time = time.time()
            current_time = time.time()
            elapsed_time = current_time - start_time

            if aiko == 0 :
                if elapsed_time < 1:
                    text = "最初は...."
                elif elapsed_time < 2:
                    text = "グー"
                    current_image = rock_img
                elif elapsed_time < 3:
                    text = "じゃんけん...."
                    current_image = None
                else:
                    text = "ポン!"
                    choice = random.choice(["rock", "paper", "scissors"])
                    current_image = pygame.image.load(f"img_file/{choice}.png")
                    show_text = False  # テキスト表示終了
            else:
                if elapsed_time < 1:
                    text = "あいこで....."
                else:
                    pygame.time.delay(1700)
                    text = "しょ!"
                    choice = random.choice(["rock", "paper", "scissors"])
                    current_image = pygame.image.load(f"img_file/{choice}.png")
                    show_text = False  # テキスト表示終了



            text_render = font.render(text, True, black)
            text_rect = text_render.get_rect(center=(width // 2, height // 2 - 200))
            screen.blit(text_render, text_rect)


            if current_image is not None:
                img_rect = current_image.get_rect(center=(width // 2, height // 2 + 50))
                screen.blit(current_image, img_rect)
        else:
            # Webカメラから画像を取得して保存
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret:
                image_filename = "gesture.jpg"
                cv2.imwrite(image_filename, frame)
            cap.release()


            # 手のジェスチャー認識
            image = mp.Image.create_from_file(image_filename)
            recognition_result = recognizer.recognize(image)


            result_text = None
            aiko = 0
            Look_check = 0
            if recognition_result.gestures:
                gesture = recognition_result.gestures[0][0]
                if gesture.category_name == 'Closed_Fist':
                    if choice == 'rock':
                        result_text = 'Draw'
                        #あいこの場合の処理（必要）
                        aiko = 1
                    elif choice == 'scissors':
                        result_text = 'Win'
                        #あっち向いてほいクラス
                        Look_check = 1
                        
                    elif choice == 'paper':
                        result_text = 'Lose'
                        #あっち向いてほいクラス
                        Look_check = 2
                    

                elif gesture.category_name == 'Victory':
                    if choice == 'rock':
                        result_text = 'Lose'
                        #あっち向いてほいクラス
                        Look_check = 2

                    elif choice == 'scissors':
                        result_text = 'Draw'
                        #あいこの場合の処理（必要）
                        aiko = 1
                    elif choice == 'paper':
                        result_text = 'Win'
                        #あっち向いてほいクラス
                        Look_check = 1
                        

                elif gesture.category_name == 'Open_Palm':
                    if choice == 'rock':
                        result_text = 'Win'
                        #あっち向いてほいクラス
                        Look_check = 1
                        

                    elif choice == 'scissors':
                        result_text = 'Lose'
                        #あっち向いてほいクラス
                        Look_check = 2

                    elif choice == 'paper':
                        result_text = 'Draw'
                        #あいこの場合の処理（必要）
                        aiko = 1

                else:
                    result_text = 'Unknown'


            if result_text:
                if aiko == 0:
                    result_render = font.render(result_text, True, black)
                    result_rect = result_render.get_rect(center=window_rect.center)
                    screen.blit(result_render, result_rect)


                    pygame.display.flip()
                    pygame.time.delay(1000)
                    if Look_check > 0:
                        screen.fill(white)
                        result_render = font.render("", True, black)
                        result_rect = result_render.get_rect(center=window_rect.center)
                        screen.blit(result_render, result_rect)
                        #あっち向いてほいクラスのメソッド呼び出し
                        Look.atti()
                        pygame.time.delay(1000)

                        #あっち向いてほいの処理
                        if(Look_check == 1):
                            Look.hand_pose()
                            pygame.time.delay(1000)
                            screen.fill(white)
                            pygame.display.flip()

                        elif (Look_check == 2):
                            Look.face_pose()
                            pygame.time.delay(1000)
                            screen.fill(white)
                            pygame.display.flip()

                    
                    game_state = STATE_START
                    show_text = True
                    start_time = None
                    current_image = None
                    choice = None
                else:
                    result_render = font.render(result_text, True, black)
                    result_rect = result_render.get_rect(center=window_rect.center)
                    screen.blit(result_render, result_rect)
                    pygame.display.flip()
                    pygame.time.delay(3000)
                    game_state = STATE_PLAYING
                    show_text = True
                    start_time = None
                    current_image = None
                    choice = None



    pygame.display.flip()


pygame.quit()

