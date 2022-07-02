#!/usr/bin/env python
# -*- coding: utf-8 -*-
#上記2行は必須構文のため、コメント文だと思って削除しないこと
#Python2.7用プログラム



import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #TensorFlowのデバッグメッセージの表示設定。0：全てのメッセージが出力される（デフォルト）。1：INFOメッセージが出ない。2：INFOとWARNINGが出ない。3：INFOとWARNINGとERRORが出ない
import warnings
warnings.filterwarnings('ignore')
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.models import load_model  # ニューラルネットワーク学習ライブラリkerasの読み込み
#import tensorflow as tf
from tensorflow.python import keras
#import japanize_matplotlib
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
import re
from natsort import natsorted
import glob
import time
import rospy
from face_recognition_pkg.msg import face_recognition_message #メッセージファイルの読み込み（from パッケージ名.msg import 拡張子なしメッセージファイル名）
from face_recognition_pkg.srv import face_recognition_service #サービスファイルの読み込み（from パッケージ名.srv import 拡張子なしサービスファイル名）
import imutils  # 基本的な画像処理操作を簡単に行うためのライブラリimutilsの読み込み
import json
import sys
# cv2の読み込みエラー防止
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
#from openpose import pyopenpose as op
import pyopenpose as op #OpenPoseのインポート
opWrapper = op.WrapperPython()
import shutil #フォルダの削除に使用
import traceback #エラーメッセージの表示に使用


# TensorFlowでGPUを強制的に使用
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
json_output_path = "/home/limlab/catkin_ws/src/face_recognition_pkg/output"
openpose_params_json_file = "/home/limlab/catkin_ws/src/face_recognition_pkg/openpose_params.json"

shutil.rmtree(json_output_path) #json形式ファイルの保存されたフォルダの削除（初期化）
os.mkdir(json_output_path) #json形式ファイルを保存するフォルダの作成
file_input_path = '/home/limlab/ビデオ/src_hirano.mp4'  # 動画の読み込みパス
movie = cv2.VideoCapture(file_input_path)  # 動画の読み込み
file_output_path = '/home/limlab/catkin_ws/src/face_recognition_pkg/output.mp4'  #動画の書き込みパス



class openpose(): #OpenPoseのクラス
    def __init__(self):
        self.key_list = [] #keyリスト
        self.value_list = [] #valueリスト
        self.params = dict() #空の辞書型配列（キーとデータのセット）の作成（OpenPoseのオプションなどの情報が入る）



    def set_key_and_value(self): #jsonファイルからキーリストと値リストの取得
        with open(openpose_params_json_file, mode="r") as f: #OpenPoseの設定ファイル（json形式）を開いて、終了後に閉じる（with）
            json_object = json.load(f) #jsonファイルの読み込み
            for key, value in json_object.items(): #json_objectからkeyとvalueを取得し、その数だけ繰り返す
                if isinstance(key, unicode): #keyがユニコード型の場合
                    key = str(key) #文字列に変換
                if isinstance(value, unicode): #valueがユニコード型の場合
                    value = str(value) #文字列に変換
                self.key_list.append(key) #keyをkey_listに追加
                self.value_list.append(value) #valueをvalue_listに追加
            return self.key_list, self.value_list



    def set_params(self): #OpenPoseのパラメータ設定
        key_list, value_list = self.set_key_and_value() #jsonファイルからキーリストと値リストの取得

        for i in range(len(key_list)): #key_listの要素数だけ繰り返す
            self.params[key_list[i]] = value_list[i] #params辞書の設定

        return self.params



    def call_fr(self): #face_recognitionの呼び出し
        fr_instance = face_recognition() #face_recognitionのインスタンス作成
        fr_instance.main_loop() #メイン処理のループ



    def op_start(self): #OpenPoseの開始
        opWrapper.configure(self.set_params()) #OpenPoseのパラメータ設定
        opWrapper.start() #OpenPoseの開始
        print("################################################################################")
        print("\nOpenPoseの設定完了\nmodel_folder：{}\nwrite_json：{}\npart_candidates：{}\nface：{}\nnet_resolution：{}\n".format(self.params["model_folder"], self.params["write_json"], self.params["part_candidates"], self.params["face"], self.params["net_resolution"]))
        print("################################################################################")
        self.call_fr() #face_recognitionの呼び出し




class face_recognition():
    # 配列の宣言
    m_vector = []
    n_vector = []
    diff_vector = []
    r_shoulder_joint_angle = []
    mouth_leaning = []
    json_object_people_length = []
    json_object_point_length_list = []
    json_object_3point_length_list = []
    json_object_4point_length_list = []
    
    which = 0
    xmin, xmax = 0, 611
    ymin, ymax = 0, 360
    t_or_f = 0
    
    # 関節点の指定
    # 角度aに使用
    point1 = 0
    point2 = 15
    point3 = 16
    # 2つのベクトルの長さの差による顔の向き判定に使用
    point4 = 15
    point5 = 17
    point6 = 16
    point7 = 18

    all_point_list = [point1, point2, point3, point4, point5, point6, point7]
    point_list1 = [point1, point2, point3]
    point_list2 = [point4, point5, point6, point7]

    #OpenPoseのパラメータ
    params = dict()

    # データとイメージの読み込み用パラメータ
    detection_model_path = '/home/limlab/catkin_ws/src/face_recognition_pkg/haarcascade_files/haarcascade_frontalface_default.xml'
    emotion_model_path = '/home/limlab/catkin_ws/src/face_recognition_pkg/models/_mini_XCEPTION.102-0.66.hdf5'
    # bounding boxes（x,y,t(z)軸の立体画像を別の軸に沿って再切断（reslice）したときの画像の中心(0,0,0)からx,y,t(z)軸までの距離を囲んだ範囲）のためのハイパーパラメータ
    # カスケード型分類器（機械学習により抽出した特徴量をまとめた学習済みデータ、ここではxmlファイル）の読み込み（カスケード型分類器の特徴量を取得）
    face_cascade = cv2.CascadeClassifier(detection_model_path)
    emotion_classifier = load_model(
    emotion_model_path, compile=False)  # 学習済みモデルの読み込み
    EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
            "neutral"]  # 感情の配列

    fps = movie.get(cv2.CAP_PROP_FPS)  # 動画のFPS（フレームレート：フレーム毎秒）を取得
    W = 480  # 横サイズ
    H = 360  # 縦サイズ
    #W = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH)) #横サイズ
    #H = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT)) #縦サイズ
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 動画保存時のfourcc設定（mp4用）
    # 書き込む動画の仕様（ファイル名、fourcc, FPS, サイズ）
    video = cv2.VideoWriter(file_output_path, fourcc, int(fps), (W, H))
    font = cv2.FONT_HERSHEY_SIMPLEX  # 文字フォント
    # 背景全体が真っ黒の画像（キャンバス）の作成
    # (高さ（縦の画素数）×幅（横の画素数）)=(250×300)、カラーチャンネル(RGBの3つ)のすべての要素に符号なし整数として0を持つ（ゼロ埋め）3次元配列の作成
    canvas = np.zeros((250, 300, 3), dtype="uint8")



    def __init__(self):
        self.p4_x = 0
        self.p4_y = 0
        self.p5_x = 0
        self.p5_y = 0
        self.p6_x = 0
        self.p6_y = 0
        self.p7_x = 0
        self.p7_y = 0
        self.diff_x = 0
        self.diff_y = 0
        self.m = 0
        self.n = 0
        self.m_length = 0
        self.n_length = 0



    def get_json_object_length(self, json_object): #json_objectの要素数の取得
        self.json_object_people_length = len(
            json_object['people']) #人数（peopleの要素数）を取得
        #print("\n人数:{}".format(self.json_object_people_length))
        #print("関節点の合計：{}".format(len(all_point_list)))
        self.delete_json_object_point_length_list() #json_objectの要素数リストの削除(リセット)

        for i in range(len(self.all_point_list)): #全ての関節点の要素数を取得
            self.json_object_point_length_list.append(len(
            json_object['part_candidates'][0][str(self.all_point_list[i])])) 

        for i in range(0, 3): #3つ(i=0,1,2)の関節点の要素数(length)を取得
            #print(all_point_list[i])
            self.json_object_3point_length_list.append(self.json_object_point_length_list[i])

        for i in range(3, 7): #4つ(i=3,4,5,6)の関節点の要素数(length)を取得
            self.json_object_4point_length_list.append(self.json_object_point_length_list[i])



    def u_v_vector(self, pointa, pointb ,pointc, json_object): #u,vベクトルの計算
        #print("pointa:{}, pointb:{}, pointc:{}".format(pointa, pointb, pointc))
        pa_x = json_object['part_candidates'][0][str(pointa)][0]
        pa_y = json_object['part_candidates'][0][str(pointa)][1]

        pb_x = json_object['part_candidates'][0][str(pointb)][0]
        pb_y = json_object['part_candidates'][0][str(pointb)][1]

        pc_x = json_object['part_candidates'][0][str(pointc)][0]
        pc_y = json_object['part_candidates'][0][str(pointc)][1]
        u = np.array([pa_x - pb_x, pa_y - pb_y])  # uベクトル
        v = np.array([pc_x - pb_x, pc_y - pb_y])  # vベクトル
        return u, v



    def vector_angle(self, u, v): #2つのベクトルのなす角の計算
        i = np.inner(u, v)  # uベクトルとvベクトルの内積を計算
        n = LA.norm(u) * LA.norm(v)  # uベクトルの長さ(ノルム)とvベクトルの長さ（ノルム）の積を計算
        if n == 0:  # 2つのベクトルの長さの積が0の場合
            angle = 0  # 2つのベクトルのなす角を0°にする
        else:
            cos = i / n  # cosの値（２つのベクトルのなす角の公式より）
            # 2つのベクトルのなす角（−1〜1の範囲でarccosの値をラジアンから度に変換）
            angle = np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0)))
        return angle  # uベクトルとvベクトルのなす角を返す



    def delete_json_object_point_length_list(self): #json_objectの要素数リストの削除(リセット)
        if len(self.json_object_point_length_list) > 0: #要素が１つでもある場合
            del self.json_object_point_length_list[:] #リストの要素を全削除（[:]で全指定）
        
        if len(self.json_object_3point_length_list) > 0: #要素が１つでもある場合
            del self.json_object_3point_length_list[:] #リストの要素を全削除（[:]で全指定）

        if len(self.json_object_4point_length_list) > 0: #要素が１つでもある場合
            del self.json_object_4point_length_list[:] #リストの要素を全削除（[:]で全指定）



    def joint_angle(self, json_object, points_list, points_length_list): #2つのベクトルのなす角の取得
        if self.json_object_people_length == 0:  # 取得した人数が0人の場合
            angle = 0  # 2つのベクトルのなす角を0°にする
        elif points_length_list[0] == 0:  # 関節点1が取得できなかった場合
            angle = 0
        elif points_length_list[1] == 0:  # 関節点2が取得できなかった場合
            angle = 0
        elif points_length_list[2] == 0:  # 関節点3が取得できなかった場合
            angle = 0
        else:  # 関節点1,2,3が取得できた場合、json形式のファイルの中身から各関節点のx,y座標の値を取り出す
            u,v = self.u_v_vector(points_list[0], points_list[1], points_list[2], json_object) #u,vベクトルの計算
            angle = self.vector_angle(u, v) #2つのベクトルのなす角の計算
            #print("u:{}, v:{}".format(u, v))
        print("角度:{}".format(angle))

        return angle



    def face_vector_calculate(self, point_list2, json_object): #m,nベクトルの長さや差の取得
        if self.json_object_people_length == 0:  # 取得した人数が0人の場合
            self.diff = 0  # 2つのベクトルの長さの差を0にする
    
        # if (self.p11_x == 0) or (self.p12_x == 0):  # 関節点11 or 12が取得できなかった場合
        #     self.diff = 0
        #     print("(self.p11_x == 0) or (self.p12_x == 0)")
    
        else:  # json形式のファイルの中身から各関節点のx,y座標の値を取り出す
            #print("json_object_4point_length_list:{}".format(self.json_object_4point_length_list))
            if self.json_object_4point_length_list[0] > 0:  # 関節点4が取得できた場合
                self.p4_x = json_object['part_candidates'][0][str(point_list2[0])][0]
                self.p4_y = json_object['part_candidates'][0][str(point_list2[0])][1]
            elif self.json_object_4point_length_list[0] == 0:  # 関節点4が取得できなかった場合
                self.p4_x = 0  # ないものとする
                self.p4_y = 0  # ないものとする
            if self.json_object_4point_length_list[1] > 0:  # 関節点5が取得できた場合
                self.p5_x = json_object['part_candidates'][0][str(point_list2[1])][0]
                self.p5_y = json_object['part_candidates'][0][str(point_list2[1])][1]
            elif self.json_object_4point_length_list[1] == 0:  # 関節点5が取得できなかった場合
                self.p5_x = 0  # ないものとする
                self.p5_y = 0  # ないものとする
            if self.json_object_4point_length_list[2] > 0:  # 関節点6が取得できた場合
                self.p6_x = json_object['part_candidates'][0][str(point_list2[2])][0]
                self.p6_y = json_object['part_candidates'][0][str(point_list2[2])][1]
            elif self.json_object_4point_length_list[2] == 0:  # 関節点6が取得できなかった場合
                self.p6_x = 0  # ないものとする
                self.p6_y = 0  # ないものとする
            if self.json_object_4point_length_list[3] > 0:  # 関節点7が取得できた場合
                self.p7_x = json_object['part_candidates'][0][str(point_list2[3])][0]
                self.p7_y = json_object['part_candidates'][0][str(point_list2[3])][1]
            elif self.json_object_4point_length_list[3] == 0:  # 関節点7が取得できなかった場合
                self.p7_x = 0  # ないものとする
                self.p7_y = 0  # ないものとする
    
            if (self.p4_x * self.p5_x != 0) and (self.p4_y * self.p5_y != 0):  # 関節点4,5が取得できた場合
                # 関節点4と5までの距離を計算し、mベクトルとする
                self.m = np.array([self.p5_x - self.p4_x, self.p5_y - self.p4_y])
            elif (self.p4_x * self.p5_x == 0) and (self.p4_y * self.p5_y == 0):  # 関節点4,5が取得できなかった場合
                self.m = 0  # 関節点4と5までの距離を計算せず、ないものとする
            if (self.p6_x * self.p7_x != 0) and (self.p6_y * self.p7_y != 0):  # 関節点6,7が取得できた場合
                # 関節点6と7までの距離を計算し、nベクトルとする
                self.n = np.array([self.p7_x - self.p6_x, self.p7_y - self.p6_y])
            elif (self.p6_x * self.p7_x == 0) and (self.p6_y * self.p7_y == 0):  # 関節点6,7が取得できなかった場合
                self.n = 0  # 関節点6と7までの距離を計算せず、ないものとする
    
            self.m_length = LA.norm(self.m)  # mベクトルの長さを計算
            self.n_length = LA.norm(self.n)  # ｎベクトルの長さを計算
    
        self.diff = self.m_length - self.n_length  # mベクトルの長さ(ノルム)とnベクトルの長さ（ノルム）の差を計算
        #print("m:{}, n:{}".format(self.m_length, self.n_length))
    
        return self.m_length, self.n_length, self.diff, self.p6_x, self.p7_x  # 複数の返り値



    def point_x(self, number): # 各関節点のx座標の値がjson形式のファイルの中身の何番目に当たるか
        return number * 3  # （"関節点1のx座標",関節点1のy座標,関節点1の信頼度,"関節点2のx座標",関節点2のy座標,関節点2の信頼度,‥）というように3つ飛ばし



    def point_y(self, number):  # 各関節点のy座標の値がjson形式のファイルの中身の何番目に当たるか
        return (number * 3) + 1 # （関節点1のx座標,"関節点1のy座標",関節点1の信頼度,関節点2のx座標,"関節点2のy座標",関節点2の信頼度,‥）というようにx座標に対して1ずれたところから3つ飛ばし



    def face_detection(self, gray_image):
        # アスペクト比を維持して切り取ったフレームのサイズ変更（幅widthをWに）
        #frame = imutils.resize(frame, width=W)
        #frame_openpose = imutils.resize(frame_openpose, width=W)
        #frameClone = frame.copy()  # グレースケール変換前のフレームのコピー（取得した画像に変更を加えると、元画像も変更されてしまうため）
    
        # グレースケール変換した画像での顔認識の実行
        # detectMultiScale()：顔を検出する関数。戻り値：顔の（左上の点のx座標,左上の点のy座標,左上の点からの横幅,左上の点からの高さ）のリスト（配列）。要素数＝顔の検出数
        # gray_image:CV_8U型の行列。ここに格納されている画像中から物体が検出される
        # ScaleFactor:入力画像のスケールが探索ごとに縮小される割合。ここでは1.1=10%ずつ縮小しながら検出用モデルのサイズと一致するか探索。大きいほど誤検出が多く、小さいほど未検出となる
        # minNeighbors:物体候補となる矩形に含まれる近傍矩形（きんぼうくけい）の最小数。検出器が検出する箇所がこの数だけ重複することになるため、重複が多い部分ほど信頼性が高い。そのしきい値＝信頼性パラメータ
        # minSize:物体が取りうる最小サイズ。これより小さい物体は無視される
        faces = self.face_cascade.detectMultiScale(
                gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE) #リスト（配列）が返される
        return faces



    def roi_process(self, gray_image, fX, fY, fW, fH):
        # グレースケールイメージから顔のROI（Region Of Interest：関心領域、対象領域、注目領域）を矩形になるように抽出（画像の一部分のトリミング）[縦の範囲,横の範囲]
        roi = gray_image[fY:fY + fH, fX:fX + fW] #画像処理ライブラリPIL(Pillow)形式のROI
        roi = cv2.resize(roi, (64, 64))  # ROIのリサイズ
        # ROIに対する画像処理
        # 浮動小数点に変換し、画像のピクセル値を255で割って0〜1の数値に正規化
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)  # numpyの3次元配列に変換。kerasで扱えるようになる
        # 配列の0番目の軸（一番外側の次元）に大きさ1の次元を新たに追加（[]:1次元(大きさ1),[[]]:2次元,[[[]]]：3次元）。1枚の画像を「1枚の画像を含む配列」に変換。kerasの判定処理を行う機能が「複数の画像を含む配列」という形式を要求するため、次元数のフォーマットを合わせる。3次元→4次元配列になる
        roi = np.expand_dims(roi, axis=0)
        #print("ROI：",roi.ndim,"次元配列\n")]
        return roi



    def angle_judge(self, a): #2つのベクトルのなす角による顔の向きの判定
        # if ((self.p6_x != 0) or (self.p7_x != 0)) and ((self.p13_x != 0) or (self.p14_x != 0)):  # 顔全体が画面内にある場合
        #     self.t_or_f = 1
        #     print("画面内に顔あり")
        # else:  # 顔が一部分でも画面外にはみ出している場合
        #     self.t_or_f = 0
        #     print("画面外に顔あり")
    
        if (self.diff <= -50):  # 右向き
            self.which = 0
        elif (self.diff > -50) and (self.diff < 50):  # 正面
            self.which = 1
        elif (self.diff >= 50):  # 左向き
            self.which = 2

        if a == 0:  # 検出できなかった場合
            self.message1 = 'No Detect'
            self.color1 = (255, 255, 255)
    
        elif a != 0:  # 検出できた場合
            # if (a > 0) and (a <= 60) and (self.which == 0):  # 右向き
            #     self.message1 = 'Right'
            #     self.color1 = (255, 255, 0)
            # elif (a > 60) and (a <= 90) and (self.which == 1):  # 正面
            #     self.message1 = 'Front'
            #     self.scolor1 = (0, 255, 255)
            # elif (a > 90) and (self.which == 2):  # 左向き
            #     self.message1 = 'Left'
            #     self.color1 = (0, 0, 255)
            if self.which == 0:  # 右向き
                self.message1 = 'Right'
                self.color1 = (255, 255, 0)
            elif self.which == 1:  # 正面
                self.message1 = 'Front'
                self.color1 = (0, 255, 255)
            elif self.which == 2:  # 左向き
                self.message1 = 'Left'
                self.color1 = (0, 0, 255)
        #print(self.message1, self.color1)
        return self.message1, self.color1



    def face_detection_setting(self): #顔認識の設定
        #OpenPoseによる画像処理
        datum = op.Datum() #データ受け渡し用オブジェクト（データム）の作成
        frag, frame = movie.read()  # 動画から1フレーム読み込む。fragはret(return valueの略、戻り値の意味）の表記もあるが、同じもの。ブール値（TrueかFalse）の情報が入る。今回は使っていない。
        # frame = frame[ymin:ymax, xmin:xmax] #フレームの右部分のトリミング
        datum.cvInputData = frame #frameをdatum.cvInputDataに格納
        opWrapper.emplaceAndPop([datum]) #リスト型で渡したデータム内に解析結果（出力画像、関節位置等々）が含まれている
        frame_openpose = datum.cvOutputData #OpenPoseの骨格を反映したフレーム
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # グレースケール変換（顔検出に使用するため）
        return frame_openpose, gray_image



    def cv2_show_setting(self, faces, canvas, frame_openpose, gray_image, a): #ビデオ描画の設定
        copy_canvas = canvas.copy()  #フレームのコピー（取得した画像に変更を加えると、以前の画像も変更されてしまうため）
        copy_frame_openpose = frame_openpose.copy()  #フレームのコピー（取得した画像に変更を加えると、以前の画像も変更されてしまうため）

        self.message1, self.color1 = self.angle_judge(a) #2つのベクトルのなす角による顔の向きの判定

        if len(faces) > 0: #顔が検出できた場合
            # 要素の並び替え
            # reverse=True:降順ソート（ここでは、検出した順番と逆）
            # lambda（ラムダ）:無名関数。facesの要素xを受け取り、x[n]（n番目の要素）を返す
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces #faces(リスト型)の展開
            roi = self.roi_process(gray_image, fX, fY, fW, fH) #ROI
            preds = self.emotion_classifier.predict(roi)[0]  # kerasで学習済みモデルによる予測。numpyの複数の画像に対する判定結果の配列が返ってくる
            #emotion_probability = np.max(preds)  # 最も信頼度の高い感情を取得
            label = self.EMOTIONS[preds.argmax()]  # 最も信頼度の高い感情をラベルに
    
            # バーの表示設定
            for (i, (emotion, prob)) in enumerate(zip(self.EMOTIONS, preds)):  # （感情,信頼度）の順にi（＝感情の数、ループ変数）だけ繰り返す
            # ラベルテキストの設定({}の部分をformat(中身)で置換。.2：小数点以下2桁、f：浮動小数点)
                canvas_text = "{}: {:.2f}%".format(emotion, prob * 100)
                bar_width = int(prob * 300)  # バーの幅の設定
            # cv2.rectangle(描画対象,左上の点の座標,右下の点の座標,矩形の色,線の太さ（-1で塗りつぶし）)
                cv2.rectangle(copy_canvas, (7, (i * 35) + 5),
                        (bar_width, (i * 35) + 35), (0, 255, 0), -1)  # バー（矩形）の描画
                cv2.putText(copy_canvas, canvas_text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)  #テキストの描画

            cv2.putText(copy_frame_openpose, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # ラベル（表情）の追加
            cv2.rectangle(copy_frame_openpose, (fX, fY), (fX + fW, fY + fH),
                    (0, 0, 255), 2)  #顔検出の矩形の描画
            cv2.putText(copy_frame_openpose, self.message1, (10, 20), self.font, 0.8, self.color1, 2, 4) #動画に顔の向きを追加
            return copy_frame_openpose, copy_canvas
        
        else: #顔が検出できなかった場合
            # self.message1 = 'No Detect'
            # self.color1 = (255, 255, 255)
            # cv2.putText(copy_frame_openpose, self.message1, (10, 20), self.font, 0.8, self.color1, 2, 4) #動画に顔の向きを追加
            # #self.message2 = 'No Detect'
            # #self.color2 = (255, 255, 255)
            return copy_frame_openpose, copy_canvas



    def show_write_video(self,processed_frame, canvas, openpose_version): #ビデオの表示と書き込み
        cv2.imshow('OpenPose:{} (Exit with Q)'.format(openpose_version), processed_frame)  # 顔の向き情報を追加した動画の表示
        #cv2.imshow("video",frame_openpose)
        self.video.write(processed_frame)  # 1フレームずつ書き込み＝動画の作成
        # canvas = cv2.resize(canvas, (250,300)) #動画サイズの縮小
        cv2.imshow("Emotion Probabilities", canvas)



    def json_file_process(self): #jsonファイルの処理
        # OpenPoseにより書き込まれた「〜_keypoints.json」のファイルをすべて読み込む(for)
        for json_file in natsorted(glob.glob('/home/limlab/openpose/output_face_hirano/output/*_keypoints.json')):
            print(os.path.basename(json_file)) #今現在のjson形式ファイルを表示
            with open(json_file) as f:  # json形式のファイルをfとして開き、終了時に自動的に閉じる（with）
                json_object = json.load(f)  # json形式ファイルを読み込む
                #self.json_object_people_length, self.json_object_point_length_list = self.get_json_object(json_object)
                openpose_version = json_object['version']
                # 関数の返り値をいっぺんに取得
                self.get_json_object_length(json_object) #json_objectの要素数の取得

                a = self.joint_angle(json_object, self.point_list1, self.json_object_3point_length_list)  #2つのベクトルのなす角の取得
                # 関数の返り値をいっぺんに取得
                #print("a：{}\na：{}".format(a, a))
                vector_info = self.face_vector_calculate(self.point_list2, json_object) #m,nベクトルの長さや差の取得
                #print(vector_info)
                # m_vector.append(vector_info[0]) #配列にmベクトルの長さ(m_length)の追加
                # n_vector.append(vector_info[1]) #配列にmベクトルの長さ(n_length)の追加
                # diff_vector.append(vector_info[2]) #配列に2つのベクトルの差(diff)の追加
                # m_length = vector_info[0] #mベクトルの長さ(m_length)の取得
                # n_length = vector_info[1] #nベクトルの長さ(n_length)の取得
                self.diff = vector_info[2]  # 2つのベクトルの長さの差(diff)を取得
                self.p6_x = vector_info[3]
                self.p7_x = vector_info[4]
        return json_file, a, openpose_version



    def finish(self): # 終了処理
        print("\n終了")
        movie.release()  # 読み込み動画を閉じる
        self.video.release()  # 書き込み動画を閉じる
        cv2.destroyAllWindows()  # すべてのウィンドウを閉じる



    def main_loop(self): #メイン処理のループ
        while True:
            frame_openpose, gray_image = self.face_detection_setting() #顔認識の設定
            faces = self.face_detection(gray_image) #顔認識の実行
            json_file, a, openpose_version = self.json_file_process() #jsonファイルの処理
            copy_frame_openpose, copy_canvas = self.cv2_show_setting(faces, self.canvas, frame_openpose, gray_image, a) #ビデオ描画の設定

            #cv2.imshow("OpenPose 1.6.0", datum.cvOutputData) #認識した骨格を反映した画像の表示
            #frame = cv2.resize(datum.cvOutputData, (W,H)) #動画サイズの縮小。入力動画と出力動画のサイズが合っていないと書き込んだ動画が再生できない。datum.cvOutputDataにしないとOpenPoseの認識した骨格が表示されない
            
            processed_frame = cv2.resize(copy_frame_openpose, (self.W, self.H)) #動画サイズの縮小。入力動画と出力動画のサイズが合っていないと書き込んだ動画が再生できない。datum.cvOutputDataにしないとOpenPoseの認識した骨格が表示されない
            
            self.show_write_video(processed_frame, copy_canvas, openpose_version) #ビデオの表示と書き込み
            #print(os.path.basename(json_file)) #今現在のjson形式ファイルを表示

            if cv2.waitKey(1) & 0xFF == ord('q'):  # qキーで終了
                self.finish() #終了処理



class Client(): #クライアントのクラス
    def __init__(self): #コンストラクタと呼ばれる初期化のための関数（メソッド）
        self.count = 0 
        #service_messageの型を作成
        self.service_message = face_recognition_service() 
        self.rate = rospy.Rate(1) #1秒間に1回データを受信する



    def call_OpenPose(self): #OpenPoseの呼び出し
        op_instance = openpose() #openposeのインスタンス作成
        op_instance.op_start() #openposeの開始



    def service_request(self): #サービスのリクエスト
        rospy.wait_for_service('openpose_service') #サービスが使えるようになるまで待機
        try:
            self.client = rospy.ServiceProxy('openpose_service', face_recognition_service) #クライアント側で使用するサービスの定義
            self.service_message.openpose_request = "サービスをリクエスト"
            response = self.client(self.service_message.openpose_request) #「戻り値 = self.client(引数)」。クライアントがsrvファイルで定義した引数（srvファイル内「---」の上側）を別ファイルのサーバーにリクエストし、サーバーからの返り値（srvファイル内「---」の下側）をresponseに代入
            rospy.loginfo("サービスのリクエストに成功：{}".format(response.openpose_response))
            self.call_OpenPose() #OpenPoseの呼び出し
        except rospy.ServiceException:
            rospy.loginfo("サービスのリクエストに失敗")



def main(): #メイン関数
    #初期化し、ノードの名前を設定
    rospy.init_node('fr_client', anonymous=True)
    #クラスのインスタンス作成（クラス内の関数や変数を使えるようにする）
    sub = Client()
    sub.service_request() #サービスのリクエスト
    rospy.spin() #callback関数を繰り返し呼び出す（終了防止）




if __name__ == '__main__':
    try:
        main() #メイン関数の実行
    except rospy.ROSInterruptException:
        print("\n{}".format(traceback.format_exc())) #エラー内容を表示
