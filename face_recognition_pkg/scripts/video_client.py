#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 上記2行は必須構文のため、コメント文だと思って削除しないこと
# Python2.7用プログラム

import traceback  # エラーメッセージの表示に使用
import shutil  # フォルダの削除に使用
import pyopenpose as op  # OpenPoseのインポート
# サービスファイルの読み込み（from パッケージ名.srv import 拡張子なしサービスファイル名）
from face_recognition_pkg.srv import calculate_service
# サービスファイルの読み込み（from パッケージ名.srv import 拡張子なしサービスファイル名）
from face_recognition_pkg.srv import face_recognition_service
# メッセージファイルの読み込み（from パッケージ名.msg import 拡張子なしメッセージファイル名）
service_message = face_recognition_service()
from face_recognition_pkg.msg import face_recognition_message
import cv2
import sys
import json
import imutils  # 基本的な画像処理操作を簡単に行うためのライブラリimutilsの読み込み
import rospy
import time
import glob
from natsort import natsorted
import re
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from tensorflow.python import keras
# ニューラルネットワーク学習ライブラリkerasの読み込み
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import img_to_array
import warnings
from logging import shutdown
import os
# TensorFlowのデバッグメッセージの表示設定。0：全てのメッセージが出力される（デフォルト）。1：INFOメッセージが出ない。2：INFOとWARNINGが出ない。3：INFOとWARNINGとERRORが出ない
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
# cv2の読み込みエラー防止
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
opWrapper = op.WrapperPython()

# TensorFlowでGPUを強制的に使用
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
json_output_path = "/home/limlab/catkin_ws/src/face_recognition_pkg/output"
openpose_params_json_file = "/home/limlab/catkin_ws/src/face_recognition_pkg/openpose_params.json"

if os.path.exists(json_output_path): #既にjson_file_pathが存在する場合
    shutil.rmtree(json_output_path)  # json形式ファイルの保存されたフォルダの削除（初期化）
os.mkdir(json_output_path)  # json形式ファイルを保存するフォルダの作成
file_input_path = '/home/limlab/ビデオ/src_hirano.mp4'  # 動画の読み込みパス
movie = cv2.VideoCapture(file_input_path)  # 動画の読み込み
file_output_path = '/home/limlab/catkin_ws/src/face_recognition_pkg/output.mp4'  # 動画の書き込みパス



class openpose():  # OpenPoseのクラス
    def __init__(self):
        self.key_list = []  # keyリスト
        self.value_list = []  # valueリスト
        self.params = dict()  # 空の辞書型配列（キーとデータのセット）の作成（OpenPoseのオプションなどの情報が入る）



    def set_key_and_value(self):  # jsonファイルからキーリストと値リストの取得
        # OpenPoseの設定ファイル（json形式）を開いて、終了後に閉じる（with）
        with open(openpose_params_json_file, mode="r") as f:
            json_object = json.load(f)  # jsonファイルの読み込み
            for key, value in json_object.items():  # json_objectからkeyとvalueを取得し、その数だけ繰り返す
                if isinstance(key, unicode):  # keyがユニコード型の場合
                    key = str(key)  # 文字列に変換
                if isinstance(value, unicode):  # valueがユニコード型の場合
                    value = str(value)  # 文字列に変換
                self.key_list.append(key)  # keyをkey_listに追加
                self.value_list.append(value)  # valueをvalue_listに追加
            return self.key_list, self.value_list



    def set_params(self):  # OpenPoseのパラメータ設定
        key_list, value_list = self.set_key_and_value()  # jsonファイルからキーリストと値リストの取得

        for i in range(len(key_list)):  # key_listの要素数だけ繰り返す
            self.params[key_list[i]] = value_list[i]  # params辞書の設定

        return self.params



    def op_start(self):  # OpenPoseの開始
        opWrapper.configure(self.set_params())  # OpenPoseのパラメータ設定
        opWrapper.start()  # OpenPoseの開始
        print("################################################################################")
        print("\nOpenPoseの設定完了\nmodel_folder：{}\nwrite_json：{}\npart_candidates：{}\nface：{}\nnet_resolution：{}\n".format(
            self.params["model_folder"], self.params["write_json"], self.params["part_candidates"], self.params["face"], self.params["net_resolution"]))
        print("################################################################################")
        # self.call_fr() #face_recognitionの呼び出し



class face_recognition():
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
    # W = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH)) #横サイズ
    # H = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT)) #縦サイズ
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
        # messageの型を作成
        self.message = face_recognition_message()
        self.cli = cal_Client()
        self.red_color = (0, 0, 255)
        self.sky_blue_color = (255, 255, 0)
        self.yellow_color = (0, 255, 255)



    def face_detection(self, gray_image):
        # アスペクト比を維持して切り取ったフレームのサイズ変更（幅widthをWに）
        #frame = imutils.resize(frame, width=W)
        #frame_openpose = imutils.resize(frame_openpose, width=W)
        # frameClone = frame.copy()  # グレースケール変換前のフレームのコピー（取得した画像に変更を加えると、元画像も変更されてしまうため）

        # グレースケール変換した画像での顔認識の実行
        # detectMultiScale()：顔を検出する関数。戻り値：顔の（左上の点のx座標,左上の点のy座標,左上の点からの横幅,左上の点からの高さ）のリスト（配列）。要素数＝顔の検出数
        # gray_image:CV_8U型の行列。ここに格納されている画像中から物体が検出される
        # ScaleFactor:入力画像のスケールが探索ごとに縮小される割合。ここでは1.1=10%ずつ縮小しながら検出用モデルのサイズと一致するか探索。大きいほど誤検出が多く、小さいほど未検出となる
        # minNeighbors:物体候補となる矩形に含まれる近傍矩形（きんぼうくけい）の最小数。検出器が検出する箇所がこの数だけ重複することになるため、重複が多い部分ほど信頼性が高い。そのしきい値＝信頼性パラメータ
        # minSize:物体が取りうる最小サイズ。これより小さい物体は無視される
        faces = self.face_cascade.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)  # リスト（配列）が返される
        return faces



    def roi_process(self, gray_image, fX, fY, fW, fH):
        # グレースケールイメージから顔のROI（Region Of Interest：関心領域、対象領域、注目領域）を矩形になるように抽出（画像の一部分のトリミング）[縦の範囲,横の範囲]
        roi = gray_image[fY:fY + fH, fX:fX + fW]  # 画像処理ライブラリPIL(Pillow)形式のROI
        roi = cv2.resize(roi, (64, 64))  # ROIのリサイズ
        # ROIに対する画像処理
        # 浮動小数点に変換し、画像のピクセル値を255で割って0〜1の数値に正規化
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)  # numpyの3次元配列に変換。kerasで扱えるようになる
        # 配列の0番目の軸（一番外側の次元）に大きさ1の次元を新たに追加（[]:1次元(大きさ1),[[]]:2次元,[[[]]]：3次元）。1枚の画像を「1枚の画像を含む配列」に変換。kerasの判定処理を行う機能が「複数の画像を含む配列」という形式を要求するため、次元数のフォーマットを合わせる。3次元→4次元配列になる
        roi = np.expand_dims(roi, axis=0)
        # print("ROI：",roi.ndim,"次元配列\n")]
        return roi



    def face_detection_setting(self):  # 顔認識の設定
        calculate_message = calculate_service()
        # OpenPoseによる画像処理
        datum = op.Datum()  # データ受け渡し用オブジェクト（データム）の作成
        # 動画から1フレーム読み込む。fragはret(return valueの略、戻り値の意味）の表記もあるが、同じもの。ブール値（TrueかFalse）の情報が入る。今回は使っていない。
        frag, frame = movie.read()
        # frame = frame[ymin:ymax, xmin:xmax] #フレームの右部分のトリミング
        datum.cvInputData = frame  # frameをdatum.cvInputDataに格納
        # リスト型で渡したデータム内に解析結果（出力画像、関節位置等々）が含まれている
        opWrapper.emplaceAndPop([datum])
        frame_openpose = datum.cvOutputData  # OpenPoseの骨格を反映したフレーム
        # グレースケール変換（顔検出に使用するため）
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        message1 = self.call_calculate_service(calculate_message)
        return frame_openpose, gray_image, message1



    def cv2_show_setting(self, faces, canvas, frame_openpose, gray_image, message1):  # ビデオ描画の設定
        
        copy_canvas = canvas.copy()  # フレームのコピー（取得した画像に変更を加えると、以前の画像も変更されてしまうため）
        # フレームのコピー（取得した画像に変更を加えると、以前の画像も変更されてしまうため）
        copy_frame_openpose = frame_openpose.copy()

        if len(faces) > 0:  # 顔が検出できた場合
            # 要素の並び替え
            # reverse=True:降順ソート（ここでは、検出した順番と逆）
            # lambda（ラムダ）:無名関数。facesの要素xを受け取り、x[n]（n番目の要素）を返す
            faces = sorted(faces, reverse=True, key=lambda x: (
                x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces  # faces(リスト型)の展開
            roi = self.roi_process(gray_image, fX, fY, fW, fH)  # ROI
            # kerasで学習済みモデルによる予測。numpyの複数の画像に対する判定結果の配列が返ってくる
            preds = self.emotion_classifier.predict(roi)[0]
            # emotion_probability = np.max(preds)  # 最も信頼度の高い感情を取得
            label = self.EMOTIONS[preds.argmax()]  # 最も信頼度の高い感情をラベルに

            # バーの表示設定
            # （感情,信頼度）の順にi（＝感情の数、ループ変数）だけ繰り返す
            for (i, (emotion, prob)) in enumerate(zip(self.EMOTIONS, preds)):
                # ラベルテキストの設定({}の部分をformat(中身)で置換。.2：小数点以下2桁、f：浮動小数点)
                canvas_text = "{}: {:.2f}%".format(emotion, prob * 100)
                bar_width = int(prob * 300)  # バーの幅の設定
            # cv2.rectangle(描画対象,左上の点の座標,右下の点の座標,矩形の色,線の太さ（-1で塗りつぶし）)
                cv2.rectangle(copy_canvas, (7, (i * 35) + 5),
                              (bar_width, (i * 35) + 35), (0, 255, 0), -1)  # バー（矩形）の描画
                cv2.putText(copy_canvas, canvas_text, (10, (i * 35) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            self.red_color, 2)  # テキストの描画

            cv2.putText(copy_frame_openpose, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.red_color, 2)  # ラベル（表情）の追加
            cv2.rectangle(copy_frame_openpose, (fX, fY), (fX + fW, fY + fH), self.red_color, 2)  # 顔検出の矩形の描画

            cv2.putText(copy_frame_openpose, message1, (10, 20),
                        self.font, 0.8, self.yellow_color, 2, 4)  # 動画に顔の向きを追加
            return copy_frame_openpose, copy_canvas

        else:  # 顔が検出できなかった場合
            cv2.putText(copy_frame_openpose, message1, (10, 20), self.font, 0.8, self.yellow_color, 2, 4) #動画に顔の向きを追加
            return copy_frame_openpose, copy_canvas



    def show_write_video(self, processed_frame, canvas, openpose_version):  # ビデオの表示と書き込み
        cv2.imshow('OpenPose:{} (Exit with Q)'.format(
            openpose_version), processed_frame)  # 顔の向き情報を追加した動画の表示
        # cv2.imshow("video",frame_openpose)
        self.video.write(processed_frame)  # 1フレームずつ書き込み＝動画の作成
        # canvas = cv2.resize(canvas, (250,300)) #動画サイズの縮小
        cv2.imshow("Emotion Probabilities", canvas)
        cv2.imwrite('/home/limlab/catkin_ws/src/face_recognition_pkg/output.png', canvas)



    def finish(self):  # 終了処理
        print("\n終了")
        movie.release()  # 読み込み動画を閉じる
        self.video.release()  # 書き込み動画を閉じる
        cv2.destroyAllWindows()  # すべてのウィンドウを閉じる
        rospy.signal_shutdown("User hit q key to quit.")
        #sys.exit()



    def call_calculate_service(self, calculate_message):
        message1 = self.cli.calculate_service_request(calculate_message)
        return message1



    def main_loop(self):  # メイン処理のループ
        while not rospy.is_shutdown():
            frame_openpose, gray_image, message1 = self.face_detection_setting()  # 顔認識の設定
            faces = self.face_detection(gray_image)  # 顔認識の実行
            copy_frame_openpose, copy_canvas = self.cv2_show_setting(
                faces, self.canvas, frame_openpose, gray_image, message1)  # ビデオ描画の設定
            # cv2.imshow("OpenPose 1.6.0", datum.cvOutputData) #認識した骨格を反映した画像の表示
            # frame = cv2.resize(datum.cvOutputData, (W,H)) #動画サイズの縮小。入力動画と出力動画のサイズが合っていないと書き込んだ動画が再生できない。datum.cvOutputDataにしないとOpenPoseの認識した骨格が表示されない

            # 動画サイズの縮小。入力動画と出力動画のサイズが合っていないと書き込んだ動画が再生できない。datum.cvOutputDataにしないとOpenPoseの認識した骨格が表示されない
            processed_frame = cv2.resize(copy_frame_openpose, (self.W, self.H))
            self.show_write_video(
                processed_frame, copy_canvas, self.message.openpose_version)  # ビデオの表示と書き込み
            # print(os.path.basename(json_file)) #今現在のjson形式ファイルを表示

            if cv2.waitKey(1) & 0xFF == ord('q'):  # qキーで終了
                self.finish()  # 終了



class op_Client():  # クライアントのクラス
    def __init__(self):  # コンストラクタと呼ばれる初期化のための関数（メソッド）
        self.rate = rospy.Rate(1)  # 1秒間に1回データを受信する
        self.op = openpose()

    def openpose_service_request(self):  # サービスのリクエスト
        rospy.wait_for_service('openpose_service')  # サービスが使えるようになるまで待機
        try:
            self.client = rospy.ServiceProxy(
                'openpose_service', face_recognition_service)  # クライアント側で使用するサービスの定義
            service_message.openpose_request = "サービスをリクエスト"
            # 「戻り値 = self.client(引数)」。クライアントがsrvファイルで定義した引数（srvファイル内「---」の上側）を別ファイルのサーバーにリクエストし、サーバーからの返り値（srvファイル内「---」の下側）をresponseに代入
            response = self.client(service_message.openpose_request)
            rospy.loginfo("サービスのリクエストに成功：{}".format(
                response.openpose_response))

        except rospy.ServiceException:
            rospy.loginfo("サービスのリクエストに失敗")



class cal_Client():  # クライアントのクラス
    def __init__(self):  # コンストラクタと呼ばれる初期化のための関数（メソッド）
        self.rate = rospy.Rate(5)  # 1秒間に1回データを受信する



    def calculate_service_request(self, calculate_message):  # サービスのリクエスト
        rospy.wait_for_service('calculate_service')  # サービスが使えるようになるまで待機
        try:
            self.client = rospy.ServiceProxy(
                'calculate_service', calculate_service)  # クライアント側で使用するサービスの定義
            calculate_message.calculate_request = "計算をリクエスト"
            # 「戻り値 = self.client(引数)」。クライアントがsrvファイルで定義した引数（srvファイル内「---」の上側）を別ファイルのサーバーにリクエストし、サーバーからの返り値（srvファイル内「---」の下側）をresponseに代入
            response = self.client(calculate_message.calculate_request)
            rospy.loginfo("計算のリクエストに成功：{}".format(response))
            return response.message1

        except rospy.ServiceException:
            rospy.loginfo("計算のリクエストに失敗")



def main():  # メイン関数
    # 初期化し、ノードの名前を設定
    rospy.init_node('fr_client2', anonymous=True)
    # クラスのインスタンス作成（クラス内の関数や変数を使えるようにする）
    op_cli = op_Client()
    op_cli.openpose_service_request()  # サービスのリクエスト
    op = openpose()
    op.op_start()
    fr_instance = face_recognition()
    fr_instance.main_loop()
    # rospy.spin() #callback関数を繰り返し呼び出す（終了防止）



if __name__ == '__main__':
    try:
        main()  # メイン関数の実行
    except rospy.ROSInterruptException:
        movie.release()  # 読み込み動画を閉じる
        face_recognition.video.release()  # 書き込み動画を閉じる
        cv2.destroyAllWindows()  # すべてのウィンドウを閉じる
        print("\n{}".format(traceback.format_exc()))  # エラー内容を表示
        sys.exit()