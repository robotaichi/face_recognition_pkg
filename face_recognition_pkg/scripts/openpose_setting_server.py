#!/usr/bin/env python
# -*- coding: utf-8 -*-
#上記2行は必須構文のため、コメント文だと思って削除しないこと
#Python2.7用プログラム



from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.models import load_model  # ニューラルネットワーク学習ライブラリkerasの読み込み
from tensorflow.python import keras
# cv2の読み込みエラー防止
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
# import pyopenpose as op #OpenPoseのインポート
# opWrapper = op.WrapperPython()
import rospy
from face_recognition_pkg.msg import face_recognition_message #メッセージファイルの読み込み（from パッケージ名.msg import 拡張子なしメッセージファイル名）
from face_recognition_pkg.srv import face_recognition_service #サービスファイルの読み込み（from パッケージ名.srv import 拡張子なしサービスファイル名）
import traceback #エラーメッセージの表示に使用
import json

# TensorFlowでGPUを強制的に使用
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

json_output_path = "/home/limlab/catkin_ws/src/face_recognition_pkg/output"
openpose_params_json_file = "/home/limlab/catkin_ws/src/face_recognition_pkg/openpose_params.json"



class openpose():
    def __init__(self):
        self.params = dict() #空の辞書型配列（キーとデータのセット）の作成（OpenPoseのオプションなどの情報が入る）


    def openpose_setting(self): #OpenPoseの設定
        #OpenPose開始時のオプション指定（その他のパラメータについては、include /openpose /flags.hppを参照）
        self.params["model_folder"] = "/home/limlab/catkin_ws/src/face_recognition_pkg/models/" #キー：model_folder、データ：../models/
        self.params["write_json"] = json_output_path #キー：write_json、データ：json_output_path。座標情報をjson形式ファイルとして書き込み
        self.params["part_candidates"] = True #キー：part_candidates、データ：True
        self.params["face"] = True #キー：face、データ：True。顔検出を有効
        #params["hand"] = True #キー：hand、データ：True。手検出を有効
        self.params["net_resolution"] = "288x144" #キー：net_resolution、データ：320x176。256x112。208x64。192x48。認識精度は落ちるが、処理速度は向上。16の倍数で数字を指定

        self.write_json() #jsonファイルに書き込み



    def write_json(self): #jsonファイルに書き込み
        with open(openpose_params_json_file, mode="w") as f:
            json_object = json.dumps(self.params, f)
            f.write(json_object) #jsonファイルに書き込み



    def make_Text(self): #確認メッセージの作成
        self.openpose_setting() #OpenPoseの設定
        Text = "\nOpenPoseの設定完了\nmodel_folder：{}\nwrite_json：{}\npart_candidates：{}\nface：{}\nnet_resolution：{}\n".format(self.params["model_folder"], self.params["write_json"], self.params["part_candidates"], self.params["face"], self.params["net_resolution"])
        return Text



# class Publishsers(): #パブリッシャーのクラス
#     def __init__(self): #コンストラクタと呼ばれる初期化のための関数（メソッド）
#         #messageの型を作成
#         self.message = face_recognition_message()
#         # face_recognition_message型のメッセージを"openpose_ropic"というトピックに送信する
#         self.publisher  = rospy.Publisher('openpose_topic', face_recognition_message, queue_size=10)
#         self.rate = rospy.Rate(100) #1秒間に2回データを送信する



#     def execute_openpose(self): #OpenPoseの実行
#         op_instance = openpose() #OpenPoseのインスタンス作成
#         Text = op_instance.make_Text() #確認メッセージの作成
#         params = op_instance.params
#         return Text, params



#     def make_msg(self): #送信するメッセージの作成
#         Text, params = self.execute_openpose() #OpenPoseの実行
#         #メッセージの変数を設定
#         self.message.openpose_message = Text



#     def send_msg(self): #メッセージを送信
#         self.make_msg() #送信するメッセージの作成
#         self.publisher.publish(self.message) #作成したメッセージの送信
#         rospy.loginfo("\nOpenPoseの設定を書き込みました\n")
#         self.rate.sleep()



class Server(): #サーバーのクラス
    def __init__(self):
        #service_messageの型を作成
        self.service_message = face_recognition_service()
        #self.pub = Publishsers() #パブリッシャークラスのインスタンス化(実体化)
        self.op = openpose()


    # def call_Publisher(self): #パブリッシャーの呼び出し
    #     while not rospy.is_shutdown(): #Ctrl + Cが押されるまで繰り返す
    #         self.pub.send_msg() #メッセージを送信
    #         self.pub.rate.sleep()



    def success_log(self, req): #成功メッセージの表示（callback関数）
        rospy.loginfo("\nopenposeサービスのリクエストがありました：\nmessage = {}\n".format(req.openpose_request))
        self.op.make_Text()
        self.service_message.openpose_response = "openposeサービスのリクエストに応えました" #クライアントに渡す返り値のメッセージ
        return self.service_message.openpose_response #srvファイルで定義した返り値をsrvに渡す。rospy.Serviceによって呼び出された関数（callback関数）内でreturnすること



    def service_response(self): #サービスの応答
        srv = rospy.Service('openpose_service', face_recognition_service, self.success_log) #サービスのリクエストがあった場合にsuccess_log関数（callback関数）を呼び出し、実行。呼び出し先の関数内で返り値をreturnする必要がある



def main(): #メイン関数
    # 初期化し、ノードの名前を設定
    rospy.init_node('openpose_server', anonymous=True)
    #クラスのインスタンス作成（クラス内の関数や変数を使えるようにする）
    srv = Server()
    srv.service_response() #サービスの応答
    rospy.spin() #callback関数を繰り返し呼び出す（終了防止）



if __name__ == '__main__':
    try:
        main() #メイン関数の実行
    except rospy.ROSInterruptException:
        print("\n{}".format(traceback.format_exc())) #エラー内容を表示