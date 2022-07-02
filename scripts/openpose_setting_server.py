#!/usr/bin/env python
# -*- coding: utf-8 -*-
#上記2行は必須構文のため、コメント文だと思って削除しないこと
#Python2.7用プログラム

from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.models import load_model  # ニューラルネットワーク学習ライブラリkerasの読み込み
from tensorflow.python import keras
# cv2の読み込みエラー防止
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os #ファイルパスを扱うのに必要
import rospy #ROSをPythonで扱うのに必要s
from face_recognition_pkg.srv import face_recognition_service #サービスファイルの読み込み（from パッケージ名.srv import 拡張子なしサービスファイル名）
import traceback #エラーメッセージの表示に使用
import json #jsonファイルを扱うのに必要

# TensorFlowでGPUを強制的に使用
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

#ファイルパスの指定
json_output_path = "/home/limlab/catkin_ws/src/face_recognition_pkg/output"
openpose_params_json_file = "/home/limlab/catkin_ws/src/face_recognition_pkg/openpose_params.json"



class OpenPose(): #OpenPoseのクラス
    def __init__(self):
        self.params = dict() #空の辞書型配列（キーとデータのセット）の作成（OpenPoseのオプションなどの情報が入る）


    def openpose_setting(self): #OpenPoseの設定
        #OpenPose開始時のオプション指定（その他のパラメータについては、include /openpose /flags.hppを参照）
        self.params["model_folder"] = "/home/limlab/catkin_ws/src/face_recognition_pkg/models/" #キー：model_folder、データ：../models/
        self.params["write_json"] = json_output_path #キー：write_json、データ：json_output_path。座標情報をjson形式ファイルとして書き込み
        self.params["part_candidates"] = True #キー：part_candidates、データ：True
        self.params["face"] = True #キー：face、データ：True。顔検出を有効
        #params["hand"] = True #キー：hand、データ：True。手検出を有効
        self.params["net_resolution"] = "192x48" #キー：net_resolution、データ：320x176。256x112。208x64。192x48。認識精度は落ちるが、処理速度は向上。16の倍数で数字を指定

        self.write_json() #jsonファイルに書き込み



    def write_json(self): #jsonファイルに書き込み
        with open(openpose_params_json_file, mode="w") as f:
            json_object = json.dumps(self.params, f)
            f.write(json_object) #jsonファイルに書き込み



    def make_Text(self): #確認メッセージの作成
        self.openpose_setting() #OpenPoseの設定
        Text = "\nOpenPoseの設定完了\nmodel_folder：{}\nwrite_json：{}\npart_candidates：{}\nface：{}\nnet_resolution：{}\n".format(self.params["model_folder"], self.params["write_json"], self.params["part_candidates"], self.params["face"], self.params["net_resolution"])
        return Text




class Server(): #サーバーのクラス
    def __init__(self):
        #service_messageの型を作成
        self.service_message = face_recognition_service()
        #self.pub = Publishsers() #パブリッシャークラスのインスタンス化(実体化)
        self.op = OpenPose()



    def success_log(self, req): #成功メッセージの表示（callback関数）
        rospy.loginfo("\nopenposeサービスのリクエストがありました：\nmessage = {}\n".format(req.openpose_request))
        self.op.make_Text()
        self.service_message.openpose_response = "openposeサービスのリクエストに応えました" #クライアントに渡す返り値のメッセージ
        return self.service_message.openpose_response #srvファイルで定義した返り値をsrvに渡す。rospy.Serviceによって呼び出された関数（callback関数）内でreturnすること



    def service_response(self): #サービスの応答
        srv = rospy.Service('openpose_service', face_recognition_service, self.success_log) #サービスのリクエストがあった場合にsuccess_log関数（callback関数）を呼び出し、実行。呼び出し先の関数内で返り値をreturnする必要がある



def main(): #メイン関数
    # 初期化し、ノードの名前を設定
    rospy.init_node('openpose_setting_server', anonymous=True)
    #クラスのインスタンス作成（クラス内の関数や変数を使えるようにする）
    srv = Server()
    srv.service_response() #サービスの応答
    rospy.spin() #callback関数を繰り返し呼び出す（終了防止）



if __name__ == '__main__':
    try:
        main() #メイン関数の実行
    except rospy.ROSInterruptException:
        print("\n{}".format(traceback.format_exc())) #エラー内容を表示