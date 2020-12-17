#!/usr/bin/env python
# -*- coding: utf-8 -*-
#上記2行は必須構文のため、コメント文だと思って削除しないこと
#Python2.7用プログラム

import os
from numpy import linalg as LA
import numpy as np
from natsort import natsorted
from glob import glob
import time
import rospy
from face_recognition_pkg.msg import face_recognition_message #メッセージファイルの読み込み（from パッケージ名.msg import 拡張子なしメッセージファイル名）
from face_recognition_pkg.srv import calculate_service #サービスファイルの読み込み（from パッケージ名.srv import 拡張子なしサービスファイル名）
import json
import sys
import traceback #エラーメッセージの表示に使用



class Calculate():
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
    json_file_number = 0 #jsonファイル番号
    json_file_path = '/home/limlab/catkin_ws/src/face_recognition_pkg/output' #jsonファイルのあるパスを指定



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
        self.rate = rospy.Rate(5)



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

        return angle



    def face_vector_calculate(self, point_list2, json_object): #m,nベクトルの長さや差の取得
        if self.json_object_people_length == 0:  # 取得した人数が0人の場合
            self.diff = 0  # 2つのベクトルの長さの差を0にする
    
        # if (self.p11_x == 0) or (self.p12_x == 0):  # 関節点11 or 12が取得できなかった場合
        #     self.diff = 0
        #     print("(self.p11_x == 0) or (self.p12_x == 0)")
    
        else:  # json形式のファイルの中身から各関節点のx,y座標の値を取り出す
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
    
        elif a != 0:  # 検出できた場合
            if self.which == 0:  # 右向き
                self.message1 = 'Right'
            elif self.which == 1:  # 正面
                self.message1 = 'Front'
            elif self.which == 2:  # 左向き
                self.message1 = 'Left'
        return self.message1



    def json_file_process(self): #jsonファイルの処理
        json_files = natsorted(glob(self.json_file_path + '/*_keypoints.json')) #指定パスにある"呼び出された段階での"全てのjsonファイルを人間が扱う数の自然な順番(natsorted：natural sorted。0から1,..10,..,100,..)に読み込む。globは*（ワイルドカード：任意の変数xに相当）を扱えるようにするもの。json形式ファイルが増えていく度にjson_filesの中身を更新する必要があるため、毎回呼び出されるところ（今回はこの行）に記述
        #print(json_files)
        json_file = open(json_files[self.json_file_number], mode = 'r') # OpenPoseにより書き込まれた「〜_keypoints.json」のファイルを読み込む
        print(u"{}".format(os.path.basename(str(json_file))).encode("utf-8")) #日本語も扱えるutf-8型にエンコード

        json_object = json.load(json_file)  # json形式ファイルの中身を読み込む
        json_file.close()
        openpose_version = json_object['version'] #OpenPoseのバージョンを取得s
        self.get_json_object_length(json_object) #json_objectの要素数の取得
        a = self.joint_angle(json_object, self.point_list1, self.json_object_3point_length_list)  #2つのベクトルのなす角の取得
        # 関数の返り値をいっぺんに取得
        vector_info = self.face_vector_calculate(self.point_list2, json_object) #m,nベクトルの長さや差の取得
        # m_vector.append(vector_info[0]) #配列にmベクトルの長さ(m_length)の追加
        # n_vector.append(vector_info[1]) #配列にmベクトルの長さ(n_length)の追加
        # diff_vector.append(vector_info[2]) #配列に2つのベクトルの差(diff)の追加
        # m_length = vector_info[0] #mベクトルの長さ(m_length)の取得
        # n_length = vector_info[1] #nベクトルの長さ(n_length)の取得
        self.diff = vector_info[2]  # 2つのベクトルの長さの差(diff)を取得
        self.p6_x = vector_info[3]
        self.p7_x = vector_info[4]
        self.message1= self.angle_judge(a) #2つのベクトルのなす角による顔の向きの判定
        self.json_file_number += 1
        return json_file, a, openpose_version, self.message1



class Server(): #サーバーのクラス
    def __init__(self):
        self.calculate_message = calculate_service()
        self.cal = Calculate() #Calculateクラスのインスタンス化(実体化)
        # self.rate = rospy.Rate(5)
        self.count = 0



    def call_Calculate(self): #Calculate関数を呼び出す
        return self.cal.json_file_process() #json_file_process関数の返り値を返す



    def make_msg(self): #送信するメッセージの作成
        json_file, a, openpose_version, message1= self.call_Calculate() #Calculate関数を呼び出す
        #配信するメッセージの作成
        self.calculate_message.a = a
        self.calculate_message.openpose_version = u"{}".format(openpose_version).encode("utf-8") #日本語も扱えるutf-8型にエンコード
        self.calculate_message.message1 = u"{}".format(message1).encode("utf-8") 
        return self.calculate_message.a, self.calculate_message.openpose_version, self.calculate_message.message1



    def success_log(self, req): #成功メッセージの表示（callback関数）
        print("count:{}".format(self.count))
        self.count += 1
        self.calculate_message.a, self.calculate_message.openpose_version, self.calculate_message.message1 = self.make_msg()
        return self.calculate_message.a, self.calculate_message.openpose_version, self.calculate_message.message1 #srvファイルで定義した返り値をsrvに渡す。rospy.Serviceによって呼び出された関数（callback関数）内でreturnすること



    def service_response(self): #サービスの応答
        srv = rospy.Service('calculate_service', calculate_service, self.success_log) #サービスのリクエストがあった場合にsuccess_log関数（callback関数）を呼び出し、実行。呼び出し先の関数内で返り値をreturnする必要がある



def main(): #メイン関数
    #初期化し、ノードの名前を設定
    rospy.init_node('calculate', anonymous=True)
    srv = Server()
    srv.service_response() #サービスの応答
    rospy.spin() #callback関数を繰り返し呼び出す（終了防止）



if __name__ == '__main__':
    try:
        main() #メイン関数の実行
    except rospy.ROSInterruptException:
        print("\n{}".format(traceback.format_exc())) #エラー内容を表示
