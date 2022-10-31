#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp
import datetime
import math

# from pyrsistent import v
import flexbuffers
import paho.mqtt.client as mqtt

from collections import Counter
from collections import deque
# from sys import platlibdir
from math import*
from utils import CvFpsCalc
from model import KeyPointAndroidClassifier


info_elements = {
    "type" : "android",
    "screensize" : "0,0",
}
finger_elements = {
    "hand" : "Right",
    "landmark" : "0",
}
gesture_elements = {
    "gesture" : "idle",
    "param1" : "0",
    "param2" : "0",
    "param3" : "0"
}
landmark_point = []
landmark_3dpoint = []

cam_width=0
cam_height=0

keypoint_android_classifier = KeyPointAncroidClassifier()

with open('model/keypoint_android_classifier/keypoint_android_classifier_label.csv',
            encoding='utf-8-sig') as f:
    keypoint_android_classifier_labels = csv.reader(f)
    keypoint_android_classifier_labels = [
        row[0] for row in keypoint_android_classifier_labels
    ]

def on_connect(client, userdata, flags, rc):
    if rc==0:
        print('Connected OK')
        client.subscribe("/android_hand")
        client.subscribe("/android_info")
    else:
        print('Bad connection Returned code = ',rc)

def on_disconnect(client, userdata, flags, rc=0):
    print(str(rc))

def on_publish(client, userdata, mid):
    blank=0
    # print('In on_pub callback mid = ',mid)

def on_message(client, userdata, msg):
    # data = flexbuffers.GetRoot(msg.payload).AsString
    
    st = str(msg.payload.decode("utf-8"))
    arr = st.split(',')
    
    global cam_width, cam_height
    
    if msg.topic == "/android_info":
        info_elements["screensize"] = st
        
        fbb = flexbuffers.Builder()
        fbb.MapFromElements(info_elements)
        data = fbb.Finish()
        client.publish("/info",data,1)
        
        cam_width = float(arr[0])
        cam_height = float(arr[1])
        
        print("/android_info  ,  screensize : "+str(arr))
        
    elif msg.topic == "/android_hand":
        landmark_point=[]
        landmark_3dpoint = []
        
        for i in range(0, 21):
            landmark_x = float(arr[i*3])
            landmark_y = float(arr[i*3 + 1]) * cam_width
            landmark_z = float(arr[i*3 + 2]) * cam_height

            landmark_point.append([landmark_x, landmark_y])
            landmark_3dpoint.append([landmark_x, landmark_y, landmark_z])
            
        # print(landmark_point)
        
        # finger_elements["hand"] = handedness.classification[0].label[0:]
        finger_elements["landmark"] = st
        
        fbb = flexbuffers.Builder()
        fbb.MapFromElements(finger_elements)
        data = fbb.Finish()
        client.publish("/finger",data,1)
        
        pre_processed_landmark_list = pre_process_landmark(landmark_point)
        
        # sign 분류
        hand_sign_id = keypoint_android_classifier(pre_processed_landmark_list)
        
        print("/android_hand   ,  " + keypoint_android_classifier_labels[hand_sign_id])
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help='0:recognize, 1:save right, 2:save left, 3: save both', type=int, default=0)
    parser.add_argument("--number", help='to save gesture number', type=int, default=0)

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    
    arg_mode = args.mode
    arg_number = args.number

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_publish = on_publish
    client.on_message = on_message
    client.connect('13.124.29.179', 1883)
    client.loop_forever()
    
    use_brect = True

    info_count=0


    

    # ラベル読み込み ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
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

    #  ########################################################################
    mode = 0
    cross = 0
    cross_pre = 0
    pre_time = datetime.datetime.now()
    last_point_gun = [0,0,0]



        #  ####################################################################
    #     if results.multi_hand_landmarks is not None:
    #         for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
    #                                               results.multi_handedness):                                  
    #             mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    #             # calc_bounding_rect
    #             brect = calc_bounding_rect(debug_image, hand_landmarks)
    #             # 랜드마크
    #             landmark_list, landmark_3Dlist = calc_landmark_list(debug_image, hand_landmarks)

    #             # 상대 좌표, 정규화 좌표로의 변환
    #             pre_processed_landmark_list = pre_process_landmark(
    #                 landmark_list)
    #             pre_processed_point_history_list = pre_process_point_history(
    #                 debug_image, point_history)
    #             # 학습데이터 저장
    #             logging_csv(number, mode, pre_processed_landmark_list,
    #                         pre_processed_point_history_list)
                
    #             finger_elements["hand"] = handedness.classification[0].label[0:]
    #             finger_elements["landmark"] = ""
    #             for landmark in landmark_3Dlist:
    #                 finger_elements["landmark"]=finger_elements["landmark"]+str(landmark[2])+','+str(landmark[0])+','+str(landmark[1])+','
                
    #             # print(finger_elements)

    #             fbb = flexbuffers.Builder()
    #             fbb.MapFromElements(finger_elements)
    #             data = fbb.Finish()
    #             client.publish("/finger",data,1)

    #             # sign 분류
    #             hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

    #             # 포인팅
    #             if hand_sign_id == 2:
    #                 point_history.append(landmark_list[8])

    #             # 걷기 뛰기
    #             elif hand_sign_id == 5:
    #                 point_history.append(landmark_list[8])
    #                 point_history.append(landmark_list[12])

    #                 gesture_elements["gesture"]="Walking"
                    
    #                 v0 = [1,0]
    #                 v1 = [landmark_list[6][0]-landmark_list[5][0],landmark_list[6][1]-landmark_list[5][1]]
    #                 v2 = [landmark_list[10][0]-landmark_list[9][0],landmark_list[10][1]-landmark_list[9][1]]
                    
    #                 a1 = angle(v0,v1)
    #                 a2 = angle(v0,v2)
                    
    #                 a1=math.degrees(a1)-90
    #                 a2=math.degrees(a2)-90
    #                 aa = (a1+a2)/2

    #                 if abs(aa)<15:
    #                     gesture_elements["param1"]="front"
    #                 elif aa<0:
    #                     gesture_elements["param1"]="left"
    #                 else:
    #                     gesture_elements["param1"]="right"

    #                 gesture_elements["param2"]="normal"

    #                 # print('fingers degree : '+str(aa) +'  direction : '+gesture_elements["param1"])

    #                 if landmark_list[8][1]>landmark_list[12][1]: cross=0
    #                 else: cross=1

    #                 if cross != cross_pre:
    #                     now = datetime.datetime.now()
    #                     diff = now-pre_time
    #                     pre_time = now
    #                     f_diff = diff.seconds + diff.microseconds/1000000
    #                     # print(int(50/f_diff))
    #                     if f_diff<0.2:
    #                         gesture_elements["param2"]="Run"

    #                 print(gesture_elements)

    #                 fbb = flexbuffers.Builder()
    #                 fbb.MapFromElements(gesture_elements)
    #                 data = fbb.Finish()
    #                 client.publish("/gesture",data,1)        
    #                 cross_pre = cross

    #             # 총 쏘기
    #             elif hand_sign_id == 7 or hand_sign_id==6:
                    
    #                 v1 = [landmark_3Dlist[8][0]-landmark_3Dlist[7][0],landmark_3Dlist[8][1]-landmark_3Dlist[7][1],landmark_3Dlist[8][2]-landmark_3Dlist[7][2]]
    #                 v2 = [landmark_3Dlist[5][0]-landmark_3Dlist[6][0],landmark_3Dlist[5][1]-landmark_3Dlist[6][1],landmark_3Dlist[5][2]-landmark_3Dlist[6][2]]

    #                 v_in = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

    #                 s_v1=sqrt(pow(v1[0],2)+pow(v1[1],2)+pow(v1[2],2))
    #                 s_v2=sqrt(pow(v2[0],2)+pow(v2[1],2)+pow(v2[2],2))

    #                 degree = int(degrees(acos(v_in/(s_v1*s_v2))))
                    
    #                 out=[]
    #                 out.append(v1[1]*v2[2]-v1[2]*v2[1])
    #                 out.append(v1[2]*v2[0]-v1[0]*v2[2])
    #                 out.append(v1[0]*v2[1]-v1[1]*v2[0])
    #                 v_in = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

    #                 gesture_elements["gesture"]= "Gun"
    #                 gesture_elements["param1"] = str(landmark_list[5][0])+','+str(landmark_list[5][1])
    #                 gesture_elements["param2"] = str(degree)

    #                 cv.putText(debug_image, "degree:" + str(degree), (400, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0,0), 2, cv.LINE_AA)
    #                 cv.putText(debug_image, "degree:" + str(degree), (400, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)       

    #                 fbb = flexbuffers.Builder()
    #                 fbb.MapFromElements(gesture_elements)
    #                 data = fbb.Finish()
    #                 client.publish("/gesture",data,1)

    #                 if degree<120:
    #                     now = datetime.datetime.now()
    #                     diff = now - pre_time
    #                     f_diff = diff.seconds + diff.microseconds/1000000
    #                     #pre_time = now
                        
    #                     print(f_diff)

    #                     if f_diff>0.1:
    #                         print("Shoot : "+str(degree))
    #                         gesture_elements["gesture"] = "Shoot"
    #                         gesture_elements["param1"] = str(last_point_gun[0])+','+str(last_point_gun[1])
    #                         gesture_elements["param2"] = str(degree)

    #                         # print(gesture_elements)

    #                         fbb = flexbuffers.Builder()
    #                         fbb.MapFromElements(gesture_elements)
    #                         data = fbb.Finish()
    #                         client.publish("/gesture",data,1)
    #                         pre_time = now
    #                 else:
    #                     last_point_gun = landmark_list[8]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []
    landmark_3dpoint = []

    # 키 포인트
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z * image_height

        landmark_point.append([landmark_x, landmark_y])
        landmark_3dpoint.append([landmark_x, landmark_y, landmark_z])

    return landmark_point,landmark_3dpoint


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # 상대 좌표로 변환
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

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 11):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

if __name__ == '__main__':
    main()
