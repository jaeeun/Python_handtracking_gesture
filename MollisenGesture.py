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

#from pyrsistent import v
import flexbuffers
import paho.mqtt.client as mqtt

from collections import Counter
from collections import deque
from math import*
from utils import CvFpsCalc
from model import MollsenLeftHandClassifier
from model import MollsenRightHandClassifier
from model import Mollsen2HandsClassifier


mode =0 
number =0 
count = 0

cross = 0
cross_pre = 0
pre_time = datetime.datetime.now()

mollsen_left_hand_classifier = MollsenLeftHandClassifier()
mollsen_right_hand_classifier = MollsenRightHandClassifier()
mollsen_2hands_classifier = Mollsen2HandsClassifier()

# ラベル読み込み ###########################################################
with open('model/mollisen_hand_classifier/mollisen_one_hand_classifier_label.csv',
            encoding='utf-8-sig') as f:
    mollsen_left_hand_classifier_labels = csv.reader(f)
    mollsen_left_hand_classifier_labels = [
        row[0] for row in mollsen_left_hand_classifier_labels
    ]
with open('model/mollisen_hand_classifier/mollisen_one_hand_classifier_label.csv',
            encoding='utf-8-sig') as f:
    mollsen_right_hand_classifier_labels = csv.reader(f)
    mollsen_right_hand_classifier_labels = [
        row[0] for row in mollsen_right_hand_classifier_labels
    ]
with open('model/mollisen_hand_classifier/mollisen_2hands_classifier_label.csv',
            encoding='utf-8-sig') as f:
    mollsen_2hands_classifier_labels = csv.reader(f)
    mollsen_2hands_classifier_labels = [
        row[0] for row in mollsen_2hands_classifier_labels
    ]

def on_connect(client, userdata, flags, rc):
    if rc==0:
        print('Connected OK')
        client.subscribe("/hand")
    else:
        print('Bad connection Returned code = ',rc)

def on_disconnect(client, userdata, flags, rc=0):
    print(str(rc))

def on_publish(client, userdata, mid):
    blank=0
    # print('In on_pub callback mid = ',mid)

gesture_elements = {        
    "gesture" : "idle",
    "param1" : "0",
    "param2" : "0",
    "param3" : "0"
}


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    # 상대 좌표로 변환
    base_x, base_y, base_z = 0, 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y, base_z = landmark_point[0], landmark_point[1], landmark_point[2]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list[index][2] = temp_landmark_list[index][2] - base_z

    # 1次元リストに変換
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # 正規化
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def logging_csv(number, landmark_list, csv_path):
    # csv_path = 'model/mollisen_hand_classifier/mollisen_hand_data.csv'
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

def on_message(client, userdata, msg):
    data = flexbuffers.GetRoot(msg.payload).AsString
    
    landmark_left=[]
    landmark_right=[]
    temp = data.split(',')
    for i in range(0,21):
        t = [float(temp[3*i+1]),float(temp[3*i+2]),float(temp[3*i+3])]
        landmark_left.append(t)
    for i in range(21,42):
        t = [float(temp[3*i+1]),float(temp[3*i+2]),float(temp[3*i+3])]
        landmark_right.append(t)
    landmark_2hands = landmark_left+landmark_right

    #  ####################################################################

    # landmark_right = [[1,2,3],[4,5,6],[4,5,6],[4,5,6],[4,5,6],[4,5,6],[4,5,6],[4,5,6],[4,5,6],[4,5,6],[4,5,6],[4,5,6],[4,5,6],[4,5,6],[4,5,6],[4,5,6],[4,5,6],[4,5,6],[4,5,6],[4,5,6],[4,5,6]]

    # 상대 좌표, 정규화 좌표로의 변환
    pre_processed_landmark_left = pre_process_landmark(
        landmark_left)
    pre_processed_landmark_right = pre_process_landmark(
        landmark_right)
    pre_processed_landmark_2hands = pre_process_landmark(
        landmark_2hands)
    
    global mode
    global number
    global count

    global cross
    global cross_pre
    global pre_time
    
    if mode == 1:
        # 학습데이터 저장 왼 손
        logging_csv(number, pre_processed_landmark_left,'model/mollisen_hand_classifier/mollisen_left_hand_data.csv')
        print('{}\r'.format(count), end='')
        count+=1
        if count>2000: quit()

    if mode == 2:
        # 학습데이터 저장 오른 손
        logging_csv(number, pre_processed_landmark_right,'model/mollisen_hand_classifier/mollisen_right_hand_data.csv')
        print('{}\r'.format(count), end='')
        count+=1
        if count>2000: quit()

    elif mode == 3:
        # 학습데이터 저장 두손
        logging_csv(number, pre_processed_landmark_2hands,'model/mollisen_hand_classifier/mollisen_2hands_data.csv')
        print('{}\r'.format(count), end='')
        count+=1
        if count>2000: quit()

    elif mode == 0:
        # sign 분류
        hand_sign_left = mollsen_left_hand_classifier(pre_processed_landmark_left)
        hand_sign_right = mollsen_right_hand_classifier(pre_processed_landmark_right)
        hand_sign_2hands = mollsen_2hands_classifier(pre_processed_landmark_2hands)

        #print(" Left hand : "+mollsen_left_hand_classifier_labels[hand_sign_left])
        print("Right hand : "+mollsen_right_hand_classifier_labels[hand_sign_right])
        # print(" Two hands : "+mollsen_2hands_classifier_labels[hand_sign_2hands])

        gesture_elements["gesture"]=mollsen_right_hand_classifier_labels[hand_sign_right]
        
        if gesture_elements["gesture"]=="Walking":
            gesture_elements["param2"]="normal"
            if landmark_right[8][0]>landmark_right[12][0]: cross=0
            else: cross=1

            if cross != cross_pre:
                now = datetime.datetime.now()
                diff = now-pre_time
                pre_time = now
                f_diff = diff.seconds + diff.microseconds/1000000
                # print(int(50/f_diff))
                if f_diff<0.2:
                    gesture_elements["param2"]="Run"

        fbb = flexbuffers.Builder()
        fbb.MapFromElements(gesture_elements)
        data = fbb.Finish()
        client.publish("/gesture",data,1)        
        cross_pre = cross

        if hand_sign_2hands == 0:
            gesture_elements["gesture"]="Drive"
            gesture_elements["param1"]="Go"
            gesture_elements["param2"]="Front"

        elif hand_sign_2hands == 1:
            gesture_elements["gesture"]="Drive"
            gesture_elements["param1"]="Go"
            gesture_elements["param2"]="Left"

        elif hand_sign_2hands == 2:
            gesture_elements["gesture"]="Drive"
            gesture_elements["param1"]="Go"
            gesture_elements["param2"]="Right"


        elif hand_sign_2hands == 3:
            gesture_elements["gesture"]="Drive"
            gesture_elements["param1"]="Stop"
            gesture_elements["param2"]="Front"


        elif hand_sign_2hands == 4:
            gesture_elements["gesture"]="Drive"
            gesture_elements["param1"]="Stop"
            gesture_elements["param2"]="Left"

            
        elif hand_sign_2hands == 5:
            gesture_elements["gesture"]="Drive"
            gesture_elements["param1"]="Stop"
            gesture_elements["param2"]="Right"


        elif hand_sign_2hands == 6:
            gesture_elements["gesture"]="Punch"
            gesture_elements["param1"]="Left"
            gesture_elements["param2"]="Left"


        elif hand_sign_2hands == 7:
            gesture_elements["gesture"]="Punch"
            gesture_elements["param1"]="Right"
            gesture_elements["param2"]="Right"

        fbb = flexbuffers.Builder()
        fbb.MapFromElements(gesture_elements)
        data = fbb.Finish()
        client.publish("/gesture",data,1)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--number", type=int, default=0)

    args = parser.parse_args()

    return args


def main():
    global mode
    global number

    args = get_args()
    
    mode = args.mode
    number = args.number

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_publish = on_publish
    client.on_message = on_message
    client.connect('127.0.0.1', 1883)

    client.loop_forever()

if __name__ == '__main__':
    main()
