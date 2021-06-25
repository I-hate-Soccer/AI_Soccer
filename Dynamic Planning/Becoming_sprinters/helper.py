#!/usr/bin/env python3

# Author(s): Taeyoung Kim, Chansol Hong, Luiz Felipe Vecchietti
# Maintainer: Chansol Hong (cshong@rit.kaist.ac.kr)
#############################################
#####전략 실행, 상황 정보 계산하는 보조 함수들#####
#####거리와 속도를 측정할 수 있도록 도와줌   ######
#############################################
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../common')
try:
    from participant import Game, Frame
except ImportError as err:
    print('player_rulebasedA: \'participant\' module cannot be imported:', err)
    raise

import math

#reset_reason
NONE = Game.NONE
GAME_START = Game.GAME_START
SCORE_MYTEAM = Game.SCORE_MYTEAM
SCORE_OPPONENT = Game.SCORE_OPPONENT
GAME_END = Game.GAME_END
DEADLOCK = Game.DEADLOCK
GOALKICK = Game.GOALKICK
CORNERKICK = Game.CORNERKICK
PENALTYKICK = Game.PENALTYKICK
HALFTIME = Game.HALFTIME
EPISODE_END = Game.EPISODE_END

#game_state
STATE_DEFAULT = Game.STATE_DEFAULT
STATE_KICKOFF = Game.STATE_KICKOFF
STATE_GOALKICK = Game.STATE_GOALKICK
STATE_CORNERKICK = Game.STATE_CORNERKICK
STATE_PENALTYKICK = Game.STATE_PENALTYKICK

#coordinates
MY_TEAM = Frame.MY_TEAM
OP_TEAM = Frame.OP_TEAM
BALL = Frame.BALL
X = Frame.X
Y = Frame.Y
Z = Frame.Z
TH = Frame.TH
ACTIVE = Frame.ACTIVE
TOUCH = Frame.TOUCH
BALL_POSSESSION = Frame.BALL_POSSESSION

def distance(x1, x2, y1, y2):   #두 점 사이의 거리
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

def degree2radian(deg):     #각도 단위 변환 degree -> radian
    return deg * math.pi / 180

def radian2degree(rad):     #각도 단위 변환 radian -> degree
    return rad * 180 / math.pi

def wrap_to_pi(theta):      #각도를 [-pi, pi]범위 변환
    while (theta > math.pi):
        theta -= 2 * math.pi
    while (theta < -math.pi):
        theta += 2 * math.pi
    return theta

def predict_ball(cur_ball, previous_ball):  #prediction_step 이후(다음 time step에서) 예측된 공 위치
    prediction_step = 1     #값을 키우거나 공과 로봇의 충동점을 예측하는 방식으로 사용해도 된대요
    dx = cur_ball[X] - previous_ball[X]
    dy = cur_ball[Y] - previous_ball[Y]
    predicted_ball = [cur_ball[X] + prediction_step*dx, cur_ball[Y] + prediction_step*dy]
    return predicted_ball

def find_closest_robot(cur_ball, cur_posture, number_of_robots):    #공과 가장 가까운 로봇의 번호
    min_idx = 0
    min_distance = 9999.99
    for i in range(number_of_robots):
        measured_distance = distance(cur_ball[X], cur_posture[i][X], cur_ball[Y], cur_posture[i][Y])
        if (measured_distance < min_distance):
            min_distance = measured_distance
            min_idx = i
    if (min_idx == 0):  #가까운 로봇이 골기퍼이면 골키퍼 대신 수비수1을 보냄. 지금 전략이 골기퍼는 보내지 않는 것이기 때문
        idx = 1
    else:
        idx = min_idx
    return idx

def ball_is_own_goal(predicted_ball, field, goal_area):     #공이 우리 팀의 영역에 있는지 위치 판단(true/false)
    return (-field[X]/2 <= predicted_ball[X] <= -field[X]/2 + goal_area[X] and
            -goal_area[Y]/2 <= predicted_ball[Y] <= goal_area[Y]/2)

def ball_is_own_penalty(predicted_ball, field, penalty_area):   #공이 패널티 영역에 있는지 위치 판단(true/false)
    return (-field[X]/2 <= predicted_ball[X] <= -field[X]/2 + penalty_area[X] and
    	-penalty_area[Y]/2 <= predicted_ball[Y] <=  penalty_area[Y]/2)

def ball_is_own_field(predicted_ball):      #공이 자신의 필드 영역에 있는지 위치 판단(true/false)
    return (predicted_ball[X] <= 0)

def ball_is_opp_goal(predicted_ball, field, goal_area):     #공이 상대방 팀의 골 영역에 있는지 위치 판단(true/false)
    return (field[X]/2  - goal_area[X] <= predicted_ball[X] <= field[X]/2 and
            -goal_area[Y]/2 <= predicted_ball[Y] <= goal_area[Y]/2)

def ball_is_opp_penalty(predicted_ball, field, penalty_area):   #공이 상대방 팀의 패널티 영역에 있는지 위치 판단(true/false)
    return (field[X]/2  - penalty_area[X] <= predicted_ball[X] <= field[X]/2 and
            -penalty_area[Y]/2 <= predicted_ball[Y] <= penalty_area[Y]/2)

def ball_is_opp_field(predicted_ball):      #공이 상대방 팀의 자신의 필드 영역에 있는지 위치 판단(true/false)
    return (predicted_ball[X] > 0)

def get_defense_kick_angle(predicted_ball, field, cur_ball):    #내 팀 골대중앙과 공을 이은 직선, 내 팀 골대중앙과 경기장 중앙을 이은 직선 사이의 각도 반환, 공과 우리 골대 사이의 각도
    if predicted_ball[X] >= -field[X] / 2:
        x = -field[X] / 2 - predicted_ball[X]
    else:
        x = -field[X] / 2 - cur_ball[X]
    y = predicted_ball[Y]
    return math.atan2(y, abs(x) + 0.00001)

def get_attack_kick_angle(predicted_ball, field):   #상대 팀 골대중앙과 공을 이은 직선, 상대 팀 골대중앙과 경기장 중앙을 이은 직선 사이의 각도 반환, 공과 상대 골대 사이의 각도

    x = field[X] / 2 - predicted_ball[X] + 0.00001
    y = predicted_ball[Y]
    angle = math.atan2(y, x)
    return -angle

def set_wheel_velocity(max_linear_velocity, left_wheel, right_wheel):   #왼쪽 or 오른쪽 바퀴 속도가 최대속도를 넘는 값이면 비율을 유지하며 더 큰 속도를 최대속도에 맞추고 변경된 바퀴 속도 값 반환
    #컨트롤러부터 받은 바퀴 속도가 최대 속도를 초과할때 사용, 속도를 적정 비율로 줄여 simulator에서 문제 없이 실행되도록 함
    ratio_l = 1
    ratio_r = 1

    if (left_wheel > max_linear_velocity or right_wheel > max_linear_velocity):
        diff = max(left_wheel, right_wheel) - max_linear_velocity
        left_wheel -= diff
        right_wheel -= diff
    if (left_wheel < -max_linear_velocity or right_wheel < -max_linear_velocity):
        diff = min(left_wheel, right_wheel) + max_linear_velocity
        left_wheel -= diff
        right_wheel -= diff

    return left_wheel, right_wheel

def printConsole(message):  #message를 시뮬레이터 Console에 print
    print(message)
    sys.__stdout__.flush()

def print_debug_flag(self): #현재 timestep 에서 코드의 어느 부분이 실행되었는지와 연결된 Flag를 Console에 print
    printConsole('GK:' + str(self.GK.flag))
    printConsole('D1:' + str(self.D1.flag))
    printConsole('D2:' + str(self.D2.flag))
    printConsole('F1:' + str(self.F1.flag))
    printConsole('F2:' + str(self.F2.flag))
    printConsole("--------")
