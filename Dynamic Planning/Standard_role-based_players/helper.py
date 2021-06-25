#!/usr/bin/env python3

# Author(s): Taeyoung Kim, Chansol Hong, Luiz Felipe Vecchietti
# Maintainer: Chansol Hong (cshong@rit.kaist.ac.kr)

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../common')
try:
    from participant import Game, Frame
except ImportError as err:
    print('player_rulebasedC: \'participant\' module cannot be imported:', err)
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

G = 9.81

def distance(x1, x2, y1, y2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

def degree2radian(deg):
    return deg * math.pi / 180

def radian2degree(rad):
    return rad * 180 / math.pi

def wrap_to_pi(theta):
    while (theta > math.pi):
        theta -= 2 * math.pi
    while (theta < -math.pi):
        theta += 2 * math.pi
    return theta

def predict_ball(cur_ball, previous_ball, reset_reason, prediction_step = 1):
    if reset_reason > 0:
        predicted_ball = cur_ball
    else:
        dx = cur_ball[X] - previous_ball[X]
        dy = cur_ball[Y] - previous_ball[Y]
        dz = cur_ball[Z] - previous_ball[Z]
        predicted_ball = [cur_ball[X] + prediction_step*dx, cur_ball[Y] + prediction_step*dy, max(0.05, cur_ball[Z] + prediction_step*dz -(G*0.05*prediction_step*0.05*prediction_step)/2)]
    return predicted_ball

def find_closest_robot(cur_ball, cur_posture, number_of_robots):
    min_idx = 0
    min_distance = 9999.99
    for i in range(number_of_robots):
        measured_distance = distance(cur_ball[X], cur_posture[i][X], cur_ball[Y], cur_posture[i][Y])
        if (measured_distance < min_distance):
            min_distance = measured_distance
            min_idx = i
    if (min_idx == 0):
        idx = 1
    else:
        idx = min_idx
    return idx

def find_closest_opp_robot(robot_id, cur_posture, cur_posture_opp, number_of_robots):
    min_idx = 0
    min_distance = 9999.99
    for i in range(number_of_robots):
        measured_distance = distance(cur_posture[robot_id][X], cur_posture_opp[i][X], cur_posture[robot_id][Y], cur_posture_opp[i][Y])
        if (measured_distance < min_distance):
            min_distance = measured_distance
            min_idx = i
    return min_distance

def predict_robot_velocity(cur_posture, prev_posture, index, ts):
    vx = (cur_posture[index][X] - prev_posture[index][X])/ts
    vy = (cur_posture[index][Y] - prev_posture[index][Y])/ts
    return [vx, vy]

def predict_ball_velocity(cur_ball, prev_ball, ts, reset_reason):
    if reset_reason > 0:
        vx = 0
        vy = 0
        vz = 0
    else:
        vx = (cur_ball[X] - prev_ball[X])/ts
        vy = (cur_ball[Y] - prev_ball[Y])/ts
        vz = (cur_ball[Z] - prev_ball[Z])/ts
    return [vx, vy, vz]

# Player Zone Regions

def ball_is_gk_zone(predicted_ball, field, goal_area):
    return (-field[X]/2 <= predicted_ball[X] <= -field[X]/2 + goal_area[X] +0.1 and
            -goal_area[Y]/2 - 0.1 <= predicted_ball[Y] <= goal_area[Y]/2 + 0.1)

def ball_is_d1_zone(predicted_ball, field, penalty_area):
    return (-field[X]/2 <= predicted_ball[X] <= -field[X]/2 + penalty_area[X] + 0.8 and
    	-penalty_area[Y]/2 - 0.3 <= predicted_ball[Y] <=  penalty_area[Y]/2 + 0.3)

def ball_is_d2_zone(predicted_ball, field):
    if (predicted_ball[X] < 0):
        return (predicted_ball[Y] < 0)
    else:
        return (predicted_ball[Y] < -field[Y]/(2*2))

def ball_is_f1_zone(predicted_ball, field):
    if (predicted_ball[X] < 0):
        return (predicted_ball[Y] >= 0)
    else:
        return (predicted_ball[Y] >= field[Y]/(2*2))

def ball_is_f2_zone(predicted_ball, field):
        return (predicted_ball[X] >= 0 and predicted_ball[Y] >= -field[Y]/(2*3) and predicted_ball[Y] <= -field[Y]/(2*3))

# Field Regions

def ball_is_own_goal(predicted_ball, field, goal_area):
    return (-field[X]/2 <= predicted_ball[X] <= -field[X]/2 + goal_area[X] and
            -goal_area[Y]/2 <= predicted_ball[Y] <= goal_area[Y]/2)

def ball_is_own_penalty(predicted_ball, field, penalty_area):
    return (-field[X]/2 <= predicted_ball[X] <= -field[X]/2 + penalty_area[X] and
    	-penalty_area[Y]/2 <= predicted_ball[Y] <=  penalty_area[Y]/2)

def ball_is_own_field(predicted_ball):
    return (predicted_ball[X] <= 0)

def ball_is_opp_goal(predicted_ball, field, goal_area):
    return (field[X]/2  - goal_area[X] <= predicted_ball[X] <= field[X]/2 and
            -goal_area[Y]/2 <= predicted_ball[Y] <= goal_area[Y]/2)

def ball_is_opp_penalty(predicted_ball, field, penalty_area):
    return (field[X]/2  - penalty_area[X] <= predicted_ball[X] <= field[X]/2 and
            -penalty_area[Y]/2 <= predicted_ball[Y] <= penalty_area[Y]/2)

def ball_is_opp_field(predicted_ball):
    return (predicted_ball[X] > 0)

def get_defense_kick_angle(predicted_ball, field, cur_ball):
    if predicted_ball[X] >= -field[X] / 2:
        x = -field[X] / 2 - predicted_ball[X]
    else:
        x = -field[X] / 2 - cur_ball[X]
    y = predicted_ball[Y]
    return math.atan2(y, abs(x) + 0.00001)

def get_attack_kick_angle(predicted_ball, field):
    x = field[X] / 2 - predicted_ball[X] + 0.00001
    y = predicted_ball[Y]
    angle = math.atan2(y, x)
    return -angle

def set_wheel_velocity(max_linear_velocity, left_wheel, right_wheel):
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

def printConsole(message):
    print(message)
    sys.__stdout__.flush()

def print_debug_flag(self):
    printConsole('TEAM:' + str(self.team.flag))
    printConsole("--------")
