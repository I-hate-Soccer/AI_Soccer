#!/usr/bin/python3

# Author(s): Luiz Felipe Vecchietti, Kyujin Choi, Taeyoung Kim
# Maintainer: Kyujin Choi (nav3549@kaist.ac.kr)

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../common')
try:
    from participant import Participant, Game, Frame
except ImportError as err:
    print('player_multi-qmix-dqn: "participant" module cannot be imported:', err)
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
STATE_GOALKICK = Game.GOALKICK
STATE_CORNERKICK = Game.CORNERKICK
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

def distance(x1, x2, y1, y2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

def get_defense_kick_angle(predicted_ball, field, cur_ball):
    if predicted_ball[X] >= -field[X] / 2:
        x = -field[X] / 2 - predicted_ball[X]
    else:
        x = -field[X] / 2 - cur_ball[X]
    y = predicted_ball[Y]
    return math.atan2(y, abs(x) + 0.00001)

def printConsole(message):
    print(message)
    sys.__stdout__.flush()

def degree2radian(deg):
    return deg * math.pi / 180

def wrap_to_pi(theta):
    while (theta > math.pi):
        theta -= 2 * math.pi
    while (theta < -math.pi):
        theta += 2 * math.pi
    return theta

def go_to(id, x, y, cur_posture, cur_ball, max_linear_velocity):
    sign = 1
    kd = 5
    ka = 0.3

    tod = 0.005 # tolerance of distance
    tot = math.pi/360 # tolerance of theta

    dx = x - cur_posture[id][X]
    dy = y - cur_posture[id][Y]
    d_e = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
    desired_th = math.atan2(dy, dx)

    d_th = wrap_to_pi(desired_th - cur_posture[id][TH])

    if (d_th > degree2radian(90)):
        d_th -= math.pi
        sign = -1
    elif (d_th < degree2radian(-90)):
        d_th += math.pi
        sign = -1

    if (d_e < tod):
        kd = 0
    if (abs(d_th) < tot):
        ka = 0

    left_wheel, right_wheel = set_wheel_velocity(max_linear_velocity,
                  sign * (kd * d_e - ka * d_th), 
                  sign * (kd * d_e + ka * d_th))

    return left_wheel, right_wheel

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