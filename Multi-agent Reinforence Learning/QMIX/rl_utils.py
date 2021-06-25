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

try:
    import _pickle as pickle
except:
    import pickle
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable

import helper

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
TH = Frame.TH
ACTIVE = Frame.ACTIVE
TOUCH = Frame.TOUCH

#robot_index
GK_INDEX = 0
D1_INDEX = 1
D2_INDEX = 2
F1_INDEX = 3
F2_INDEX = 4

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LONG = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

def predict_ball_velocity(cur_ball, prev_ball, ts):
    vx = (cur_ball[X] - prev_ball[X])/ts
    vy = (cur_ball[Y] - prev_ball[Y])/ts
    vd = math.atan2(vy, vx)
    vr = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
    return [vd*180/math.pi, vr]

def predict_robot_velocity(cur_posture, prev_posture, index, ts):
    vx = (cur_posture[index][X] - prev_posture[index][X])/ts
    vy = (cur_posture[index][Y] - prev_posture[index][Y])/ts
    vd = math.atan2(vy, vx)
    vr = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
    return [vd*180/math.pi, vr]

def predict_ball(cur_ball, previous_ball):
    prediction_step = 2
    dx = cur_ball[X] - previous_ball[X]
    dy = cur_ball[Y] - previous_ball[Y]
    predicted_ball = [cur_ball[X] + prediction_step*dx, cur_ball[Y] + prediction_step*dy]
    return predicted_ball

def get_state(cur_posture, prev_posture, cur_posture_opp, prev_posture_opp, cur_ball, prev_ball, field, goal, max_linear_velocity):
    # relative state: (shape: 8)
    states = [[] for _ in range(5)]
    pxx = field[X] + goal[X]
    pyy = field[Y]
    defense_angle = helper.get_defense_kick_angle(cur_ball, field, cur_ball)
    defense_x = math.cos(defense_angle)*0.6 - field[X]/2
    defense_y = math.sin(defense_angle)*0.6
    ball_velocity = predict_ball_velocity(cur_ball, prev_ball, 0.05)
    for i in range(5):
        if i == 0:
            states[i] =[round((defense_x - cur_posture[i][X])/pxx, 2), round((defense_y - cur_posture[i][Y])/pyy, 2),  
                        round((cur_ball[X] - field[X]/2)/pxx, 2), round((cur_ball[Y] - 0)/pyy, 2),
                        round((-field[X]/2 - cur_posture[i][X])/pxx, 2), round((0 - cur_posture[i][Y])/pyy, 2),
                        round(cur_posture[i][TH], 2), round((math.atan2(cur_ball[Y]-cur_posture[i][Y], cur_ball[X]-cur_posture[i][X]) - cur_posture[i][TH])/math.pi, 2),
                        round(ball_velocity[X]/180, 2), round(ball_velocity[Y], 2)]
        else:
            states[i] =[round((cur_ball[X] - cur_posture[i][X])/pxx, 2), round((cur_ball[Y] - cur_posture[i][Y])/pyy, 2),  
                        round((cur_ball[X] - field[X]/2)/pxx, 2), round((cur_ball[Y] - 0)/pyy, 2),
                        round((-field[X]/2 - cur_posture[i][X])/pxx, 2), round((0 - cur_posture[i][Y])/pyy, 2),
                        round(cur_posture[i][TH], 2), round((math.atan2(cur_ball[Y]-cur_posture[i][Y], cur_ball[X]-cur_posture[i][X]) - cur_posture[i][TH])/math.pi, 2),
                        round(ball_velocity[X]/180, 2), round(ball_velocity[Y], 2)]

    return states

def get_global_state(cur_posture, prev_posture, cur_posture_opp, prev_posture_opp, cur_ball, prev_ball, field, goal, max_linear_velocity):
    ##### for state
    ball_velocity = predict_ball_velocity(cur_ball, prev_ball, 0.05)
    robot_velocity = [predict_robot_velocity(cur_posture, prev_posture, a, 0.05) for a in range(5)]
    opp_robot_velocity = [predict_robot_velocity(cur_posture_opp, prev_posture_opp, a, 0.05) for a in range(5)]
    pxx = field[X] + goal[X]
    pyy = field[Y]
    states = [round((-field[X]/2 - cur_ball[X])/pxx, 2), round((0 - cur_ball[Y])/pyy, 2), 
     round((-field[X]/2 - cur_posture[0][X])/pxx, 2), round((0 - cur_posture[0][Y])/pyy, 2), round(cur_posture[0][TH]/math.pi, 2),
     round((-field[X]/2 - cur_posture[1][X])/pxx, 2), round((0 - cur_posture[1][Y])/pyy, 2), round(cur_posture[1][TH]/math.pi, 2), 
     round((-field[X]/2 - cur_posture[2][X])/pxx, 2), round((0 - cur_posture[2][Y])/pyy, 2), round(cur_posture[2][TH]/math.pi, 2),
     round((-field[X]/2 - cur_posture[3][X])/pxx, 2), round((0 - cur_posture[3][Y])/pyy, 2), round(cur_posture[3][TH]/math.pi, 2), 
     round((-field[X]/2 - cur_posture[4][X])/pxx, 2), round((0 - cur_posture[4][Y])/pyy, 2), round(cur_posture[4][TH]/math.pi, 2), 
     round((-field[X]/2 - cur_posture_opp[0][X])/pxx, 2), round((0 - cur_posture_opp[0][Y])/pyy, 2),
     round((-field[X]/2 - cur_posture_opp[1][X])/pxx, 2), round((0 - cur_posture_opp[1][Y])/pyy, 2),  
     round((-field[X]/2 - cur_posture_opp[2][X])/pxx, 2), round((0 - cur_posture_opp[2][Y])/pyy, 2), 
     round((-field[X]/2 - cur_posture_opp[3][X])/pxx, 2), round((0 - cur_posture_opp[3][Y])/pyy, 2), 
     round((-field[X]/2 - cur_posture_opp[4][X])/pxx, 2), round((0 - cur_posture_opp[4][Y]) /pyy, 2),
     round(cur_posture[0][TH], 2), round((math.atan2(cur_ball[Y]-cur_posture[0][Y], cur_ball[X]-cur_posture[0][X]) - cur_posture[0][TH])/math.pi, 2),
     round(cur_posture[1][TH], 2), round((math.atan2(cur_ball[Y]-cur_posture[1][Y], cur_ball[X]-cur_posture[1][X]) - cur_posture[1][TH])/math.pi, 2),
     round(cur_posture[2][TH], 2), round((math.atan2(cur_ball[Y]-cur_posture[2][Y], cur_ball[X]-cur_posture[2][X]) - cur_posture[2][TH])/math.pi, 2),
     round(cur_posture[3][TH], 2), round((math.atan2(cur_ball[Y]-cur_posture[3][Y], cur_ball[X]-cur_posture[3][X]) - cur_posture[3][TH])/math.pi, 2),
     round(cur_posture[4][TH], 2), round((math.atan2(cur_ball[Y]-cur_posture[4][Y], cur_ball[X]-cur_posture[4][X]) - cur_posture[4][TH])/math.pi, 2)]
    return states

def get_reward(cur_posture, prev_posture, cur_ball, prev_ball, field, id, epsilon):
    dist_robot2ball = helper.distance(cur_posture[id][X] , cur_ball[X], cur_posture[id][Y], cur_ball[Y])
    defense_angle = helper.get_defense_kick_angle(cur_ball, field, cur_ball)
    defense_x = math.cos(defense_angle)*0.6 - field[X]/2
    defense_y = math.sin(defense_angle)*0.6
    defense_dis = helper.distance(cur_posture[id][X], defense_x, cur_posture[id][Y], defense_y)
    robot_th_error = abs(math.atan2(cur_ball[Y]-cur_posture[id][Y], cur_ball[X]-cur_posture[id][X]) - cur_posture[id][TH])
    ball_robot_dis = helper.distance(cur_posture[id][X] , cur_ball[X], cur_posture[id][Y], cur_ball[Y])
    ball_robot_dis_prev = helper.distance(prev_posture[id][X] , prev_ball[X], prev_posture[id][Y], prev_ball[Y])
    ball_robot_velocity = (ball_robot_dis_prev - ball_robot_dis)/0.05
    ball_velocity = predict_ball_velocity(cur_ball, prev_ball, 0.05)

    score = 0
    concede = 0
    if cur_ball[X] > field[X]/2 :
        score = 10
    if cur_ball[X] < -field[X]/2 :
        concede = -10

    if id == 0:
        gk_th_error = abs(math.atan2(defense_y - cur_posture[id][Y], defense_x - cur_posture[id][X]) - cur_posture[id][TH])
        return concede + epsilon*0.5*(math.exp(-1*defense_dis)) + epsilon*0.5*(math.exp(-1*gk_th_error))
    elif id == 1 or id == 2:
        return concede + epsilon*0.5*(math.exp(-1*dist_robot2ball)) + epsilon*0.5*(math.exp(-1*robot_th_error)) + min(ball_velocity[Y], 1)
    else:
        return score + epsilon*0.5*(math.exp(-1*dist_robot2ball)) + epsilon*0.5*(math.exp(-1*robot_th_error)) + min(ball_velocity[Y], 1)

def get_action(robot_id, action_number, max_linear_velocity, cur_posture, cur_ball, prev_ball, field):
    # 5 actions: go forwards, go backwards, rotate right, rotate left, stop
    predicted_ball = predict_ball(cur_ball, prev_ball)

    defense_angle = helper.get_defense_kick_angle(predicted_ball, field, cur_ball)
    defense_x = math.cos(defense_angle)*0.6 - field[X]/2
    defense_y = math.sin(defense_angle)*0.6
    delta = 1.5

    GK_WHEELS = [[predicted_ball[X],  predicted_ball[Y],  0.0,  0.0,  0.0, 0.0],
                [defense_x,  defense_y,  5.0,  0.0,  0.0, 0.0],
                [defense_x,  defense_y,  10.0,  10.0,  0.0, 0.0],
                [defense_x, defense_y,  0.0,  0.0,  0.0, 0.0],
                [cur_posture[robot_id][X],  cur_posture[robot_id][Y],  0.0,  0.0,  0.0, 0.0],
                [defense_x, 0.0,  0.0,  0.0,  0.0, 0.0],
                [-field[X]/2,  defense_y,  0.0,  0.0,  0.0, 0.0],
                [delta,  defense_y,  10.0,  10.0,  0.0, 0.0]]

    D1_WHEELS = [[predicted_ball[X],  predicted_ball[Y],  0.0,  0.0,  0.0, 0.0],
                [predicted_ball[X],  predicted_ball[Y],  5.0,  0.0,  0.0, 0.0],
                [predicted_ball[X],  predicted_ball[Y],  10.0,  5.0,  0.0, 0.0],
                [predicted_ball[X]-delta, predicted_ball[Y],  0.0,  0.0,  0.0, 0.0],
                [cur_posture[robot_id][X],  cur_posture[robot_id][Y],  0.0,  0.0,  0.0, 0.0],
                [predicted_ball[X]+delta, predicted_ball[Y],  0.0,  0.0,  0.0, 0.0],
                [predicted_ball[X],  predicted_ball[Y]-delta,  0.0,  0.0,  0.0, 0.0],
                [predicted_ball[X],  predicted_ball[Y]+delta,  10.0,  5.0,  0.0, 0.0]]

    D2_WHEELS = [[predicted_ball[X],  predicted_ball[Y],  0.0,  0.0,  0.0, 0.0],
                [predicted_ball[X],  predicted_ball[Y],  5.0,  0.0,  0.0, 0.0],
                [predicted_ball[X],  predicted_ball[Y],  10.0,  5.0,  0.0, 0.0],
                [predicted_ball[X]-delta, predicted_ball[Y],  0.0,  0.0,  0.0, 0.0],
                [cur_posture[robot_id][X],  cur_posture[robot_id][Y],  0.0,  0.0,  0.0, 0.0],
                [predicted_ball[X]+delta, predicted_ball[Y],  0.0,  0.0,  0.0, 0.0],
                [predicted_ball[X],  predicted_ball[Y]-delta,  0.0,  0.0,  0.0, 0.0],
                [predicted_ball[X],  predicted_ball[Y]+delta,  10.0,  5.0,  0.0, 0.0]]

    F1_WHEELS = [[predicted_ball[X],  predicted_ball[Y],  0.0,  0.0,  0.0, 0.0],
                [predicted_ball[X],  predicted_ball[Y],  5.0,  0.0,  0.0, 0.0],
                [predicted_ball[X],  predicted_ball[Y],  10.0,  5.0,  0.0, 0.0],
                [predicted_ball[X]-delta, predicted_ball[Y],  0.0,  0.0,  0.0, 0.0],
                [cur_posture[robot_id][X],  cur_posture[robot_id][Y],  0.0,  0.0,  0.0, 0.0],
                [predicted_ball[X]+delta, predicted_ball[Y],  0.0,  0.0,  0.0, 0.0],
                [predicted_ball[X], predicted_ball[Y]-delta,  0.0,  0.0,  0.0, 0.0],
                [predicted_ball[X], predicted_ball[Y]+delta,  10.0,  5.0,  0.0, 0.0]]

    F2_WHEELS = [[predicted_ball[X], predicted_ball[Y],  0.0,  0.0,  0.0, 0.0],
                [predicted_ball[X], predicted_ball[Y],  5.0,  0.0,  0.0, 0.0],
                [predicted_ball[X], predicted_ball[Y],  10.0,  5.0,  0.0, 0.0],
                [predicted_ball[X]-delta, predicted_ball[Y],  0.0,  0.0,  0.0, 0.0],
                [cur_posture[robot_id][X],  cur_posture[robot_id][Y],  0.0,  0.0,  0.0, 0.0],
                [predicted_ball[X]+delta, predicted_ball[Y],  0.0,  0.0,  0.0, 0.0],
                [predicted_ball[X], predicted_ball[Y]-delta,  0.0,  0.0,  0.0, 0.0],
                [predicted_ball[X], predicted_ball[Y]+delta,  10.0,  5.0,  0.0, 0.0]]

    speeds = [GK_WHEELS, D1_WHEELS, D2_WHEELS, F1_WHEELS, F2_WHEELS]

    x, y = helper.go_to(robot_id, speeds[robot_id][action_number][0], speeds[robot_id][action_number][1], cur_posture, cur_ball, max_linear_velocity[robot_id])
    return x, y, \
           speeds[robot_id][action_number][2], \
           speeds[robot_id][action_number][3], speeds[robot_id][action_number][4], speeds[robot_id][action_number][5]

class Logger():
    def __init__(self):

        self.episode = []
        self.m_episode = []
        self.value = []
        self.mean_value = []

    def update(self, episode, value, num):

        self.episode.append(episode)
        self.value.append(value)
        self.num = num
        if len(self.value) >= self.num :
            self.m_episode.append(episode - self.num/2)
            self.mean_value.append(np.mean(self.value[-self.num:]))

    def plot(self, name):
        filename = os.path.dirname(__file__) + '/TOTAL_' + str(name) + '.png'
        plt.title(str(name))
        plt.plot(self.episode, self.value, c = 'lightskyblue', label='total_reward') 
        plt.plot(self.m_episode, self.mean_value, c = 'b', label='Average_Total_Reward') 
        if len(self.episode) <= 10:
            plt.legend(loc=1)
        plt.savefig(filename)