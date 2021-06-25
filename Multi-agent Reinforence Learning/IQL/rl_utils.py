#!/usr/bin/python3

# Author(s): Luiz Felipe Vecchietti, Kyujin Choi, Taeyoung Kim
# Maintainer: Kyujin Choi (nav3549@kaist.ac.kr)

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../common')
try:
    from participant import Participant, Game, Frame
except ImportError as err:
    print('player_deeplearning-multi-iql-dqn: \'participant\' module cannot be imported:', err)
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

class TouchCounter(object):
    MaxCountMove = 20
    MaxCountGoal = 40

    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.reset()
    
    def reset(self):
        self.touch_count_move = np.zeros(self.num_agents)
        self.touch_count_goal = np.zeros(self.num_agents)
        self.once_touch = [False for _ in range(self.num_agents)]

    def isTouchedMoveCurrent(self):
        return [(self.touch_count_move[i] > 0) for i in range(self.num_agents)]

    def isTouchedGoalCurrent(self):
        return [(self.touch_count_goal[i] > 0) for i in range(self.num_agents)]

    def onceTouched(self):
        return self.once_touch
    
    def ShowTouchedInfo(self):
        helper.printConsole("touch count move: " + str(self.touch_count_move))
        helper.printConsole("touch count goal: " + str(self.touch_count_goal))

    def Counts(self, cur_posture, reset_reason):
        if not reset_reason == None:
            if reset_reason > 0:
                self.reset()

        touch_flag = [cur_posture[i][TOUCH] for i in range(self.num_agents)]

        for i in range(self.num_agents):
            if touch_flag[i]:
                self.touch_count_move[i] = self.MaxCountMove
                self.touch_count_goal[i] = self.MaxCountGoal
                self.once_touch[i] = True
            else:
                if self.touch_count_move[i] > 0:
                    self.touch_count_move[i] -= 1
                if self.touch_count_goal[i] > 0:
                    self.touch_count_goal[i] -= 1

def predict_ball_velocity(cur_ball, prev_ball, ts, reset_reason):
    if reset_reason > 0:
        vx = 0
        vy = 0
    else:
        vx = (cur_ball[X] - prev_ball[X])/ts
        vy = (cur_ball[Y] - prev_ball[Y])/ts
    return [vx, vy]

def predict_robot_velocity(cur_posture, prev_posture, index, ts):
    vx = (cur_posture[index][X] - prev_posture[index][X])/ts
    vy = (cur_posture[index][Y] - prev_posture[index][Y])/ts
    vd = math.atan2(vy, vx)
    vr = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
    return [vd*180/math.pi, vr]

def get_state(cur_posture, prev_posture, cur_posture_opp, prev_posture_opp, cur_ball, prev_ball, field, goal, max_linear_velocity, reset_reason):
    # relative state: (shape: 8)
    states = [[] for _ in range(5)]
    pxx = field[X] + goal[X]
    pyy = field[Y]
    defense_angle = helper.get_defense_kick_angle(cur_ball, field, cur_ball)
    defense_x = math.cos(defense_angle)*0.6 - field[X]/2
    defense_y = math.sin(defense_angle)*0.6
    
    ball_velocity = predict_ball_velocity(cur_ball, prev_ball, 0.05, reset_reason)

    for i in range(5):
        if i == 0:
            states[i] =[round((defense_x - cur_posture[i][X])/pxx, 2), round((defense_y - cur_posture[i][Y])/pyy, 2),  
                        round((cur_ball[X] - field[X]/2)/pxx, 2), round((cur_ball[Y] - 0)/pyy, 2),
                        round((-field[X]/2 - cur_posture[i][X])/pxx, 2), round((0 - cur_posture[i][Y])/pyy, 2),
                        round(cur_posture[i][TH], 2), round((math.atan2(cur_ball[Y]-cur_posture[i][Y], cur_ball[X]-cur_posture[i][X]) - cur_posture[i][TH])/math.pi, 2),
                        ball_velocity[X]/4, ball_velocity[Y]/4]
        elif i == 1:
            states[i] =[round((cur_ball[X] - cur_posture[i][X])/pxx, 2), round((cur_ball[Y] - cur_posture[i][Y])/pyy, 2),  
                        round((cur_ball[X] - field[X]/2)/pxx, 2), round((cur_ball[Y] - 0)/pyy, 2),
                        round((cur_posture[2][X] - cur_posture[i][X])/pxx, 2), round((cur_posture[2][Y] - cur_posture[i][Y])/pyy, 2),
                        round(cur_posture[i][TH], 2), round((math.atan2(cur_ball[Y]-cur_posture[i][Y], cur_ball[X]-cur_posture[i][X]) - cur_posture[i][TH])/math.pi, 2),
                        ball_velocity[X]/4, ball_velocity[Y]/4]
        elif i == 2:
            states[i] =[round((cur_ball[X] - cur_posture[i][X])/pxx, 2), round((cur_ball[Y] - cur_posture[i][Y])/pyy, 2),  
                        round((cur_ball[X] - field[X]/2)/pxx, 2), round((cur_ball[Y] - 0)/pyy, 2),
                        round((cur_posture[1][X] - cur_posture[i][X])/pxx, 2), round((cur_posture[1][Y] - cur_posture[i][Y])/pyy, 2),
                        round(cur_posture[i][TH], 2), round((math.atan2(cur_ball[Y]-cur_posture[i][Y], cur_ball[X]-cur_posture[i][X]) - cur_posture[i][TH])/math.pi, 2),
                        ball_velocity[X]/4, ball_velocity[Y]/4]
        elif i == 3:
            states[i] =[round((cur_ball[X] - cur_posture[i][X])/pxx, 2), round((cur_ball[Y] - cur_posture[i][Y])/pyy, 2),  
                        round((cur_ball[X] - field[X]/2)/pxx, 2), round((cur_ball[Y] - 0)/pyy, 2),
                        round((cur_posture[4][X] - cur_posture[i][X])/pxx, 2), round((cur_posture[4][Y] - cur_posture[i][Y])/pyy, 2),
                        round(cur_posture[i][TH], 2), round((math.atan2(cur_ball[Y]-cur_posture[i][Y], cur_ball[X]-cur_posture[i][X]) - cur_posture[i][TH])/math.pi, 2),
                        ball_velocity[X]/4, ball_velocity[Y]/4]
        elif i == 4:
            states[i] =[round((cur_ball[X] - cur_posture[i][X])/pxx, 2), round((cur_ball[Y] - cur_posture[i][Y])/pyy, 2),  
                        round((cur_ball[X] - field[X]/2)/pxx, 2), round((cur_ball[Y] - 0)/pyy, 2),
                        round((cur_posture[3][X] - cur_posture[i][X])/pxx, 2), round((cur_posture[3][Y] - cur_posture[i][Y])/pyy, 2),
                        round(cur_posture[i][TH], 2), round((math.atan2(cur_ball[Y]-cur_posture[i][Y], cur_ball[X]-cur_posture[i][X]) - cur_posture[i][TH])/math.pi, 2),
                        ball_velocity[X]/4, ball_velocity[Y]/4]

    return states

def get_reward(cur_posture, prev_posture, cur_ball, prev_ball, field, id, touch_counter, reset_reason):
    dist_robot2ball = helper.distance(cur_posture[id][X] , cur_ball[X], cur_posture[id][Y], cur_ball[Y])
    defense_angle = helper.get_defense_kick_angle(cur_ball, field, cur_ball)
    defense_x = math.cos(defense_angle)*0.6 - field[X]/2
    defense_y = math.sin(defense_angle)*0.6
    defense_dis = helper.distance(cur_posture[id][X], defense_x, cur_posture[id][Y], defense_y)
    robot_th_error = abs(math.atan2(cur_ball[Y]-cur_posture[id][Y], cur_ball[X]-cur_posture[id][X]) - cur_posture[id][TH])
    ball_robot_dis = helper.distance(cur_posture[id][X] , cur_ball[X], cur_posture[id][Y], cur_ball[Y])
    ball_robot_dis_prev = helper.distance(prev_posture[id][X] , prev_ball[X], prev_posture[id][Y], prev_ball[Y])
    ball_robot_velocity = (ball_robot_dis_prev - ball_robot_dis)/0.05 

    dist_ball2goal = helper.distance(field[X]/2 , cur_ball[X], 0, cur_ball[Y])
    prev_dist_ball2goal = helper.distance(field[X]/2 , prev_ball[X], 0, prev_ball[Y])
    delta_ball2goal = dist_ball2goal - prev_dist_ball2goal
    
    isTouchedMove = touch_counter.isTouchedMoveCurrent()
    isTouchedGoal = touch_counter.isTouchedGoalCurrent()
    onceTouched = touch_counter.onceTouched()

    dist_gk2defense = (0.5*(math.exp(-1*defense_dis)))
    basic = (0.5*(math.exp(-1*dist_robot2ball)) + 0.5*(math.exp(-1*robot_th_error)) + np.clip(1-math.exp(-1*(ball_robot_velocity)),0,1))

    differential = 0
    if isTouchedMove[id] > 0:
        if delta_ball2goal <= 0:
            differential = differential + ( (-50) * delta_ball2goal )
    
    concede = 0
    score = 0
    if reset_reason == SCORE_OPPONENT:
        concede = -20
    if reset_reason == SCORE_MYTEAM:
        score = 20

    if id == 0:
        return dist_gk2defense + differential + concede
    elif id == 1 or id == 2:
        return basic + differential
    else:
        return basic + differential + score

def get_action(robot_id, action_number, max_linear_velocity):
    # 13 actions: go forwards, go backwards, rotate right, rotate left, stop

    GK_WHEELS = [[1,  1,  0.0,  0.0,  0.0,  0.0],
                [1,  1,  5.0,  0.0,  0.0,  0.0],
                [0,  0,  0.0,  0.0,  1.0,  0.0],
                [0.8,  1,  0.0,  0.0,  0.0,  0.0],
                [1,  0.8,  0.0,  8.0,  0.0,  0.0],
                [-1, -1,  0.0,  0.0,  0.0,  0.0],
                [0.0,  0.0,  0.0,  0.0,  2.0,  0.0],
                [0.1, -0.1,  0.0,  0.0,  0.0,  0.0],
                [-0.1,  0.1,  0.0,  0.0,  0.0,  0.0],
                [0.2, 0.2,  0.0,  0.0,  0.0,  0.0],
                [-0.2,  0.2,  0.0,  0.0,  0.0,  0.0],
                [0.2, -0.2,  0.0,  0.0,  0.0,  0.0],
                [-0.2,  -0.2,  0.0,  0.0,  0.0,  0.0]]

    D1_WHEELS = [[1,  1,  0.0,  0.0,  0.0,  0.0],
                [1,  1,  5.0,  0.0,  0.0,  0.0],
                [1,  1,  10.0,  10.0,  0.0,  0.0],
                [0.8,  1,  0.0,  0.0,  0.0,  0.0],
                [1,  0.8,  0.0,  8.0,  0.0,  0.0],
                [-1, -1,  0.0,  0.0,  0.0,  0.0],
                [0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                [0.1, -0.1,  0.0,  0.0,  0.0,  0.0],
                [-0.1,  0.1,  0.0,  0.0,  0.0,  0.0],
                [0.5, 0.5,  5.0,  0.0,  0.0,  0.0],
                [0.5,  0.5,  10.0,  5.0,  0.0,  0.0],
                [0.2, -0.2,  0.0,  0.0,  0.0,  0.0],
                [-0.2,  0.2,  0.0,  0.0,  0.0,  0.0]]

    D2_WHEELS = [[1,  1,  0.0,  0.0,  0.0,  0.0],
                [1,  1,  5.0,  0.0,  0.0,  0.0],
                [1,  1,  10.0,  10.0,  0.0,  0.0],
                [0.8,  1,  0.0,  0.0,  0.0,  0.0],
                [1,  0.8,  0.0,  8.0,  0.0,  0.0],
                [-1, -1,  0.0,  0.0,  0.0,  0.0],
                [0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                [0.1, -0.1,  0.0,  0.0,  0.0,  0.0],
                [-0.1,  0.1,  0.0,  0.0,  0.0,  0.0],
                [0.5, 0.5,  5.0,  0.0,  0.0,  0.0],
                [0.5,  0.5,  10.0,  5.0,  0.0,  0.0],
                [0.2, -0.2,  0.0,  0.0,  0.0,  0.0],
                [-0.2,  0.2,  0.0,  0.0,  0.0,  0.0]]

    F1_WHEELS = [[1,  1,  0.0,  0.0,  0.0,  0.0],
                [1,  1,  5.0,  0.0,  0.0,  0.0],
                [1,  1,  10.0,  10.0,  0.0,  0.0],
                [0.8,  1,  0.0,  0.0,  0.0,  0.0],
                [1,  0.8,  0.0,  8.0,  0.0,  0.0],
                [-1, -1,  0.0,  0.0,  0.0,  0.0],
                [0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                [0.1, -0.1,  0.0,  0.0,  0.0,  0.0],
                [-0.1,  0.1,  0.0,  0.0,  0.0,  0.0],
                [0.5, 0.5,  5.0,  0.0,  0.0,  0.0],
                [0.5,  0.5,  10.0,  5.0,  0.0,  0.0],
                [0.2, -0.2,  0.0,  0.0,  0.0,  0.0],
                [-0.2,  0.2,  0.0,  0.0,  0.0,  0.0]]

    F2_WHEELS = [[1,  1,  0.0,  0.0,  0.0,  0.0],
                [1,  1,  5.0,  0.0,  0.0,  0.0],
                [1,  1,  10.0,  10.0,  0.0,  0.0],
                [0.8,  1,  0.0,  0.0,  0.0,  0.0],
                [1,  0.8,  0.0,  8.0,  0.0,  0.0],
                [-1, -1,  0.0,  0.0,  0.0,  0.0],
                [0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                [0.1, -0.1,  0.0,  0.0,  0.0,  0.0],
                [-0.1,  0.1,  0.0,  0.0,  0.0,  0.0],
                [0.5, 0.5,  5.0,  0.0,  0.0,  0.0],
                [0.5,  0.5,  10.0,  5.0,  0.0,  0.0],
                [0.2, -0.2,  0.0,  0.0,  0.0,  0.0],
                [-0.2,  0.2,  0.0,  0.0,  0.0,  0.0]]

    speeds = [GK_WHEELS, D1_WHEELS, D2_WHEELS, F1_WHEELS, F2_WHEELS]

    return speeds[robot_id][action_number][0]*max_linear_velocity[robot_id], speeds[robot_id][action_number][1]*max_linear_velocity[robot_id], speeds[robot_id][action_number][2], \
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