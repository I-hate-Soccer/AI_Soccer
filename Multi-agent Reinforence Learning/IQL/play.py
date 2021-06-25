#!/usr/bin/python3

# Author(s): Luiz Felipe Vecchietti, Kyujin Choi, Taeyoung Kim
# Maintainer: Kyujin Choi (nav3549@kaist.ac.kr)

import random
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../common')
try:
    from participant import Participant, Game, Frame
except ImportError as err:
    print('player_deeplearning-multi-iql-dqn: \'participant\' module cannot be imported:', err)
    raise

import sys

import math
import numpy as np

import helper
from dqn import DQN
from rl_utils import  get_action, get_state

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

#robot_index
GK_INDEX = 0 
D1_INDEX = 1 
D2_INDEX = 2 
F1_INDEX = 3 
F2_INDEX = 4

#robot_training_checkpoint
CHECKPOINT_GK= os.path.join(os.path.dirname(__file__), 'models/policy_gk.pt1200000')
CHECKPOINT_D1= os.path.join(os.path.dirname(__file__), 'models/policy_d1.pt1200000')
CHECKPOINT_D2= os.path.join(os.path.dirname(__file__), 'models/policy_d2.pt1200000')
CHECKPOINT_F1= os.path.join(os.path.dirname(__file__), 'models/policy_f1.pt1200000')
CHECKPOINT_F2= os.path.join(os.path.dirname(__file__), 'models/policy_f2.pt1200000')
CHECKPOINT = [CHECKPOINT_GK, CHECKPOINT_D1, CHECKPOINT_D2, CHECKPOINT_F1, CHECKPOINT_F2]

class Frame(object):
    def __init__(self):
        self.time = None
        self.score = None
        self.reset_reason = None
        self.game_state = None
        self.subimages = None
        self.coordinates = None
        self.half_passed = None

class Player(Participant):
    def init(self, info):
        params_file = open(os.path.dirname(__file__) + '/parameters.json')
        params = json.loads(params_file.read())
        self.field = info['field']
        self.max_linear_velocity = info['max_linear_velocity']
        self.goal = info['goal']
        self.number_of_robots = info['number_of_robots']
        self.end_of_frame = False
        self._frame = 0 
        self.speeds = [0 for _ in range(30)]
        self.cur_posture = []
        self.prev_posture = []
        self.cur_posture_opp = []
        self.cur_ball = []
        self.prev_ball = []

        self.previous_frame = Frame()
        self.frame_skip = params['sim_parameters']['frame_skip'] # number of frames to skip
        self.obs_size = 10 # state size
        self.act_size = 13 # number of discrete actions

        # for RL
        self.action = [0 for _ in range(self.number_of_robots)]
        self.previous_action = [0 for _ in range(self.number_of_robots)]
        self.state = [[0 for _ in range(self.obs_size)] for _ in range(self.number_of_robots)]
        self.previous_state = [[0 for _ in range(self.obs_size)] for _ in range(self.number_of_robots)]

        self.num_inputs = self.obs_size
        # RL algorithm class
        self.load = True
        self.play = True
        if ( params['sim_parameters']['algorithm'] == 'iql'):
            self.trainer =[DQN(role, self.obs_size, self.act_size, CHECKPOINT[role], self.load, self.play) for role in range(self.number_of_robots)]

        helper.printConsole("Initializing variables...")

    def get_coord(self, received_frame):
        self.cur_ball = received_frame.coordinates[BALL]
        self.cur_posture = received_frame.coordinates[MY_TEAM]
        self.cur_posture_opp = received_frame.coordinates[OP_TEAM]
        self.prev_posture = self.previous_frame.coordinates[MY_TEAM]
        self.prev_posture_opp = self.previous_frame.coordinates[OP_TEAM]
        self.prev_ball = self.previous_frame.coordinates[BALL]

    def update(self, received_frame):

        if received_frame.end_of_frame:
        
            self._frame += 1

            if (self._frame == 1):
                self.previous_frame = received_frame
                self.get_coord(received_frame)

            self.get_coord(received_frame)

            # Get reward and state
            state = get_state(self.cur_posture, self.prev_posture, self.cur_posture_opp, self.prev_posture_opp, self.cur_ball, self.prev_ball, self.field, self.goal, self.max_linear_velocity, received_frame.reset_reason) 

            # select next action
            self.state = np.reshape([state],(self.number_of_robots, self.obs_size))
            self.action = [self.trainer[role].select_action(np.reshape([self.state[role]],(1, self.obs_size))) for role in range(self.number_of_robots)]

            # Set wheel speeds and send to the simulator
            for role in range(self.number_of_robots):
                self.speeds[6*role : 6*role + 6] = get_action(role, self.action[role], self.max_linear_velocity)

            self.set_speeds(self.speeds)

            self.end_of_frame = False
            self.previous_action = self.action
            self.previous_frame = received_frame

if __name__ == '__main__':
    player = Player()
    player.run()