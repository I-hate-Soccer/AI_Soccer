#!/usr/bin/env python3

# Author(s): Taeyoung Kim, Chansol Hong, Luiz Felipe Vecchietti
# Maintainer: Chansol Hong (cshong@rit.kaist.ac.kr)

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../common')
try:
    from participant import Participant, Game, Frame
except ImportError as err:
    print('player_follow-ball: \'participant\' module cannot be imported:', err)
    raise

import math
import numpy as np

import helper

from players import Goalkeeper, Defender_1, Defender_2, Forward_1, Forward_2

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
        self.field = info['field']
        self.max_linear_velocity = info['max_linear_velocity']
        self.robot_size = info['robot_size'][0]
        self.goal = info['goal']
        self.penalty_area = info['penalty_area']
        self.goal_area = info['goal_area']
        self.number_of_robots = info['number_of_robots']
        self.end_of_frame = False
        self._frame = 0 
        self.speeds = [0 for _ in range(30)]
        self.cur_posture = []
        self.cur_posture_opp = []
        self.cur_ball = []
        self.previous_ball = []
        self.predicted_ball = []
        self.idx = 0
        self.idx_opp = 0
        self.previous_frame = Frame()
        self.defense_angle = 0
        self.attack_angle = 0
        self.gk_index = 0
        self.d1_index = 1
        self.d2_index = 2
        self.f1_index = 3
        self.f2_index = 4
        self.GK = Goalkeeper(self.field, self.goal, self.penalty_area, 
                                self.goal_area, self.robot_size,
                                self.max_linear_velocity)
        self.D1 = Defender_1(self.field, self.goal, self.penalty_area, 
                                self.goal_area, self.robot_size,
                                self.max_linear_velocity)
        self.D2 = Defender_2(self.field, self.goal, self.penalty_area, 
                                self.goal_area, self.robot_size,
                                self.max_linear_velocity)
        self.F1 = Forward_1(self.field, self.goal, self.penalty_area, 
                                self.goal_area, self.robot_size,
                                self.max_linear_velocity)
        self.F2 = Forward_2(self.field, self.goal, self.penalty_area,
                                self.goal_area, self.robot_size,
                                self.max_linear_velocity)
        helper.printConsole("Initializing variables...")

    def get_coord(self, received_frame):
        self.cur_ball = received_frame.coordinates[BALL]
        self.cur_posture = received_frame.coordinates[MY_TEAM]
        self.cur_posture_opp = received_frame.coordinates[OP_TEAM]

    def update(self, received_frame):

        if (received_frame.end_of_frame):
	    
            self._frame += 1

            if (self._frame == 1):
                self.previous_frame = received_frame
                self.get_coord(received_frame)
                self.previous_ball = self.cur_ball

            self.get_coord(received_frame)
            self.predicted_ball = helper.predict_ball(self.cur_ball, self.previous_ball)
            self.idx = helper.find_closest_robot(self.cur_ball, self.cur_posture, self.number_of_robots)
            self.idx_opp = helper.find_closest_robot(self.cur_ball, self.cur_posture_opp, self.number_of_robots)
            self.defense_angle = helper.get_defense_kick_angle(self.predicted_ball, self.field, self.cur_ball)
            self.attack_angle = helper.get_attack_kick_angle(self.predicted_ball, self.field)

##############################################################################
            #(update the robots wheels)
            # Robot Functions
            self.speeds[6 * self.gk_index : 6 * self.gk_index + 6] = self.GK.move(self.gk_index, 
                                                                                self.idx, self.idx_opp, 
                                                                                self.defense_angle, self.attack_angle,
                                                                                self.cur_posture, self.cur_posture_opp,
                                                                                self.previous_ball, self.cur_ball, self.predicted_ball)
            self.speeds[6 * self.d1_index : 6 * self.d1_index + 6] = self.D1.move(self.d1_index, 
                                                                                self.idx, self.idx_opp, 
                                                                                self.defense_angle, self.attack_angle,
                                                                                self.cur_posture, self.cur_posture_opp,
                                                                                self.previous_ball, self.cur_ball, self.predicted_ball)
            self.speeds[6 * self.d2_index : 6 * self.d2_index + 6] = self.D2.move(self.d2_index, 
                                                                                self.idx, self.idx_opp, 
                                                                                self.defense_angle, self.attack_angle,
                                                                                self.cur_posture, self.cur_posture_opp,
                                                                                self.previous_ball, self.cur_ball, self.predicted_ball)
            self.speeds[6 * self.f1_index : 6 * self.f1_index + 6] = self.F1.move(self.f1_index, 
                                                                                self.idx, self.idx_opp, 
                                                                                self.defense_angle, self.attack_angle,
                                                                                self.cur_posture, self.cur_posture_opp,
                                                                                self.previous_ball, self.cur_ball, self.predicted_ball)
            self.speeds[6 * self.f2_index : 6 * self.f2_index + 6] = self.F2.move(self.f2_index, 
                                                                                self.idx, self.idx_opp, 
                                                                                self.defense_angle, self.attack_angle,
                                                                                self.cur_posture, self.cur_posture_opp,
                                                                                self.previous_ball, self.cur_ball, self.predicted_ball)
            
            self.set_speeds(self.speeds)
##############################################################################

            self.previous_frame = received_frame
            self.previous_ball = self.cur_ball

            helper.print_debug_flag(self)

if __name__ == '__main__':
    player = Player()
    player.run()