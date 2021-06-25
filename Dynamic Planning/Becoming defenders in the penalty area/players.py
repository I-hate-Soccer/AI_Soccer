#!/usr/bin/env python3

# Author(s): Taeyoung Kim, Chansol Hong, Luiz Felipe Vecchietti
# Maintainer: Chansol Hong (cshong@rit.kaist.ac.kr)

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../common')
try:
    from participant import Game, Frame
except ImportError as err:
    print('player_rulebasedH: \'participant\' module cannot be imported:', err)
    raise

import math

import helper
from action import ActionControl

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

class TeamK:

    def __init__(self, field, goal, penalty_area, goal_area, robot_size, max_linear_velocity):
        self.field = field
        self.goal = goal
        self.penalty_area = penalty_area
        self.goal_area = goal_area
        self.robot_size = robot_size
        self.max_linear_velocity = max_linear_velocity
        self.gk_index = 0
        self.d1_index = 1
        self.d2_index = 2
        self.f1_index = 3
        self.f2_index = 4
        self.pass_count = 0
        self.cross_count = 0
        self.shoot_count = 0
        self.gk_target_robot_id = self.gk_index
        self.d1_target_robot_id = self.d1_index
        self.d2_target_robot_id = self.d2_index
        self.f1_target_robot_id = self.f1_index
        self.f2_target_robot_id = self.f2_index
        self.action = ActionControl(max_linear_velocity)
        self.robot_height = 0.421
        self.flag = 0

    def reset_counter(self):
        self.pass_count = 0
        self.cross_count = 0
        self.shoot_count = 0        

    def move(self, idx, idx_opp, defense_angle, attack_angle, cur_posture, cur_posture_opp, 
            prev_posture, prev_posture_opp, prev_ball, cur_ball, predicted_ball, reset_reason, game_state):
        
        self.action.update_state(cur_posture, prev_posture, cur_ball, prev_ball, reset_reason)
        
        # GK variables
        gk_protection_radius = self.goal_area[Y]/2 - 0.1
        gk_protection_x = math.cos(defense_angle) * gk_protection_radius - self.field[X]/2
        gk_protection_y = math.sin(defense_angle) * gk_protection_radius
        # D1 variables
        d1_protection_radius = 1.7
        d1_protection_x = math.cos(defense_angle) * d1_protection_radius - self.field[X]/2
        d1_protection_y = math.sin(defense_angle) * d1_protection_radius

        if game_state == STATE_DEFAULT:

            # GK ZONE
            if helper.ball_is_gk_zone(predicted_ball, self.field, self.goal_area):
                # GK
                gk_control = self.action.defend_ball(self.gk_index)
                if gk_control == None:
                    if (cur_posture[self.gk_index][BALL_POSSESSION]):
                        gk_control = self.action.shoot_to(self.gk_index, 0, 0, 10, 10)
                    else:
                        if -self.field[X]/2 - 0.05 < cur_posture[self.gk_index][X] < -self.field[X]/2 + 0.15 and -0.02 < cur_posture[self.gk_index][Y] < 0.02:
                            gk_control = self.action.turn_to(self.gk_index, 0, 0)
                        else:
                            gk_control = self.action.go_to(self.gk_index, -self.field[X]/2, 0)
                # D1
                d1_control = self.action.go_to(self.d1_index, -5.0, self.penalty_area[Y]/2)
                # D2
                d2_control = self.action.go_to(self.d2_index, -5.0, -self.penalty_area[Y]/2)
                # F1
                f1_control = self.action.go_to(self.f1_index, -3.9, 1)
                # F2
                f2_control = self.action.go_to(self.f2_index, -3.9, -1)
                self.flag = 1
            # D1 ZONE
            elif helper.ball_is_d1_zone(predicted_ball, self.field, self.penalty_area):
                # GK
                gk_control = self.action.defend_ball(self.gk_index)
                if gk_control == None:
                    if (cur_posture[self.gk_index][BALL_POSSESSION]):
                        gk_control = self.action.shoot_to(self.gk_index, 0, 0, 10, 10)
                    else:
                        if -self.field[X]/2 - 0.05 < cur_posture[self.gk_index][X] < -self.field[X]/2 + 0.15 and -0.02 < cur_posture[self.gk_index][Y] < 0.02:
                            gk_control = self.action.turn_to(self.gk_index, 0, 0)
                        else:
                            gk_control = self.action.go_to(self.gk_index, -self.field[X]/2, 0)
                # D1
                if (self.d1_index == idx):
                    if (cur_posture[self.d1_index][BALL_POSSESSION]):
                        if self.pass_count == 0:
                            self.d1_target_robot_id = self.d2_index if (cur_posture[self.d1_index][TH] > 0) else self.f1_index
                            self.pass_count += 1
                        else:
                            self.pass_count += 1
                        d1_control = self.action.shoot_to(self.d1_index, 0, 0, 10, 10)
                        #d1_control = self.action.defend_ball(self.d1_index)
                        #d1_control = self.action.turn_to(self.d1_index, 0, 0)
                    else:
                        if (cur_posture[self.d1_index][X] > predicted_ball[X]):
                            d1_control = self.action.turn_to(self.d1_index, 0, 0)
                        else:
                            #d1_min_x = -self.field[X]/2 + self.goal_area[X] + 0.1
                            d1_control = self.action.go_to(self.d1_index, -5.0, self.penalty_area[Y]/2)
                else:
                    d1_control = self.action.go_to(self.d1_index, -5.0, self.penalty_area[Y]/2)
                # D2
                d2_control =  self.action.go_to(self.d2_index, -5.0, -self.penalty_area[Y]/2)
                # F1
                f1_control =  self.action.go_to(self.f1_index, -3.9, 1)
                # F2
                f2_control = self.action.go_to(self.f2_index, -3.9, -1)
                self.flag = 2
            # D2 ZONE
            elif helper.ball_is_d2_zone(predicted_ball, self.field):
                # GK
                gk_control = self.action.defend_ball(self.gk_index)
                if gk_control == None:
                    if -self.field[X]/2 - 0.05 < cur_posture[self.gk_index][X] < -self.field[X]/2 + 0.15 and -0.02 < cur_posture[self.gk_index][Y] < 0.02:
                        gk_control = self.action.turn_to(self.gk_index, 0, 0)
                    else:
                        gk_control = self.action.go_to(self.gk_index, -self.field[X]/2, 0)
                # D1
                d1_control = self.action.turn_to(self.d1_index, 0, 0)
                # D2
                if (cur_posture[self.d2_index][BALL_POSSESSION]):
                    if self.pass_count == 0:
                       self.d2_target_robot_id = self.f2_index if (cur_posture[self.d1_index][TH] > 0) else self.f1_index
                       self.pass_count += 1
                    else:
                        self.pass_count += 1
                    d2_control = self.action.shoot_to(self.d2_index, 0, 0, 10, 10)
                    #d2_control = self.action.defend_ball(self.d2_index)
                    #d2_control = self.action.turn_to(self.d2_index, 0, 0)
                else:
                    if (cur_posture[self.d2_index][X] > predicted_ball[X]):
                        d2_control = self.action.turn_to(self.d2_index, 0, 0)
                    else:
                        # d2_min_x = -self.field[X]/2 + self.goal_area[X] + 0.1
                        d2_control = self.action.go_to(self.d2_index, -5.0, -self.penalty_area[Y]/2)                      
                # F1
                f1_control = self.action.go_to(self.f1_index, -3.9, 1)
                # F2
                if cur_ball[X] < 0:
                    f2_control = self.action.go_to(self.f2_index, -3.9, -1)
                else:
                    f2_control = self.action.go_to(self.f2_index, -3.9, -1)
                self.flag = 3 #3
            # F1 ZONE
            elif helper.ball_is_f1_zone(predicted_ball, self.field):
                # GK
                gk_control = self.action.defend_ball(self.gk_index)
                if gk_control == None:
                    if -self.field[X]/2 - 0.05 < cur_posture[self.gk_index][X] < -self.field[X]/2 + 0.15 and -0.02 < cur_posture[self.gk_index][Y] < 0.02:
                        gk_control = self.action.turn_to(self.gk_index, 0, 0)
                    else:
                        gk_control = self.action.go_to(self.gk_index, -self.field[X]/2, 0)
                # D1
                d1_control = self.action.go_to(self.d1_index, -5.0, self.penalty_area[Y]/2)
                # D2
                d2_control =  self.action.go_to(self.d2_index, -5.0, -self.penalty_area[Y]/2)
                # F1
                if (cur_posture[self.f1_index][BALL_POSSESSION]):
                    if self.cross_count == 0:
                        self.f1_target_robot_id = self.f2_index
                        self.cross_count += 1
                    else:
                        self.cross_count += 1
                    f1_control = self.action.shoot_to(self.f1_index, 0, 0, 10, 10)
                    if f1_control == None:
                        self.f1_target_robot_id = self.d2_index
                        #f1_control = self.action.defend_ball(self.f1_index)
                        #f1_control = self.action.turn_to(self.f1_index, 0, 0)
                        f1_control = self.action.pass_to(self.f1_index, cur_posture[self.f1_target_robot_id][X], cur_posture[self.f1_target_robot_id][Y])
                else:
                    if (cur_posture[self.f1_index][X] > predicted_ball[X]):
                        f1_control = self.action.go_to(self.f1_index, -3.9, 1)
                    else:
                        #f1_min_x = -self.field[X]/2 + self.goal_area[X] + 0.1
                        f1_control = self.action.go_to(self.f1_index, -3.9, 1)
                # F2     
                if cur_ball[X] < 0:
                    f2_control = self.action.go_to(self.f2_index, -3.9, -1)
                else:
                    f2_control = self.action.go_to(self.f2_index, -3.9, -1)
                self.flag = 4
            # F2 ZONE
            else:
                # GK
                gk_control = self.action.defend_ball(self.gk_index)
                if gk_control == None:
                    if -self.field[X]/2 - 0.05 < cur_posture[self.gk_index][X] < -self.field[X]/2 + 0.15 and -0.02 < cur_posture[self.gk_index][Y] < 0.02:
                        gk_control = self.action.turn_to(self.gk_index, 0, 0)
                    else:
                        gk_control = self.action.go_to(self.gk_index, -self.field[X]/2, 0)
                # D1
                d1_control = self.action.go_to(self.d1_index, -5.0, self.penalty_area[Y]/2)
                # D2
                d2_control = self.action.go_to(self.d2_index, -5.0, -self.penalty_area[Y]/2)
                # F1
                f1_control = self.action.go_to(self.f1_index, -3.9, 1)
                # F2
                if (cur_posture[self.f2_index][BALL_POSSESSION]):
                        self.shoot_count += 1
                        f2_control = self.action.shoot_to(self.f2_index, 0, 0, 10, 10)
                else:
                    f2_control = self.action.go_to(self.f2_index, -3.9, -1)
                self.flag = 6

            if (self.pass_count > 20 or self.cross_count > 20 or self.shoot_count > 20):
                self.reset_counter()

        elif game_state == STATE_GOALKICK:
            print('GOALKICK\n')
            if helper.distance(cur_ball[X], cur_posture[self.gk_index][X], cur_ball[Y], cur_posture[self.gk_index][Y]) <= 0.2:
                gk_control = [0,0,10,8,0,0]
            else:
                gk_control = self.action.go_to(self.gk_index, cur_ball[X], cur_ball[Y])
            d1_control = [0, 0, 0, 0, 0, 0]
            d2_control = [0, 0, 0, 0, 0, 0]
            f1_control = [0, 0, 0, 0, 0, 0]
            f2_control = [0, 0, 0, 0, 0, 0]
        elif game_state == STATE_CORNERKICK:
            print('CORNERKICK\n')
            gk_control = [0, 0, 0, 0, 0, 0]
            d1_control = [0, 0, 0, 0, 0, 0]
            d2_control = [0, 0, 0, 0, 0, 0]
            f1_control = [0, 0, 0, 0, 0, 0]
            if helper.distance(cur_ball[X], cur_posture[self.f2_index][X], cur_ball[Y], cur_posture[self.f2_index][Y]) <= 0.2:
                f2_control = [0,0,7,5,0,0]
            else:
                f2_control = self.action.go_to(self.f2_index, cur_ball[X], cur_ball[Y])
        elif game_state == STATE_KICKOFF:
            print('KICKOFF\n')
            gk_control = [0, 0, 0, 0, 0, 0]
            d1_control = [0, 0, 0, 0, 0, 0]
            d2_control = [0, 0, 0, 0, 0, 0]
            f1_control = [0, 0, 0, 0, 0, 0]
            if helper.distance(cur_ball[X], cur_posture[self.f2_index][X], cur_ball[Y], cur_posture[self.f2_index][Y]) <= 0.2:
                f2_control = [0,0,5,0,0,0]
            else:
                f2_control = self.action.go_to(self.f2_index, cur_ball[X], cur_ball[Y])
        elif game_state == STATE_PENALTYKICK:
            print('PENALTYKICK\n')
            gk_control = [0, 0, 0, 0, 0, 0]
            d1_control = [0, 0, 0, 0, 0, 0]
            d2_control = [0, 0, 0, 0, 0, 0]
            f1_control = [0, 0, 0, 0, 0, 0]
            if helper.distance(cur_ball[X], cur_posture[self.f2_index][X], cur_ball[Y], cur_posture[self.f2_index][Y]) <= 0.3:
                f2_control = [1.5,2.5,0,0,0,0]
            else:
                f2_control = self.action.go_to(self.f2_index, cur_ball[X], cur_ball[Y])
        else:
            gk_control = [0, 0, 0, 0, 0, 0]
            d1_control = [0, 0, 0, 0, 0, 0]
            d2_control = [0, 0, 0, 0, 0, 0]
            f1_control = [0, 0, 0, 0, 0, 0]
            f2_control = [0, 0, 0, 0, 0, 0]

        return gk_control + d1_control + d2_control + f1_control + f2_control