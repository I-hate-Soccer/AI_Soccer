#!/usr/bin/env python3

# Author(s): Taeyoung Kim, Chansol Hong, Luiz Felipe Vecchietti
# Maintainer: Chansol Hong (cshong@rit.kaist.ac.kr)

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../common')
try:
    from participant import Game, Frame
except ImportError as err:
    print('player_rulebasedA: \'participant\' module cannot be imported:', err)
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

class Goalkeeper:

    def __init__(self, field, goal, penalty_area, goal_area, robot_size, max_linear_velocity):
        self.field = field
        self.goal = goal
        self.penalty_area = penalty_area
        self.goal_area = goal_area
        self.robot_size = robot_size
        self.max_linear_velocity = max_linear_velocity
        self.action = ActionControl(max_linear_velocity)
        self.flag = 0

    def move(self, robot_id, idx, idx_opp, defense_angle, attack_angle, cur_posture, cur_posture_opp, previous_ball, cur_ball, predicted_ball, cross=False, shoot=False, quickpass=False, jump=False, dribble=False):

        # self.action.update_state(cur_posture, cur_ball)
        self.action.update_state_2(cur_posture, cur_ball, cur_posture_opp)

        x = cur_posture_opp[robot_id][X]
        y = cur_posture_opp[robot_id][Y]
        self.flag = 1

        """
        protection_radius = self.goal_area[Y] / 2 - 0.1
        angle = defense_angle
        protection_x = math.cos(angle) * protection_radius - self.field[X] / 2
        protection_y = math.sin(angle) * protection_radius
        
        if helper.ball_is_own_goal(predicted_ball, self.field, self.goal_area):
            x = protection_x
            y = protection_y
            self.flag = 1
        elif helper.ball_is_own_penalty(predicted_ball, self.field, self.penalty_area):
            x = protection_x
            y = protection_y
            self.flag = 2
        elif helper.ball_is_own_field(predicted_ball):
            x = protection_x
            y = protection_y
            self.flag = 3
        elif helper.ball_is_opp_goal(predicted_ball, self.field, self.goal_area):
            x = protection_x
            y = protection_y
            self.flag = 4
        elif helper.ball_is_opp_penalty(predicted_ball, self.field, self.penalty_area):
            x = protection_x
            y = protection_y
            self.flag = 5
        else:
            x = protection_x
            y = protection_y
            self.flag = 6
"""
        left_wheel, right_wheel = self.action.go_to(robot_id, x, y)
        kick_speed, kick_angle = self.action.kick(cross, shoot, quickpass)
        jump_speed = self.action.jump(jump)
        dribble_mode = self.action.dribble(dribble)
        return left_wheel, right_wheel, kick_speed, kick_angle, jump_speed, dribble_mode

class Defender_1:

    def __init__(self, field, goal, penalty_area, goal_area, robot_size, max_linear_velocity):
        self.field = field
        self.goal = goal
        self.penalty_area = penalty_area
        self.goal_area = goal_area
        self.robot_size = robot_size
        self.max_linear_velocity = max_linear_velocity
        self.action = ActionControl(max_linear_velocity)
        self.flag = 0

    def move(self, robot_id, idx, idx_opp, defense_angle, attack_angle, cur_posture, cur_posture_opp, previous_ball, cur_ball, predicted_ball, cross=False, shoot=False, quickpass=False, jump=False, dribble=False):

        # self.action.update_state(cur_posture, cur_ball)
        self.action.update_state_2(cur_posture, cur_ball, cur_posture_opp)

        x = cur_posture_opp[robot_id][X]
        y = cur_posture_opp[robot_id][Y]
        self.flag = 1
        
        left_wheel, right_wheel = self.action.go_to(robot_id, x, y)
        kick_speed, kick_angle = self.action.kick(cross, shoot, quickpass)
        jump_speed = self.action.jump(jump)
        dribble_mode = self.action.dribble(dribble)
        return left_wheel, right_wheel, kick_speed, kick_angle, jump_speed, dribble_mode

class Defender_2:

    def __init__(self, field, goal, penalty_area, goal_area, robot_size, max_linear_velocity):
        self.field = field
        self.goal = goal
        self.penalty_area = penalty_area
        self.goal_area = goal_area
        self.robot_size = robot_size
        self.max_linear_velocity = max_linear_velocity
        self.action = ActionControl(max_linear_velocity)
        self.flag = 0

    def move(self, robot_id, idx, idx_opp, defense_angle, attack_angle, cur_posture, cur_posture_opp, previous_ball, cur_ball, predicted_ball, cross=False, shoot=False, quickpass=False, jump=False, dribble=False):

        # self.action.update_state(cur_posture, cur_ball)
        self.action.update_state_2(cur_posture, cur_ball, cur_posture_opp)
        
        x = cur_posture_opp[robot_id][X]
        y = cur_posture_opp[robot_id][Y]
        self.flag = 1
        
        left_wheel, right_wheel = self.action.go_to(robot_id, x, y)
        kick_speed, kick_angle = self.action.kick(cross, shoot, quickpass)
        jump_speed = self.action.jump(jump)
        dribble_mode = self.action.dribble(dribble)
        return left_wheel, right_wheel, kick_speed, kick_angle, jump_speed, dribble_mode

class Forward_1:

    def __init__(self, field, goal, penalty_area, goal_area, robot_size, max_linear_velocity):
        self.field = field
        self.goal = goal
        self.penalty_area = penalty_area
        self.goal_area = goal_area
        self.robot_size = robot_size
        self.max_linear_velocity = max_linear_velocity
        self.action = ActionControl(max_linear_velocity)
        self.flag = 0

    def move(self, robot_id, idx, idx_opp, defense_angle, attack_angle, cur_posture, cur_posture_opp, previous_ball, cur_ball, predicted_ball,  cross=False, shoot=False, quickpass=False, jump=False, dribble=False):

        # self.action.update_state(cur_posture, cur_ball)
        self.action.update_state_2(cur_posture, cur_ball, cur_posture_opp)

        x = cur_posture_opp[robot_id][X]
        y = cur_posture_opp[robot_id][Y]
        self.flag = 1
        
        left_wheel, right_wheel = self.action.go_to(robot_id, x, y)
        kick_speed, kick_angle = self.action.kick(cross, shoot, quickpass)
        jump_speed = self.action.jump(jump)
        dribble_mode = self.action.dribble(dribble)
        return left_wheel, right_wheel, kick_speed, kick_angle, jump_speed, dribble_mode

class Forward_2:

    def __init__(self, field, goal, penalty_area, goal_area, robot_size, max_linear_velocity):
        self.field = field
        self.goal = goal
        self.penalty_area = penalty_area
        self.goal_area = goal_area
        self.robot_size = robot_size
        self.max_linear_velocity = max_linear_velocity
        self.action = ActionControl(max_linear_velocity)
        self.flag = 0

    def move(self, robot_id, idx, idx_opp, defense_angle, attack_angle, cur_posture, cur_posture_opp, previous_ball, cur_ball, predicted_ball, cross=False, shoot=False, quickpass=False, jump=False, dribble=False):
        
        #self.action.update_state(cur_posture, cur_ball)
        self.action.update_state_2(cur_posture, cur_ball, cur_posture_opp)

        x = cur_posture_opp[robot_id][X]
        y = cur_posture_opp[robot_id][Y]
        self.flag = 1
        
        left_wheel, right_wheel = self.action.go_to(robot_id, x, y)
        kick_speed, kick_angle = self.action.kick(cross, shoot, quickpass)
        jump_speed = self.action.jump(jump)
        dribble_mode = self.action.dribble(dribble)
        return left_wheel, right_wheel, kick_speed, kick_angle, jump_speed, dribble_mode
