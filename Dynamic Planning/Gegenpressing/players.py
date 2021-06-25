#!/usr/bin/env python3

# Author(s): Taeyoung Kim, Chansol Hong, Luiz Felipe Vecchietti
# Maintainer: Chansol Hong (cshong@rit.kaist.ac.kr)

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../common')
try:
    from participant import Game, Frame
except ImportError as err:
    print('Gegenpressing: \'participant\' module cannot be imported:', err)
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
        self.robot_height = 0.421
        self.flag = 0

    def move(self, robot_id, idx, idx_opp, defense_angle, attack_angle, cur_posture, cur_posture_opp,prev_posture, previous_ball, cur_ball, predicted_ball, cross=False, shoot=False, quickpass=False, jump=False, dribble=False):
        
        self.action.update_state(cur_posture, prev_posture, cur_ball, previous_ball)
        self.flag=0
        protection_radius = self.goal_area[Y]/2 - 0.1
        angle = defense_angle
        protection_x = math.cos(angle) * protection_radius - self.field[X]/2
        protection_y = math.sin(angle) * protection_radius
        if helper.ball_is_own_goal(predicted_ball, self.field, self.goal_area):
            # if ball inside goal area: DANGER
            x = protection_x
            y = protection_y
        elif helper.ball_is_own_penalty(predicted_ball, self.field, self.penalty_area):
            # if the ball is behind the goalkeeper
            if (cur_ball[X] < cur_posture[robot_id][X]):
                # if the ball is not blocking the goalkeeper's path
                if (abs(cur_ball[Y] - cur_posture[robot_id][Y]) > 2 * self.robot_size):
                    # try to get ahead of the ball
                    x = cur_ball[X] - self.robot_size
                    y = cur_posture[robot_id][Y]
                else:
                    # just give up and try not to make a suicidal goal
                    left_wheel, right_wheel = self.action.turn_to(robot_id, math.pi /2)
                    kick_speed, kick_angle = self.action.kick(cross, shoot, quickpass)
                    jump_speed = self.action.jump(jump)
                    dribble_mode = self.action.dribble(dribble)
                    return left_wheel, right_wheel, kick_speed, kick_angle, jump_speed, dribble_mode
            # if the ball is ahead of the goalkeeper
            else:
                desired_th = helper.direction_angle(self, robot_id, cur_ball[X], cur_ball[Y], cur_posture)
                rad_diff = helper.wrap_to_pi(desired_th - cur_posture[robot_id][TH])
                # if the robot direction is too away from the ball direction
                if (rad_diff > math.pi / 3):
                    # give up kicking the ball and block the goalpost
                    x = -self.field[X] / 2 + self.robot_size / 2 + 0.05
                    y = max(min(cur_ball[Y], (self.goal[Y] / 2 - self.robot_size / 2)), -self.goal[Y] / 2 + self.robot_size / 2)
                else:
                    if cur_ball[Z] > 0.2:
                        # try to jump in the ball direction
                        x = cur_ball[X]
                        y = cur_ball[Y]
                        jump = True                        
                    else:
                        # try to kick the ball away from the goal
                        x = cur_ball[X]
                        y = cur_ball[Y]
                        
                        #슛을 F2에게로
                        try:
                            self.action.shoot_to(robot_id, cur_posture[5][X], cur_posture[5][Y])
                            self.flag=1
                            
                        except:
                            shoot = True
                            self.flag=2
                        
        else:
            # if ball outside penalty area, protect against kicks
            x = protection_x
            y = protection_y
            
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

    def move(self, robot_id, idx, idx_opp, defense_angle, attack_angle, cur_posture, cur_posture_opp, prev_posture, previous_ball, cur_ball, predicted_ball, cross=False, shoot=False, quickpass=False, jump=False, dribble=False):
        
        self.action.update_state(cur_posture, prev_posture, cur_ball, previous_ball)
        
        # 공격수 1 마크 
        x=cur_posture_opp[3][X]
        y=cur_posture_opp[3][Y]
        self.flag=1
        
        
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

    def move(self, robot_id, idx, idx_opp, defense_angle, attack_angle, cur_posture, cur_posture_opp, prev_posture, previous_ball, cur_ball, predicted_ball, cross=False, shoot=False, quickpass=False, jump=False, dribble=False):
        
        self.action.update_state(cur_posture, prev_posture, cur_ball, previous_ball)
        
        #공격수 2 마크
        x=cur_posture_opp[4][X]
        y=cur_posture_opp[4][Y]
        self.flag=1
        
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

    def move(self, robot_id, idx, idx_opp, defense_angle, attack_angle, cur_posture, cur_posture_opp, prev_posture, previous_ball, cur_ball, predicted_ball,  cross=False, shoot=False, quickpass=False, jump=False, dribble=False):
        
        self.action.update_state(cur_posture, prev_posture, cur_ball, previous_ball)
        
        x=cur_posture_opp[2][X]
        y=cur_posture_opp[2][Y]
        self.flag=1
        

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

    def move(self, robot_id, idx, idx_opp, defense_angle, attack_angle, cur_posture, cur_posture_opp, prev_posture, previous_ball, cur_ball, predicted_ball, cross=False, shoot=False, quickpass=False, jump=False, dribble=False):
        
        self.action.update_state(cur_posture, prev_posture, cur_ball, previous_ball)
        
        ## 회전하기
        dx = cur_ball[X] - previous_ball[X]
        dy = cur_ball[Y] - previous_ball[Y]+0.00000001 #zero division
        pred_x = predicted_ball[X]
        steps = (cur_posture[robot_id][Y] - cur_ball[Y]) / dy
        
        #로봇 앞에 공 위치할 때
        if (pred_x > cur_posture[robot_id][X]):
            pred_dist = pred_x - cur_posture[robot_id][X]
            
            # 예측된 공의 위치가 가까우면, 소유할 경우만 (가져도 회전하느라 공이 이동해버린다....)
            if (pred_dist > 0.1 and pred_dist < 0.3 and steps < 10) and  cur_posture[robot_id][BALL_POSSESSION]:
                
                #상대팀 골대 방향 찾고 그쪽으로 회전
                goal_angle = helper.direction_angle(self, robot_id, self.field[X] / 2, 0, cur_posture)
                left_wheel, right_wheel = self.action.turn_to(robot_id, goal_angle)
                kick_speed, kick_angle = self.action.kick(cross, shoot, quickpass)
                jump_speed = self.action.jump(jump)
                dribble_mode = self.action.dribble(dribble)
                return left_wheel, right_wheel, kick_speed, kick_angle, jump_speed, dribble_mode


        ## 공이 내 진영 골 영역 or 패널티 영역에 있을 때
        if helper.ball_is_own_goal(predicted_ball, self.field, self.goal_area) or helper.ball_is_own_penalty(predicted_ball, self.field, self.penalty_area):
            x = cur_ball[X]
            y = cur_ball[Y]
            self.flag = 1
            
            #근데 골 점유한 경우
            if cur_posture[robot_id][BALL_POSSESSION]:
                self.flag = 2
                #크로스
                #a=self.action.cross_to(self,robot_id,self.field[X]/2,0,cur_posture)
                a=self.action.cross_to(robot_id, self.field[X]/2,0, 0.421) #0.421: robot_height
                #크로스 실패시
                if a==None:
                    shoot=True
                    self.flag = 3
            

        ## 현재 위치에서 shoot chance가 있을 때
        elif (helper.shoot_chance(self, robot_id, cur_posture, cur_ball)):
            x = predicted_ball[X]
            y = predicted_ball[Y]
            self.flag = 4
            
            if cur_posture[robot_id][BALL_POSSESSION]:
                try:
                    b=self.action.shoot_to(robot_id,self.field[X]/2,0)
                    self.flag = 5
                except:
                    shoot = True
                    self.flag = 6

                
        else:
            if (cur_ball[X] > -0.3 * self.field[X] / 2):
                self.flag = 7
                
                # if the robot can push the ball toward opponent's side, do it
                if (cur_posture[robot_id][X] < cur_ball[X] - 0.05):
                    x = cur_ball[X]
                    y = cur_ball[Y]
                    self.flag = 8
                    
                    if cur_posture[robot_id][BALL_POSSESSION]:
                        self.flag = 9
                        shoot = True
                else:
                    # otherwise go behind the ball
                    if (abs(cur_ball[Y] - cur_posture[robot_id][Y]) > 0.3):
                        x = cur_ball[X] - 0.2
                        y = cur_ball[Y]
                        self.flag = 10
                        
                    else:
                        x = cur_ball[X] - 0.2
                        y = cur_posture[robot_id][Y]
                        self.flag = 11
                        
            else:
                #x = -0.1 * self.field[X] / 2

                try: #조금씩 이동+패스
                    goalx=self.field[X]/2
                    goaly=0
                    robotx=cur_posture[robot_id][X]
                    roboty=cur_posture[robot_id][Y]
                    
                    dx=(robotx+goalx)/6
                    dy=(roboty+goaly)/6 
                    
                    self.action.pass_to(robot_id,dx,dy)
                    #print(dx)
                    #print(dy)
                    
                    x=dx
                    y=dy
                    
                    self.flag = 12
                    
                except:
                    x=cur_ball[X]
                    y = cur_ball[Y]
                    self.flag = 13
                    

        '''
        if helper.ball_is_own_goal(predicted_ball, self.field, self.goal_area):
            x = -0.5
            y = -1
            self.flag = 1
        elif helper.ball_is_own_penalty(predicted_ball, self.field, self.penalty_area):
            x = -0.5
            y = -1
            self.flag = 2
        elif helper.ball_is_own_field(predicted_ball):
            x = cur_ball[X]
            y = cur_ball[Y]
            self.flag = 3
        elif helper.ball_is_opp_goal(predicted_ball, self.field, self.goal_area):
            x = cur_ball[X]
            y = cur_ball[Y]
            self.flag = 4
            if cur_posture[robot_id][BALL_POSSESSION]:
                quickpass = True
                self.flag = 5
        elif helper.ball_is_opp_penalty(predicted_ball, self.field, self.penalty_area):
            x = cur_ball[X]
            y = cur_ball[Y]
            self.flag = 6
            if cur_posture[robot_id][BALL_POSSESSION]:
                cross = True
                self.flag = 7
        else:
            x = cur_ball[X]
            y = cur_ball[Y]
            self.flag = 8
            if cur_posture[robot_id][BALL_POSSESSION]:
                shoot = True
                self.flag = 9
        '''
        
        '''
        # if the ball is coming toward the robot, seek for shoot chance
        if (robot_id == idx and helper.ball_coming_toward_robot(robot_id, cur_posture, previous_ball, cur_ball)):
            dx = cur_ball[X] - previous_ball[X]
            dy = cur_ball[Y] - previous_ball[Y]
            pred_x = predicted_ball[X]
            steps = (cur_posture[robot_id][Y] - cur_ball[Y]) / dy
            # if the ball will be located in front of the robot
            if (pred_x > cur_posture[robot_id][X]):
                pred_dist = pred_x - cur_posture[robot_id][X]
                # if the predicted ball location is close enough
                if (pred_dist > 0.1 and pred_dist < 0.3 and steps < 10):
                    # find the direction towards the opponent goal and look toward it
                    goal_angle = helper.direction_angle(self, robot_id, self.field[X] / 2, 0, cur_posture)
                    left_wheel, right_wheel = self.action.turn_to(robot_id, goal_angle)
                    kick_speed, kick_angle = self.action.kick(cross, shoot, quickpass)
                    jump_speed = self.action.jump(jump)
                    dribble_mode = self.action.dribble(dribble)
                    return left_wheel, right_wheel, kick_speed, kick_angle, jump_speed, dribble_mode
                
        # if the robot can shoot from current position
        if (robot_id == idx and helper.shoot_chance(self, robot_id, cur_posture, cur_ball)):
            x = predicted_ball[X]
            y = predicted_ball[Y]
            if cur_posture[robot_id][BALL_POSSESSION]:
                shoot = True
                
        # if this forward is closer to the ball
        elif (robot_id == idx):
            if (cur_ball[X] > -0.3 * self.field[X] / 2):
                # if the robot can push the ball toward opponent's side, do it
                if (cur_posture[robot_id][X] < cur_ball[X] - 0.05):
                    x = cur_ball[X]
                    y = cur_ball[Y]
                    if cur_posture[robot_id][BALL_POSSESSION]:
                        shoot = True
                else:
                    # otherwise go behind the ball
                    if (abs(cur_ball[Y] - cur_posture[robot_id][Y]) > 0.3):
                        x = cur_ball[X] - 0.2
                        y = cur_ball[Y]
                    else:
                        x = cur_ball[X] - 0.2
                        y = cur_posture[robot_id][Y]
            else:
                x = -0.1 * self.field[X] / 2
                y = cur_ball[Y]
        # if this forward is not closer to the ball
        else:
            if (cur_ball[X] > -0.3 * self.field[X] / 2):
                x = cur_ball[X] - 0.25
                y = cur_ball[Y] - 0.5
            else:
                # ball is on right side
                if (cur_ball[Y] < 0):
                    x = -0.1 * self.field[X] / 2
                    y = min(cur_ball[Y] - 0.5, self.field[Y] / 2 - self.robot_size / 2)
                # ball is on left side
                else:
                    x = -0.1 * self.field[X] / 2
                    y = max(cur_ball[Y] - 0.5, -self.field[Y] / 2 + self.robot_size / 2)
        '''
        
        left_wheel, right_wheel = self.action.go_to(robot_id, x, y)
        kick_speed, kick_angle = self.action.kick(cross, shoot, quickpass)
        jump_speed = self.action.jump(jump)
        dribble_mode = self.action.dribble(dribble)
        return left_wheel, right_wheel, kick_speed, kick_angle, jump_speed, dribble_mode
