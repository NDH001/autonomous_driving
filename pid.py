import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='loop_empty')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()


class PIDController:
    def __init__(self):
        self.kp = 5
        self.ki = 0
        self.kd = 200
        self.sum_error = 0
        self.previous_error = 0
        self.SPEED = 0.22
        self.average = 0
        self.count = 0

    # def action_from_output(self, output):
    #     MAX_Speed = self.SPEED
    #     MAX_TURN = 1
    #     output = max(min(output, MAX_TURN), -MAX_TURN)
    #     perc = abs(output)/MAX_TURN
    #     action = np.array([self.SPEED, output])
    #     print("action: ", action)
    #     return action
    #
    # def average_out(self, output):
    #     self.average = self.average * self.count + abs(output)
    #     self.average/self.count
    #     if self.count%10 == 0:
    #         print("AVERAGE: ", self.average)

    def update(self, dt):
        cur_error = env.compute_reward(env.cur_pos, env.cur_angle, env.robot_speed)
        derivation = (cur_error - self.previous_error)
        print("Error = ", cur_error, " Integral = ", self.sum_error, " Damper = ", derivation, "time = ", dt)
        output = self.kp * cur_error + self.sum_error * self.ki + derivation * self.kd
        # self.average_out(output)
        # action = self.action_from_output(output)
        self.sum_error += cur_error
        # print('step_count = %s, distance=%.3f, d_time=%s' % (env.unwrapped.step_count, self.error, dt))
        # obs, reward, done, info = env.step(action)
        self.previous_error = cur_error
        # if done:
        #     print('done!')
        #     env.reset()
        #     env.render()
        #     self.integral_value = 0
        #     self.previous_error = 0

        env.render()
