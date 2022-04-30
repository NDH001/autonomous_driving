#!/usr/bin/env python
# manual
import time
"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper

# from experiments.utils import save_img

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
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


class PIDController:
    def __init__(self):
        self.k_P = 5
        self.k_I = 0
        self.k_D = 200
        self.integral_value = 0
        self.previous_error = 0
        self.SPEED = 0.22
        self.average = 0
        self.count = 0

    def action_from_output(self, output):
        MAX_Speed = self.SPEED
        MAX_TURN = 1
        output = max(min(output, MAX_TURN), -MAX_TURN)
        perc = abs(output)/MAX_TURN
        action = np.array([self.SPEED, output])
        print("action: ", action)
        return action

    def average_out(self, output):
        self.average = self.average * self.count + abs(output)
        self.average/self.count
        if self.count%10 == 0:
            print("AVERAGE: ", self.average)

    def update(self, dt):
        error = env.compute_reward(env.cur_pos, env.cur_angle, env.robot_speed)
        C_P = self.k_P*error
        derivation = (error - self.previous_error)
        self.integral_value = self.integral_value + error
        print("Error = ", error, " Integral = ", self.integral_value, " Damper = ", derivation, "time = ", dt)
        output = C_P + self.integral_value * self.k_I + derivation * self.k_D
        self.average_out(output)
        action = self.action_from_output(output)
        if key_handler[key.UP]:
            action = np.array([0.22, 0.0])
        if key_handler[key.DOWN]:
            action = np.array([-0.22, 0])
        if key_handler[key.LEFT]:
            action = np.array([0.1, +1])
        if key_handler[key.RIGHT]:
            action = np.array([0.1, -1])
        if key_handler[key.SPACE]:
            action = np.array([0, 0])
        if key_handler[key.S]:
            time.sleep(1)
        # Speed boost
        if key_handler[key.LSHIFT]:
            action *= 1.5
        # print('step_count = %s, distance=%.3f, d_time=%s' % (env.unwrapped.step_count, self.error, dt))
        obs, reward, done, info = env.step(action)

        if key_handler[key.RETURN]:
            from PIL import Image
            im = Image.fromarray(obs)

            im.save('screen.png')
        self.previous_error = error
        if done:
            print('done!')
            env.reset()
            env.render()
            self.integral_value = 0
            self.previous_error = 0

        env.render()


pid = PIDController()


def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    print('step_count = %s, distance=%.3f, d_time=%s' % (env.unwrapped.step_count, reward, dt))

    if key_handler[key.RETURN]:
        from PIL import Image
        im = Image.fromarray(obs)

        im.save('screen.png')

    if done:
        print('done!')
        env.reset()
        env.render()

    env.render()


pyglet.clock.schedule_interval(pid.update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
