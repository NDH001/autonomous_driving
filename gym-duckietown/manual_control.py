#!/usr/bin/env python
# manual

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
import cv2
# from template_matching import doTemplateMatch

# from experiments.utils import save_img


# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', default='map2')
parser.add_argument('--seed', type=int, default=11, help='random seed')
args = parser.parse_args()

env = DuckietownEnv(
    map_name = args.map_name,
    domain_rand = False,
    draw_bbox = False,
    max_steps = args.max_steps,
    seed = args.seed,
)

env.reset()
env.render()


from collections import deque
# for keeping center points of object
buffer_size = 16
pts = deque(maxlen=buffer_size)

# blue HSV
# blueLower = (191,  76.1, 100.0)
# # blueUpper = (98.8, 98.8,98.8)
# blueUpper = (191,  76.1, 100.0)

blueLower = (20, 120, 120)
blueUpper =(30, 255, 255)




def locate_obj_by_color(imgOriginal):



    crop_img = imgOriginal[0:100]

    blurred = cv2.GaussianBlur(crop_img, (11,11), 0)

    # HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # cv2.imshow("HSV IMAGE", hsv)

    # mask for blue
    mask = cv2.inRange(hsv, blueLower, blueUpper)



    # deleting noises which are in area of mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # cv2.imshow("Mask + Erosion + Dilation", mask)

    # contours
    contours,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(contours) > 0:

        # get max contour
        c = max(contours, key=cv2.contourArea)

        # return rectangle
        rect = cv2.minAreaRect(c)
        ((x,y), (width, height), rotation) = rect

        s = f"x {np.round(x)}, y: {np.round(y)}, width: {np.round(width)}, height: {np.round(height)}, rotation: {np.round(rotation)}"
        print(s)

        # box
        box = cv2.boxPoints(rect)
        box = np.int64(box)

        # moment
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # draw contour
        cv2.drawContours(imgOriginal, [box], 0, (0, 255, 255), 2)

        # point in center
        cv2.circle(imgOriginal, center, 5, (255, 0, 255), -1)

        # print inform
        cv2.putText(imgOriginal, s, (25, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)


    # deque
    pts.appendleft(center)
    for i in range(1, len(pts)):

        if pts[i - 1] is None or pts[i] is None: continue

        cv2.line(imgOriginal, pts[i - 1], pts[i], (0, 255, 0), 3)

    cv2.imshow("DETECTED IMAGE", imgOriginal)


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
    image=cv2.cvtColor(obs,cv2.COLOR_BGR2RGB)
    locate_obj_by_color(image)

    # cv2.imshow("HSV IMAGE", im)
    # locate_obj_by_color(obs)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))
    # doTemplateMatch(obs)
    if key_handler[key.RETURN]:
        from PIL import Image
        im = Image.fromarray(obs)

        im.save('screen.png')

    if done:
        print('done!')
        env.reset()
        env.render()

    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
