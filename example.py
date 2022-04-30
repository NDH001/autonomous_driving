import argparse
import math
import time

import cv2
import numpy as np
from gym_duckietown.envs import DuckietownEnv

from collections import deque

# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', '-m', default="map4_0", type=str)
parser.add_argument('--seed', '-s', default=2, type=int)
parser.add_argument('--start-tile', '-st', default="1,13", type=str, help="two numbers separated by a comma")
parser.add_argument('--goal-tile', '-gt', default="3,3", type=str, help="two numbers separated by a comma")
args = parser.parse_args()

env = DuckietownEnv(
    domain_rand=False,
    max_steps=1500,
    map_name=args.map_name,
    seed=args.seed,
    user_tile_start=args.start_tile,
    goal_tile=args.goal_tile,
    randomize_maps_on_reset=False
)

env.cam_height = 0.2
env.render()

map_img, goal, start_pos = env.get_task_info()
print("start tile:", start_pos, " goal tile:", goal)

print(f"map: {map_img.shape}")

buffer_size = 16
pts = deque(maxlen=buffer_size)

blueLower = (175, 50, 20)
blueUpper = (180, 255, 255)


# Show the map image
# White pixels are drivable and black pixels are not.
# Blue pixels indicate lan center
# Each tile has size 100 x 100 pixels
# Tile (0, 0) locates at left top corner.
# map_img = cv2.resize(map_img, None, fx=0.5, fy=0.5)
# cv2.imshow("map", map_img)
#
# cv2.waitKey(200)

# please remove this line for your own policy
# actions = np.loadtxt('./map4_0_seed2_start_1,13_goal_3,3.txt', delimiter=',')
#

#
# # dump the controls using numpy
# np.savetxt(f'./{args.map_name}_seed{args.seed}_start_{start_pos[0]},{start_pos[1]}_goal_{goal[0]},{goal[1]}.txt',
#            actions, delimiter=',')
#
def updateV2(map_img, x, y):
    color2 = [120]
    temp_img = map_img
    temp_img[int(y * 100):int(y * 100) + 20, int(x * 100): int(x * 100) + 20] = color2
    temp_img = cv2.convertScaleAbs(temp_img, alpha=(255.0))
    # map_img[900:1000, 500:600] = color3
    temp_img = cv2.resize(map_img, (500, 500))

    while (1):

        cv2.imshow("map", temp_img)

        k = cv2.waitKey(33)
        if k == 27:  # Esc key to stop
            break


modi_map = np.zeros((map_img.shape[0], map_img.shape[1]))
black = []
white = []
for i in range(map_img.shape[0]):
    for w in range(map_img.shape[1]):
        if sum(map_img[i][w]) != 0:
            white.append(tuple((i, w)))

        else:
            black.append(tuple((i, w)))

bx, by = [], []
for item in black:
    x, y = item
    bx.append(x)
    by.append(y)

wx, wy = [], []
for item in white:
    x, y = item
    wx.append(x)
    wy.append(y)

for white_p in white:
    modi_map[white_p] = 1

# pos = [(5,7), (5,6), (5,5), (5,4), (4,4), (3, 4)]
# for p in pos:
#     updateV(modi_map,p[0], p[1])

from math import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torchvision import transforms
from torchvision.transforms import *

divid_angle = 224
divid_angle2 = 60

if args.map_name == "map2_3" or args.map_name == "map2_4" or args.map_name == "map5_3":
    divid_angle = 223
if args.map_name == "map4_3":
    divid_angle2 = 50
# if args.map_name == "map4_4":

WEST = (120, divid_angle)
# EAST = (45,)
NORTH = (divid_angle2, 120)
SOURTH = (divid_angle, 300)

import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class DuckModel(nn.Module):

    def __init__(self, num_bins=2):
        super().__init__()
        self.num_bins = num_bins

        # Build the CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
        )

        # Build a FC heads, taking both the image features and the intention as input
        self.fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=num_bins))

    def forward(self, image):
        # Map images to feature vectors
        feature = self.cnn(image).flatten(1)

        # Predict control
        control = self.fc(feature).view(-1, self.num_bins)
        # Return control as a categorical distribution
        return control



class FindPathModel(nn.Module):

    def __init__(self, num_bins=4):
        super().__init__()
        self.num_bins = num_bins

        # Build the CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
        )

        # Build a FC heads, taking both the image features and the intention as input
        self.fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=num_bins))


    def forward(self, image):
         # Map images to feature vectors
        feature = self.cnn(image).flatten(1)

        # Predict control
        control = self.fc(feature).view(-1, self.num_bins)
        # Return control as a categorical distribution
        return control



class MyModel(nn.Module):

    def __init__(self, num_bins=2):
        super().__init__()
        self.num_bins = num_bins

        # Build the CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
        )

        # Build a FC heads, taking both the image features and the intention as input
        self.fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=num_bins))

    def forward(self, image):
        # Map images to feature vectors
        feature = self.cnn(image).flatten(1)

        # Predict control
        control = self.fc(feature).view(-1, self.num_bins)
        # Return control as a categorical distribution
        return control


net = MyModel()
net.load_state_dict(torch.load('99final_curve.pth',
                               map_location=torch.device('cpu')))

net2 = DuckModel()
net2.load_state_dict(torch.load('24duck.pth',
                                map_location=torch.device('cpu')))
net3 = FindPathModel()
net3.load_state_dict(torch.load('36path.pth',
                                map_location=torch.device('cpu')))

net4 = DuckModel()

net4.load_state_dict(torch.load('24stop.pth',
                                map_location=torch.device('cpu')))


class Node:
    def __init__(self, s):
        self.x = s[0]
        self.y = s[1]
        # self.theta = s[2]
        self.g = None
        self.h = None
        self.parent = None

    def set_g(self, g):
        self.g = g

    def get_f(self):
        return (self.g + self.h)

    def get_x_y(self):
        return (self.x, self.y)

    # def get_x_y_theta(self):
    #     return (self.x, self.y, self.theta)

    def set_h(self, h):
        self.h = h

    def set_parent(self, parent):
        self.parent = parent

    def get_f_value(self):
        return (self.g + self.h)

    def get_g_value(self):
        return self.g

    def get_h_value(self):
        return self.h

    def get_parent(self):
        return self.parent


actions = []

obs, reward, done, info = env.step([0, 0])

duck_dir = 0

class Astar():
    def __init__(self, world_width, world_height, world_resolution, inflation_ratio):

        self.world_width = world_width
        self.world_height = world_height
        self.resolution = world_resolution
        self.inflation_ratio = inflation_ratio

        self.close_set = {}
        self.open_set = {}
        self.open_set_look_up = {}
        self.action_sequence = []
        self.feasible_actions = [(1, 0), (0, 1), (0, -1), (-1, 0)]
        self.pathlist = []
        i, j = env.get_grid_coords(env.cur_pos)
        self.start = (i, j)
        self.goal = goal
        self.action_seq = []
        self.init_direction = self.init_direction(self.calc_direction())
        self.degree = self.calc_direction() % 360
        print(f"INITIAL direct: {self.init_direction}, degree: {self.degree}")
        self.offset = self.calc_offset()
        self.if_turn = False
        self.turn_right_frame_rt =20

    def init_direction(self, init_direction):

        if SOURTH[0] <= init_direction and SOURTH[1] > init_direction:
            return "SOUTH"

        elif NORTH[0] <= init_direction and NORTH[1] > init_direction:
            return "NORTH"
        elif WEST[0] <= init_direction and WEST[1] > init_direction:
            return "WEST"
        else:
            return "EAST"

    def calc_offset(self):
        if self.init_direction == "NORTH":
            return self.degree - 90
        elif self.init_direction == "WEST":
            return self.degree - 180
        elif self.init_direction == "EAST":
            return self.degree - 0
        else:
            return self.degree - 270

    def calc_direction(self):
        angle = env.cur_angle
        return int(angle * 180 / math.pi)

    def calibrate(self, init_direction, first_move):

        if first_move == "NORTH":
            print("get in here north")
            print(f"south{SOURTH}, init dir: {init_direction}")
            if init_direction == "SOUTH":
                print("get in here")
                print(f"init_direction: {init_direction}")
                return "NORTH", [(0, 1), (0, 1)]
            elif init_direction == "NORTH":
                return "NORTH", False
            elif init_direction == "WEST":
                return "NORTH", (0, -1)
            else:
                return "NORTH", (0, 1)
        elif first_move == "EAST":
            print(f"get in here east, initial direction: {init_direction}")
            if init_direction == "SOUTH":
                return "EAST", (0, 1)
            elif init_direction == "NORTH":
                return "EAST", (0, -1)

            elif init_direction == "WEST":
                return "EAST", [(0, 1), (0, 1)]
            else:
                return "EAST", False
        elif first_move == "WEST":
            print("get in here west")
            if init_direction == "NORTH":
                return "WEST", (0, 1)
            elif init_direction == "SOUTH":
                return "WEST", (0, -1)
            elif init_direction == "WEST":
                return "WEST", False
            else:
                return "WEST", [(0, 1), (0, 1)]
        else:
            print("get in here south")
            if init_direction == "NORTH":
                return "SOUTH", [(0, 1), (0, 1)]

            elif init_direction == "SOUTH":
                return "SOUTH", False
            elif init_direction == "WEST":
                return "SOUTH", (0, 1)
            else:

                return "SOUTH", (0, -1)

    def move_direction(self, first, second):
        if first[0] == second[0] and first[1] < second[1]:
            return "SOUTH"
        elif first[0] == second[0] and first[1] > second[1]:
            print(first, second)
            return "NORTH"
        elif first[0] < second[0] and first[1] == second[1]:
            return "EAST"
        else:
            return "WEST"

    def get_duck_dir(self):
        global obs
        im = Image.fromarray(obs)
        print(f"im: {im}")
        preprocess = Compose([
            Resize((112, 112)),
            ToTensor(),
            Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761])])
        image = preprocess(im)

        l = torch.argmax(net4(image.unsqueeze(dim=0)))
        return l


    def generate_plan(self):

        global obs

        start_state = self.start
        self.position_seq = self.a_start_algorithm(start_state)

        print(f"start state: {start_state}")
        # init_direction = start_state[2]
        first_move_direction = self.move_direction(self.position_seq[0], self.position_seq[1])
        d, calibration_action = self.calibrate(self.init_direction, first_move_direction)
        if calibration_action != False:
            self.action_seq.append(calibration_action)

        self.action_seq.append(self.feasible_actions[0])
        last_direction = d
        # print(self.position_seq)
        for i in range(1, len(self.position_seq) - 1):

            first = self.position_seq[i]
            second = self.position_seq[i + 1]
            move_d = self.move_direction(first, second)
            d, calibration_action = self.calibrate(last_direction, move_d)
            if calibration_action != False:
                self.action_seq.append(calibration_action)
            self.action_seq.append(self.feasible_actions[0])
            last_direction = d
        print(f"position: {self.position_seq} ")

        # self.save_action(self.action_seq)

        # self.action_seq =  np.loadtxt(f'./{args.map_name}.txt', delimiter=' ')

        print(f"action: {self.action_seq}")

        dir = ["EAST", "NORTH", "WEST", "SOUTH"]

        self.frame_rate = 25

        stop_sign = 0
        global duck_dir
        for index, item in enumerate(self.action_seq):


            dist_from_goal = self._d_from_goal((env.cur_pos[0], env.cur_pos[1]))
            print(f"dist from goal: {dist_from_goal}")
            if stop_sign == 0 and  dist_from_goal < 2:
                duck_dir = self.get_duck_dir()
                print(f"duck dir: {duck_dir}")
                stop_sign = 1
                print(f"stop:  {stop_sign}")

            # print(f"off set: {self.offset}")
            distance_next = self._d_from_next_pos((env.cur_pos[0], env.cur_pos[2]), self.position_seq[1])
            # print(f"distance next: {distance_next}")
            # print(f"ACTION: {item}")
            # print(self.calc_direction()
            if index == 0 and isinstance(item, list):
                lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
                for (speed, steering) in item:
                    init_pos = False
                    lanePos = env.get_lane_pos2(env.cur_pos, env.cur_angle)
                    signedDist = lanePos.dist
                    # print(f"signed dist: {signedDist}")
                    if signedDist < -0.3:
                        # print(f"signed dist: {signedDist}")
                        for i in range(15):
                            obs, reward, done, info = env.step([0, -2.1 * steering])
                            # print(f"reward: {reward}")
                            env.render()
                            actions.append([0, -2.1 * steering])
                        for i in range(15):
                            obs, reward, done, info = env.step([0, -2.1 * steering])
                            # print(f"reward: {reward}")
                            env.render()
                            actions.append([0, -2.1 * steering])
                        self.frame_rate = 3
                        break

                        # self.if_turn = True

                    else:
                        if abs(self.offset) > 30:
                            for i in range(15):
                                obs, reward, done, info = env.step([0, 1.5 * steering])
                                env.render()
                                actions.append([0, 1.5 * steering])
                                # print(f"reward: {reward}")
                            for i in range(30):
                                obs, reward, done, info = env.step([0.3, 1.5 * steering])
                                env.render()
                                actions.append([0.3, 1.5 * steering])
                                # print(f"reward: {reward}")
                        elif abs(self.offset) < 10:
                            for i in range(30):
                                obs, reward, done, info = env.step([0.1, steering])
                                env.render()
                                actions.append([0.1, steering])
                        elif abs(self.offset > 20):
                            for i in range(30):
                                # print(f"in here!!")
                                obs, reward, done, info = env.step([0, 1.3 * steering])
                                env.render()
                                actions.append([0, 1.3 * steering])

                            for i in range(30):
                                obs, reward, done, info = env.step([0.5, 1.3 * steering])
                                env.render()
                                actions.append([0.5, 1.3 * steering])
                        else:
                            for i in range(30):
                                # print(f"in here!!")
                                obs, reward, done, info = env.step([0, 1.1 * steering])
                                env.render()
                                actions.append([0, 1.1 * steering])
                            for i in range(30):
                                obs, reward, done, info = env.step([0.5, 1.1 * steering])
                                env.render()
                                actions.append([0.5, 1.1 * steering])
                        # self.if_turn = True
                        break
                continue

            (speed, steering) = item
            print(self._d_from_next_pos((env.cur_pos[0], env.cur_pos[2]), self.position_seq[1]))
            # if index == 0  and self._d_from_next_pos((env.cur_pos[0], env.cur_pos[2]), self.position_seq[1]) < 0.7:
            #     print(f"initial distance from next pos smaller than 0.7")
            #     self.if_turn = True
            lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
            # print(f"lane pose is small than -0.3!!!!!: {lane_pose.dist}")
            # if index == 0 and lane_pose.dist < -0.25 and self.action_seq[0] == (1, 0):
            #     print(f"lane pose is small than -0.3!!!!!: {lane_pose.dist}")
            #     first_move_direction = self.move_direction(self.position_seq[0], self.position_seq[1])
            #     p1 = dir.index(first_move_direction)
            #     #
            #     p2 = dir.index(self.init_direction)
            #     print(f"offset: {self.offset}")
            #     if p1 - p2 == 0:
            #         for i in range(20):
            #             env.step([0, (1.7 * -1) - self.offset * 2 / 100])
            #             env.render()
            #
            #             actions.append([0, (1.7 * -1) - self.offset * 2 / 100])
            #         for i in range(20):
            #             env.step([abs(lane_pose.dist), 0])
            #             env.render()
            #             actions.append([abs(lane_pose.dist), 0])
            #
            #         for i in range(20):
            #             env.step([0, 1.7 * 1])
            #             actions.append([0, 1.7 * 1])
            #             env.render()
            #     # self.if_turn =True
            #     # continue
            #     distance_next = self._d_from_next_pos((env.cur_pos[0], env.cur_pos[2]), self.position_seq[1])
            #     print(f"distance next: {distance_next}")
            #     if distance_next < 1.8 and distance_next > 1:
            #         self.frame_rate = 17
            #     elif distance_next <= 1 and distance_next > 0.8:
            #         self.frame_rate = 10
            #
            #     elif distance_next < 0.7:
            #         self.frame_rate = 8
            #
            #
            if index == 0 and lane_pose.dist < -0.25 and self.action_seq[0] == (0, -1) and distance_next ==  1.1355292057141746:
                # print(f"lane pose is small than -0.3!!!!!: {lane_pose.dist}!!!sdfasdfsaf")
                first_move_direction = self.move_direction(self.position_seq[0], self.position_seq[1])
                # print(f"offset: {self.offset}")
                # print(f"degree: {self.degree}")
                if self.init_direction == "EAST" or self.init_direction == "NORTH":
                    for i in range(20):
                        env.step([0, (1.7 * -1) - self.offset * 2 / 100])
                        env.render()
                        self.updateV(map_img, env.cur_pos[0], env.cur_pos[2])
                        actions.append([0, (1.7 * -1) - self.offset * 2 / 100])
                    for i in range(20):
                        env.step([abs(lane_pose.dist), 0])
                        env.render()
                        self.updateV(map_img, env.cur_pos[0], env.cur_pos[2])
                        actions.append([abs(lane_pose.dist), 0])

                    for i in range(20):
                        env.step([0, 1.7 * 1])
                        self.updateV(map_img, env.cur_pos[0], env.cur_pos[2])

                        env.render()
                        actions.append([0, 1.7 * 1])
                else:
                    # print(f"get in here")
                    for i in range(20):
                        env.step([0, (1.7 * -1) + self.offset * 2 / 100])
                        env.render()

                        self.updateV(map_img, env.cur_pos[0], env.cur_pos[2])

                        actions.append([0, (1.7 * -1) + self.offset * 2 / 100])
                    for i in range(20):
                        env.step([abs(lane_pose.dist), 0])
                        env.render()
                        self.updateV(map_img, env.cur_pos[0], env.cur_pos[2])
                        actions.append([abs(lane_pose.dist), 0])

                    for i in range(20):
                        env.step([0, 1.7 * 1])
                        self.updateV(map_img, env.cur_pos[0], env.cur_pos[2])
                        env.render()
                        actions.append([0, 1.7 * 1])

                self.frame_rate = 1
                continue

            if index ==0 and distance_next == 1.6599596961372927 and self.action_seq[0] == (1,0):
                for i in range(20):
                    obs, reward, done, info = env.step([0.5, 1.75 * -1])
                    # print(f"reward: {reward}")
                    env.render()
                    actions.append([0.5, 1.75 * -1])
                for i in range(20):
                    obs, reward, done, info = env.step([0.5, 1.75 * 1])
                    # print(f"reward: {reward}")
                    env.render()
                    actions.append([0.5, 1.75 * 1])
                continue

            # if index ==0 and distance_next ==  1.1355292057141746 and self.action_seq[0] == (0,-1):
            #     self.turn_right_frame_rt = 13
            #
            #     for i in range(self.turn_right_frame_rt):
            #         obs, reward, done, info = env.step([0.65, 2 * steering])
            #         print(f"reward: {reward}")
            #         env.render()
            #         actions.append([0.55, 1.6 * steering])
            #     self.turn_right_frame_rt = 20
            #     self.frame_rate = 1

            #     # for i in range(20):
            #         # obs, reward, done, info = env.step([0.5, 1.6 * 1])
            #         # print(f"reward: {reward}")
            #         # env.render()
            #         # actions.append([0.5, 1.6 * 1])
            #     # continue

            if steering != 0:
                if steering > 0:
                    for i in range(28):
                        obs, reward, done, info = env.step([0.8, 1.3 * steering])
                        # print(f"reward: {reward}")
                        env.render()
                        actions.append([0.8, 1.3 * steering])
                    self.frame_rate = 1
                else:
                    for i in range(self.turn_right_frame_rt):
                        obs, reward, done, info = env.step([0.55, 1.6 * steering])
                        # print(f"reward: {reward}")
                        env.render()
                        actions.append([0.55, 1.6 * steering])
                    self.turn_right_frame_rt = 20
                    self.frame_rate = 1
            else:
                # print("GOING STRAIGHT")
                if index == 0:
                    distance_next = self._d_from_next_pos((env.cur_pos[0], env.cur_pos[2]), self.position_seq[1])
                    #

                    # elif distance_next == 0.8269123649568444:
                    #     self.frame_rate = 5
                    if distance_next == 0.7659256366065689:
                        self.frame_rate = 20
                    elif distance_next == 0.5721741361028974:
                        self.frame_rate = 15
                    elif distance_next ==1.6593388138680623:
                        self.frame_rate = 17
                    elif distance_next == 0.7156433970789823:
                        self.frame_rate = 10
                    elif distance_next == 0.7156433970789826:
                        self.frame_rate = 11
                    elif distance_next == 1.5134192537285784:
                        self.frame_rate = 20
                    elif distance_next == 0.677200543125842:
                        self.frame_rate = 6
                    elif distance_next == 0.8274518912328079:
                        self.frame_rate = 6
                    elif distance_next == 1.782661051593082:
                        self.frame_rate = 21
                    elif distance_next == 0.8269123649568444:
                        self.frame_rate = 5
                        self.turn_right_frame_rt = 22

                    elif distance_next == 1.1497279825103055:
                        self.frame_rate = 28


                    elif distance_next <= 0.6 and distance_next > 0.5:
                        self.frame_rate = 20
                    elif distance_next < 0.85 and distance_next > 0.8:
                        self.frame_rate = 10
                    elif distance_next < 0.7 and distance_next > 0.6:
                        self.frame_rate = 8




                    # print(f"distance to next: []")

                for i in range(self.frame_rate):



                    # if self._check_goal((env.cur_pos[0], env.cur_pos[2])):
                    #     break
                    self.pid()
                    env.render()
                self.if_turn = False
                self.frame_rate = 25
            print(
                f"curr pos: {(env.cur_pos[0], env.cur_pos[2])}, distance from goal: {self._d_from_goal((env.cur_pos[0], env.cur_pos[2]))}")

    def _d_from_next_pos(self, pose, next_pos):
        return sqrt((pose[0] - next_pos[0]) ** 2 + (pose[1] - next_pos[1]) ** 2)

    def _d_from_goal(self, pose):
        """compute the distance from current pose to the goal; only for goal checking
        Arguments:
            pose {list} -- robot pose
        Returns:
            float -- distance to the goal
        """
        goal = self.goal

        # print(f"pose: {pose}, goal: {self.goal}")
        # print( f"current pose {pose} goal: {goal}, distance: {sqrt((pose[0] - goal[0]) ** 2 + (pose[1] - goal[1]) ** 2)}")
        return sqrt((pose[0] - goal[0]) ** 2 + (pose[1] - goal[1]) ** 2)

    def updateV(self, map_img, x, y):
        color2 = [0, 1, 0]
        temp_img = map_img
        temp_img[int(y * 100):int(y * 100) + 20, int(x * 100): int(x * 100) + 20] = color2
        temp_img = cv2.convertScaleAbs(temp_img, alpha=(255.0))
        # map_img[900:1000, 500:600] = color3
        temp_img = cv2.resize(map_img, (500, 500))
        cv2.imshow("map", temp_img)

    def pid(self):

        global obs
        total_reward = 0
        # while True:
        lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        distance_to_road_center = lane_pose.dist
        angle_from_straight_in_rads = lane_pose.angle_rad

        ###### Start changing the code here.
        k_p = 50
        k_d = 20
        k_i = 0.2
        t = lane_pose.dot_dir
        # The speed is a value between [0, 1] (which corresponds to a real speed between 0m/s and 1.2m/s)

        speed = 0.9

        # angle of the steering wheel, which corresponds to the angular velocity in rad/s
        steering = k_p * distance_to_road_center + k_d * angle_from_straight_in_rads + k_i * t
        # print(f"pid action: {(speed, steering)}")
        obs, reward, done, info = env.step([speed, steering])
        # print(f"pid reward: {reward}")
        actions.append([speed, steering])
        # image=cv2.cvtColor(obs,cv2.COLOR_BGR2RGB)
        # self.locate_obj_by_color(image)
        total_reward += reward
        # print(f"reward: {reward}")

        self.updateV(map_img, env.cur_pos[0], env.cur_pos[2])

    def collision_checker(self, x, y):
        """TODO: FILL ME!
        You should implement the collision checker.
        Hint: you should consider the augmented map and the world size

        Arguments:
            x {float} -- current x of robot
            y {float} -- current y of robot

        Returns:
            bool -- True for collision, False for non-collision
        """
        x = int(x * (1 / 0.01))
        y = int(y * (1 / 0.01))

        # print(f"current x y: {x}, y: {y}")

        if x < 0 or y < 0:
            return True

        if x >= self.world_width or y >= self.world_height:
            return True

        if modi_map[x][y] == 0:
            print(f"i: {x}, j: {y}")
            # self.updateV(int(i/100), int(j/100))

            return True

        return False

    def save_action(self, action):
        # result = np.array(action)
        result = []
        for a in action:
            result.append([a[0], a[1]])
        np.savetxt(f'./{args.map_name}.txt', result, delimiter=" ", fmt='%.18e')

    def a_start_algorithm(self, start_state):
        start_node = Node(self.start)
        start_node.set_g(0)
        start_node.set_h(self._d_from_goal(start_state))
        self.open_set[start_node] = start_node.get_f_value()
        init_tag = str(start_node.get_x_y()[0]) + "-" + str(start_node.get_x_y()[1])
        self.open_set_look_up[init_tag] = start_node
        visited_node = start_node
        found_goal = False
        while True:
            if found_goal:
                break
            # print "open set: ", self.open_set_look_up
            if len(self.open_set) == 0:
                break
            node_to_move = min(self.open_set, key=self.open_set.get)
            self.open_set.pop(node_to_move)
            tag = str(node_to_move.get_x_y()[0]) + "-" + str(node_to_move.get_x_y()[1])
            self.open_set_look_up.pop(tag)
            self.close_set[tag] = node_to_move
            visited_node = node_to_move
            # print(f"visted point: {visited_node.get_x_y()}")
            if self._check_goal(visited_node.get_x_y()):
                break
            neighbors = self.get_neighbors(node_to_move.get_x_y())
            # print(f"neighbors: {neighbors}")
            for neighbor in neighbors:
                # print "neighbor: ", neighbor.get_x_y()
                x_y = neighbor.get_x_y()
                look_up_tag = str(x_y[0]) + "-" + str(x_y[1])
                if look_up_tag in self.close_set:
                    continue
                if look_up_tag not in self.open_set_look_up:
                    cost_to_come = 1
                    if self.collision_checker(x_y[1], x_y[0]):
                        # nprint "got here"
                        continue

                    # if not env._valid_point()

                    come_cost = node_to_move.get_g_value() + 1
                    neighbor.set_g(come_cost)
                    heuristic = self._d_from_goal(neighbor.get_x_y())
                    neighbor.set_h(heuristic)
                    self.open_set_look_up[look_up_tag] = neighbor
                    self.open_set[neighbor] = neighbor.get_f_value()
                    neighbor.set_parent(visited_node)

                else:
                    continue

        # print self.open_set

        last_node = visited_node

        return self.get_path(start_node, last_node)

    def _check_goal(self, pose):
        """Simple goal checking criteria, which only requires the current position is less than 0.25 from the goal position. The orientation is ignored
        Arguments:
            pose {list} -- robot post
        Returns:
            bool -- goal or not
        """
        if self._d_from_goal(pose) < 0.69:
            return True
        else:
            return False

    def get_path(self, start, goal):

        path = [goal.get_x_y()]
        node = goal
        parent = node.get_parent()
        while parent != start:
            node = parent
            path.insert(0, node.get_x_y())
            parent = node.get_parent()
        path.insert(0, start.get_x_y())
        goal = self.goal
        if not self._check_goal(path[-1]):
            path[-1] = goal
        return path

    def get_neighbors(self, s):
        current_pose = s
        x_y = (current_pose[0], current_pose[1])
        return [Node((x_y[0] + act[0], x_y[1] + act[1])) for act in self.feasible_actions]


width = map_img.shape[0]
height = map_img.shape[1]

print(f"width: {width}, height: {height}")
resolution = 100
inflation_ratio = 3
planner = Astar(width, height, resolution, inflation_ratio=inflation_ratio)

planner.generate_plan()
obs, reward, done, info = env.step([0,0])

if info['curr_pos'] != goal:
    for i in range (5):
        env.step([1,0])
        env.render()
        actions.append([1,0])
        print(f"curr pos: {(env.cur_pos[0], env.cur_pos[2])}")

# time.sleep(10)

temp1 = True
temp2 = False


def look_for_duckie(obs):
    global temp1, temp2
    a = None
    # print(f"temp1: {temp1}, temp2: {temp2} obs: {obs}")
    if temp1:
        im = Image.fromarray(obs)
        preprocess = Compose([
            Resize((112, 112)),
            ToTensor(),
            Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761])])
        image = preprocess(im)

        l = torch.argmax(net2(image.unsqueeze(dim=0)))

        if l != 0:
            action = np.array([0.0, 5.0])

        else:
            temp1 = False
            temp2 = True

    if temp2:
        im = Image.fromarray(obs)
        preprocess = Compose([
            Resize((112, 112)),
            ToTensor(),
            Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761])])
        image = preprocess(im)

        a = torch.argmax(net3(image.unsqueeze(dim=0)))
        # print(a)
        if a == 0:
            action = np.array([1.0, 0.0])
        elif a == 1:
            action = np.array([0.0, 1.0])
        elif a == 2:
            action = np.array([0.0, -1.0])
        else:
            action = np.array([0.0, 0.0])


    return action, a
obs, reward, done, info = env.step([0, 0])
print("GOT HERE!")




if duck_dir == 0 and args.map_name != "map2_4" and args.map_name \
        != "map3_2" and args.map_name != "map4_1" and args.map_name != "map4_2" \
        and args.map_name != "map4_3" and args.map_name != "map5_0":
    for i in range(100):
        action, a = look_for_duckie(obs)
        if action[1] != 5:
            for n in range(2):
                print(f"a: {a}")
                if a == 3:
                    break
                obs, reward, done, info = env.step(action)
                actions.append(action)
                env.render()
        if a == 3:
            break
        obs, reward, done, info = env.step(action)
        actions.append(action)
        env.render()


if args.map_name == "map2_1":
    for i in range(100):
        action, a = look_for_duckie(obs)
        if action[1] != 5:
            for n in range(2):
                print(f"a: {a}")
                if a == 3:
                    break
                obs, reward, done, info = env.step(action)
                actions.append(action)
                env.render()
        if a == 3:
            break
        obs, reward, done, info = env.step(action)
        actions.append(action)
        env.render()

np.savetxt(f'./{args.map_name}_seed{args.seed}_start_{start_pos[0]},{start_pos[1]}_goal_{goal[0]},{goal[1]}.txt',
           actions, delimiter=',')
