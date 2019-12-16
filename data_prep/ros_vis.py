# A simple tool to visualise numpy pcls in Rviz with keyboard control

import rospy
from pynput.keyboard import Listener, Key


class RosVisualiser(object):

    def __init__(self, max_id, rate=50, verbose=False):
        self.max_id = max_id
        self.verbose = verbose
        self.state = 'start'
        self.id = 0
        self.rate = rospy.Rate(rate)

    def update_state(self, key):
        if key == Key.space:
            if self.state == 'stop':
                self.state = 'play'
            elif self.state == 'play':
                self.state = 'stop'
        if key == Key.right:
            if self.state == 'stop':
                self.state = 'step_right1'
        if key == Key.left:
            if self.state == 'stop':
                self.state = 'step_left1'
        if key == Key.up:
            if self.state == 'stop':
                self.state = 'step_right10'
        if key == Key.down:
            if self.state == 'stop':
                self.state = 'step_left10'
        if key == Key.page_up:
            if self.state == 'stop':
                self.state = 'step_right100'
        if key == Key.page_down:
            if self.state == 'stop':
                self.state = 'step_left100'
        if key == Key.esc:
            self.state = 'exit'

    def update_id(self):
        with Listener(on_press=self.update_state) as listener:
            while self.state == 'stop':
                self.rate.sleep()
            if self.state == 'start':
                self.state = 'stop'
            if self.state == 'play':
                self.id = min(self.id + 1, self.max_id - 1)
                self.rate.sleep()
            if self.state == 'step_right1':
                self.id = min(self.id + 1, self.max_id - 1)
                self.state = 'stop'
            if self.state == 'step_right10':
                self.id = min(self.id + 10, self.max_id - 1)
                self.state = 'stop'
            if self.state == 'step_right100':
                self.id = min(self.id + 100, self.max_id - 1)
                self.state = 'stop'
            if self.state == 'step_left1':
                self.state = 'stop'
                self.id = max(0, self.id - 1)
            if self.state == 'step_left10':
                self.state = 'stop'
                self.id = max(0, self.id - 10)
            if self.state == 'step_left100':
                self.state = 'stop'
                self.id = max(0, self.id - 100)
            if self.verbose:
                print(self.id)
        return self.id
