#!/usr/bin/env python

class Cfg():

    def __init__(self):

        # Gale paths
        # self.ROOT_DIR = '/home/leo/phd/particle_detection/src/particle_detection/particle_detection'
        # self.LOG_DIR = '/home/leo/phd/particle_detection/src/particle_detection/particle_detection/logs'
        # self.RAW_DATA_DIR = '/home/leo/phd/particle_detection/src/particle_detection/data'
        # self.DATASETS_DIR = '/home/leo/phd/particle_detection/src/particle_detection/particle_detection/data'
        # self.ROSBAGS_DIR = '/home/leo/phd/particle_detection/src/particle_detection/data/rosbags'

        # HPC paths
        # self.ROOT_DIR = '/home/n9615687/phd/particle_detection/particle_detection'
        # self.LOG_DIR = '/home/n9615687/phd/particle_detection/particle_detection/logs'
        # self.RAW_DATA_DIR = '/home/n9615687/phd/particle_detection/data'
        # self.DATASETS_DIR = '/home/n9615687/phd/particle_detection/particle_detection/data'
        # self.ROSBAGS_DIR = '/home/n9615687/phd/particle_detection/data/rosbags'

        # Zeus paths
        self.ROOT_DIR = '/home/guest/leo-ws/particle_detection/'
        self.LOG_DIR = '/home/guest/data_hdd/leo_data/particle_detection/particle_detection/logs'
        self.RAW_DATA_DIR = '/home/guest/data_hdd/leo_data/particle_detection/data'
        self.DATASETS_DIR = '/home/guest/data_hdd/leo_data/particle_detection/particle_detection/data'

cfg = Cfg()
