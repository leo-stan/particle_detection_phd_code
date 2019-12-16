#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename: asl_score.py
Author: Leo Stanislas
Date Created: 13 Sep. 2019
Description: Average Score Loss function
"""

import numpy as np

def average_score_loss(y_target,y_proba):
    return np.sum(abs(y_proba - y_target)) / y_target.shape[0]
