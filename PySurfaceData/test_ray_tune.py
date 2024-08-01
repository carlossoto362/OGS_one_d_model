#!/usr/bin/env python

"""This example demonstrates the usage of BOHB with Ray Tune.

Requires the HpBandSter and ConfigSpace libraries to be installed
(`pip install hpbandster ConfigSpace`).
"""

import json
import os
import time

import numpy as np

import ray
from ray import train, tune
from ray.tune import Trainable
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import ConfigSpace as CS

import torch
import time

from torch.distributions.multivariate_normal import MultivariateNormal

mean_time = 0
for i in range(100):
    time_init = time.time()
    samples1 = MultivariateNormal(torch.zeros(3),scale_tril=torch.eye(3)).sample(sample_shape=torch.Size([1000,1]))
    mean_time += time.time() - time_init
print(mean_time/100,samples1.shape)

mean_time = 0
for i in range(100):
    time_init = time.time()
    samples1 = torch.randn(torch.Size([1000,1,3]))
    mean_time += time.time() - time_init
print(mean_time/100, samples1.shape)


    
