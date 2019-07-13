from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import pprint

from dotmap import DotMap

from dmbrl.misc.MBExp import MBExperiment
from dmbrl.controllers.MPC import MPC
from dmbrl.config import create_config
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

def plot_returns(returns):
    plt.plot(returns)
    print(len(returns))
    plt.xlabel('Iteration')
    plt.ylabel("Return")
    plt.title("Training Curve")
    plt.ylim(0, 110)
    plt.savefig('returns.png')
    plt.show()
    plt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-log_path', type=str, required=True)
    args = parser.parse_args()
    logging_data = sio.loadmat('log/' + args.log_path + '/logs.mat')
    returns = logging_data['returns'][0]
    print(returns, len(returns))
    plot_returns(returns)