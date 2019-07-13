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
from dmbrl.values import DeepValueFunction
from dmbrl.env.pointbot import PointBotTeacher


def main(env, ctrl_type, ctrl_args, overrides, logdir, mode):
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
    overrides.append(["exp_cfg.log_cfg.nrecord", "0"])
    cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir, mode)

    if cfg.exp_cfg.exp_cfg.use_value:
        cfg.exp_cfg.exp_cfg.value = DeepValueFunction("value", cfg.val_cfg)
        cfg.exp_cfg.exp_cfg.value_target = DeepValueFunction("target", cfg.val_cfg)
        
    if not cfg.exp_cfg.exp_cfg.load_samples:
        cfg.exp_cfg.exp_cfg.teacher = cfg.exp_cfg.sim_cfg.env.teacher(cfg.exp_cfg.exp_cfg.value.model.sess)
        cfg.exp_cfg.exp_cfg.teacher.env.set_mode(mode)

    cfg.ctrl_cfg.value_func = cfg.exp_cfg.exp_cfg.value
    cfg.ctrl_cfg.target_value_func = cfg.exp_cfg.exp_cfg.value_target
    cfg.ctrl_cfg.use_value = cfg.exp_cfg.exp_cfg.use_value

    if ctrl_type == "MPC":
        cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)

    exp = MBExperiment(cfg.exp_cfg)

    cfg.pprint()
    os.makedirs(exp.logdir)
    with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
        f.write(pprint.pformat(cfg.toDict()))

    exp.run_experiment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, required=True,
                        help='Environment name: select from [pointbot1, pointbot2, pointbot3, pointbot4, reachersparse, pick_and_place]')
    parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=[],
                        help='Controller arguments, see https://github.com/bthananjeyan/saved-rl')
    parser.add_argument('-o', '--override', action='append', nargs=2, default=[],
                        help='Override default parameters, see https://github.com/bthananjeyan/saved-rl')
    parser.add_argument('-logdir', type=str, default='log',
                        help='Directory to which results will be logged (default: ./log)')
    args = parser.parse_args()

    mode = None
    if "pointbot" in args.env:
        mode = int(args.env[-1])
        args.env = "pointbot"

    main(args.env, "MPC", args.ctrl_arg, args.override, args.logdir, mode)
