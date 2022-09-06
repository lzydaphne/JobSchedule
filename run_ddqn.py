import argparse
import os
import shutil

from env.job_env import JobEnv
from model import Net2
from pf_helper.agent_builder import build_dqn_agent
from pf_helper.pf_runner import PfRunner
from util import global_util
from util.file_loader import load_instances


def copy_data_folder_to_output(args, override=False):
    """
    拷贝data目录到输出目录下
    :param args:
    :param override: 默认是否覆盖
    :return:
    """
    make_new_dir = True
    if args.test:
        make_new_dir = False
    elif os.path.exists(args.output):
        if args.test:
            make_new_dir = False
        else:
            if override:
                key = "y"
            else:
                key = input("输出目录已经存在，是否覆盖? (y/n)")
            if key == "y":
                make_new_dir = True
                shutil.rmtree(args.output)
            else:
                make_new_dir = False
    if make_new_dir:
        shutil.copytree(os.path.join(global_util.get_project_root(), "data"), os.path.join(args.output, "data"))

    args.data_dir = os.path.join(args.output, "data")


def parse_args():
    parser = argparse.ArgumentParser("Parse configuration")
    parser.add_argument("--output", type=str, default="../output_jobshop", help="root path of output dir")
    parser.add_argument("--seed", type=int, default=42, help="seed of the random")
    parser.add_argument("--test", default=False, action="store_true", help="whether in test mode")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu id to use. If less than 0, use cpu instead.")
    parser.add_argument("--render", default=False, action="store_true", help="whether in the gui mode")
    parser.add_argument("--model_dir", type=str, default="model", help="folder path to save/load neural network models")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    copy_data_folder_to_output(args, True)
    run_config = global_util.load_yaml(os.path.join(args.output, "data/run_config.yml"))
    instance_dict = load_instances(os.path.join(args.output, "data/jobshop1.txt"))

    env = JobEnv(instance_dict[run_config["instance"]])
    action_size = env.action_space.n
    q_local_net = Net2(action_size)
    agent = build_dqn_agent(lambda x: x, args.data_dir, gpu=args.gpu, model=q_local_net, action_size=action_size)

    # train
    pf_runner = PfRunner(args, run_config, env)
    if args.test:
        pf_runner.validate(agent, start_i=1, phase="test")
    else:
        pf_runner.train(agent)