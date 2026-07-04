import argparse
import pybullet
import pybullet_envs
from rlil.environments import GymEnvironment, ENVS
from rlil.experiments import Experiment
from rlil.presets import get_default_args
from rlil.presets import continuous
from rlil.initializer import get_logger, set_device, set_seed, get_writer
import torch
import logging
import ray
import pickle
import os
import shutil


def main():
    parser = argparse.ArgumentParser(
        description="Run an online_il benchmark.")
    parser.add_argument("env", help="Name of the env") # 游戏的名字
    parser.add_argument("agent",
                        help="Name of the online imitation learning agent \
                            (e.g. gail). See presets for available agents.") # 强化学习方法名字
    parser.add_argument("base_agent",
                        help="Name of the base agent (e.g. ddpg). \
                            See presets for available agents.") # RL算法名称
    parser.add_argument("dir",
                        help="Directory where the transitions.pkl is saved.") # 专家数据目录
    parser.add_argument("--device", default="cuda",
                        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--train_minutes", type=int, default=60,
                        help="Minutes to train.")
    parser.add_argument("--trains_per_episode", type=int, default=5,
                        help="Number of training steps per episode")
    parser.add_argument("--num_workers", type=int,
                        default=1, help="Number of workers for training")
    parser.add_argument("--exp_info", default="default experiment",
                        help="One line descriptions of the experiment. \
                            Experiments' results are saved in 'runs/[exp_info]/[env_id]/'")

    args = parser.parse_args()

    # initialization
    ray.init(include_webui=False, ignore_reinit_error=True) # 启动 Ray 分布式框架。这个库用 Ray 做并行采样（多个 worker 同时收集环境交互数据）。必须在任何 Ray 相关操作前调用。
    set_device(torch.device(args.device))
    set_seed(args.seed)
    logger = get_logger()
    logger.setLevel(logging.DEBUG)

    # set environment
    if args.env in ENVS:
        env_id = ENVS[args.env]
    else:
        env_id = args.env
    env = GymEnvironment(env_id, append_time=True)

    # set base_agent
    base_preset = getattr(continuous, args.base_agent) # 获取基础强化学习算法的类型（可能是SAC算法）
    base_agent_fn = base_preset() # 构建强化学习算法的对象

    # set agent 加载专家数据
    with open(os.path.join(args.dir, "transitions.pkl"), mode='rb') as f:
        transitions = pickle.load(f)
    preset = getattr(continuous, args.agent) # 继续获取强化学习算法（应该是SQIL算法），todo 为什么会构建两个强化学习算法对象
    agent_fn = preset(
        transitions=transitions,
        base_agent_fn=base_agent_fn,
    ) # 构建强化学习的Agent，在这个Agent里面会传入`专家数据`和`基础强化学习算法对象`
    
    # 这里应该是去掉名字前面的.或者_
    agent_name = agent_fn.__name__[1:]
    base_agent_name = base_agent_fn.__name__[1:]

    # set args_dict
    args_dict = {"args": {}, base_agent_name: {}, agent_name: {}}
    args_dict["args"] = vars(args)
    args_dict[base_agent_name] = get_default_args(base_preset)
    args_dict[agent_name] = get_default_args(preset)

    Experiment(
        agent_fn, env, # agent强化学习算法工厂函数+算法
        agent_name=agent_name + "-" + base_agent_name, # 用于日志目录的命令
        num_workers=args.num_workers, # 并行的采集样本数量
        train_minutes=args.train_minutes, # 训练的市场
        trains_per_episode=args.trains_per_episode, # 每次训练的生命周期的数量
        args_dict=args_dict, # 训练参数字典
        seed=args.seed,
        exp_info=args.exp_info, # 专家数据信息
    )

    # copy demo_return.json if exists 应该是备份训练的记录和数据
    demo_return_path = os.path.join(args.dir, "demo_return.json")
    if os.path.exists(demo_return_path):
        writer = get_writer()
        shutil.copy2(demo_return_path, writer.log_dir)


if __name__ == "__main__":
    main()
