import ray
import numpy as np
import os
import torch
from rlil.initializer import get_replay_buffer, call_seed
from rlil.environments import State, Action
from rlil.samplers import Sampler
from collections import defaultdict, namedtuple

# 这是一个构建每次训练时传入当前训练进度的信息
StartInfo = namedtuple("StartInfo",
                       ["sample_frames", # 训练的帧数
                        "sample_episodes", # 训练的生命周期数
                        "train_steps"], # 训练的总步数
                       defaults=(None, ) * 3)


@ray.remote
class Worker:
    def __init__(self, make_env, seed):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self._env = make_env()
        self._env.seed(seed)

        print("Worker initialized in PID: {}".format(os.getpid()))

    def sample(self, lazy_agent, worker_frames, worker_episodes):
        """
        Args:
            lazy_agent (rlil.agent.LazyAgent): agent for sampling 一个强化学习算法Agent，应该主要是用来采样动作
            worker_frames (int): number of frames to collect 这个是配置每次采样需要执行的步数
            worker_episodes (int): number of episodes to collect 这是控制进行采样时最多走多少步

        Returns:
            sample_info (StartInfo):
                keys: 
                    frames: the number of frames each episode
                    returns: the return per episode

            (States, Actions, rewards, NextStates)
        """

        sample_info = {"frames": [], "returns": []} # 构建采样的返回信息：步数和奖励reward
        lazy_agent.set_replay_buffer(self._env)

        # Sample until it reaches worker_frames or worker_episodes.
        while sum(sample_info["frames"]) < worker_frames \
                and len(sample_info["frames"]) < worker_episodes:

            self._env.reset()
            action = lazy_agent.act(self._env.state, self._env.reward) # 在这个里面会在每一步的时候构建一个当前步的对象存储样本
            _return = 0
            _frames = 0

            while not self._env.done:
                self._env.step(action)
                action = lazy_agent.act(self._env.state, self._env.reward)
                _frames += 1
                _return += self._env.reward.item()

            lazy_agent.replay_buffer.on_episode_end()
            sample_info["frames"].append(_frames)
            sample_info["returns"].append(_return)

        samples = lazy_agent.replay_buffer.get_all_transitions()
        samples.weights = lazy_agent.compute_priorities(samples)

        return sample_info, samples


class AsyncSampler(Sampler):
    """
    构建异步采样器
    AsyncSampler collects samples with asynchronous workers.
    All the workers have the same agent, which is given by the argument
    of the start_sampling method.
    """

    def __init__(
            self,
            env,
            num_workers=1,
    ):
        # todo 后面是怎么运行的
        self._env = env
        seed = call_seed()
        self._workers = [Worker.remote(env.duplicate, seed+i)
                         for i in range(num_workers)]
        self._work_ids = {worker: None for worker in self._workers} # 存储每个Work对应的id和起始训练信息，用于判断这个work是否有分配工作，这个主要是用来持有ray分布式对象的异步操作对象，方便后续获取异步的执行结果
        self.replay_buffer = get_replay_buffer()

    def start_sampling(self,
                       lazy_agent, # 传入的是一个强化学习算的构建的Agent对象，比如SAC
                       start_info=StartInfo(),
                       worker_frames=np.inf,
                       worker_episodes=np.inf,
                       ):

        # start_info has the information about when the sampling starts
        assert worker_frames != np.inf or worker_episodes != np.inf, \
            "worker_frames or worker_episodes must be specified"

        # start sample method if the worker is ready
        for worker in self._workers:
            if self._work_ids[worker] is None: # 防止重复派活
                # .remote 是RAY调用远程对象需要使用的方法，真正执行的是
                # .sample。这是一个异步对象，类似feature，可以用于获取异步对象的返回值
                self._work_ids[worker] = \
                    {"id": worker.sample.remote(
                        lazy_agent, worker_frames, worker_episodes),
                     "start_info": start_info}

    def store_samples(self, timeout=-1, evaluation=False):
        # if timeout < 0, wait until the sampling finishes

        # result is a dict of {start_info: {"frames": [], "returns": []}}
        result = defaultdict(lambda: {"frames": [], "returns": []})

        # store samples when the worker finishes sampling
        for worker, item in self._work_ids.items():
            _id = item["id"]
            start_info = item["start_info"]
            if timeout > 0:
                ready_id, remaining_id = \
                    ray.wait([_id], num_returns=1, timeout=timeout)
            else:
                ready_id = [_id]

            # if there is at least one finished worker
            if len(ready_id) > 0:
                # merge results
                sample_info, samples = ray.get(ready_id[0]) # 从远程获取制定id的对象的返回值
                result[start_info]["frames"] += sample_info["frames"] # 更新执行的步数
                result[start_info]["returns"] += sample_info["returns"] # 更新获取的奖励

                self._work_ids[worker] = None # 某个样本采集已经结束了则将其设置为None，方便后续部署新的任务
                if not evaluation:
                    self.replay_buffer.store(samples, priorities=samples.weights) # 将样本存储起来

        return result
