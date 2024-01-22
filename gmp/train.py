import numpy as np

from rl.base import AlgoType, Base, EnvProcs, EnvType
from rl.buffer import OffPolicyBuffer, OnPolicyBuffer
import rl.callbacks.callback as callback
from rl.callbacks.episode_return_callback import EpisodeReturnCallback
from rl.logging import Logger
from rl.save import Saver, SaverContext
from rl.train import process_action, process_reward
from rl.types import EnvLike

from gmp.buffer import ExperienceLatent
from gmp.latent_space import random_ball_numpy


def process_termination(
    step: int,
    env: EnvLike,
    done,
    trunc,
    logs: dict,
    env_type: EnvType,
    env_procs: EnvProcs,
    callbacks: list[callback.Callback],
    rng: np.random.Generator,
    dimension: int,
    latents: np.ndarray,
):
    def single_one_process(env, done, trunc, logs):
        if done or trunc:
            print(step, " > ", logs["episode_return"])
            callback.on_episode_end(
                callbacks,
                callback.CallbackData.on_episode_end(0, logs["episode_return"]),
            )
            logs["episode_return"] = 0.0
            next_observation, info = env.reset()
            latents = random_ball_numpy(rng, 1, dimension)
            return next_observation, info, latents
        return None, None

    def single_many_process(env, done, trunc, logs):
        for i, (d, t) in enumerate(zip(done, trunc)):
            if d or t:
                callback.on_episode_end(
                    callbacks,
                    callback.CallbackData.on_episode_end(0, logs["episode_return"][i]),
                )
                if i == 0:
                    print(step, " > ", logs["episode_return"][i])
                logs["episode_return"][i] = 0.0
                latents[i] = random_ball_numpy(rng, 1, dimension)
        return None, None, latents

    def parallel_one_process(env, done, trunc, logs):
        raise NotImplementedError
        # if any(done.values()) or any(trunc.values()):
        #     print(step, " > ", logs["episode_return"])
        #     callback.on_episode_end(
        #         callbacks,
        #         callback.CallbackData.on_episode_end(0, logs["episode_return"]),
        #     )
        #     logs["episode_return"] = 0.0
        #     next_observation, info = env.reset()
        #     return next_observation, info
        # return None, None

    def parallel_many_process(env, done, trunc, logs):
        raise NotImplementedError
        # check_d, check_t = np.stack(list(done.values()), axis=1), np.stack(
        #     list(trunc.values()), axis=1
        # )
        # for i, (d, t) in enumerate(zip(check_d, check_t)):
        #     if np.any(d) or np.any(t):
        #         callback.on_episode_end(
        #             callbacks,
        #             callback.CallbackData.on_episode_end(0, logs["episode_return"][i]),
        #         )
        #         if i == 0:
        #             print(step, " > ", logs["episode_return"][i])
        #         logs["episode_return"][i] = 0.0
        # return None, None

    if env_type == EnvType.SINGLE and env_procs == EnvProcs.ONE:
        return single_one_process(env, done, trunc, logs)
    elif env_type == EnvType.SINGLE and env_procs == EnvProcs.MANY:
        return single_many_process(env, done, trunc, logs)
    elif env_type == EnvType.PARALLEL and env_procs == EnvProcs.ONE:
        return parallel_one_process(env, done, trunc, logs)
    elif env_type == EnvType.PARALLEL and env_procs == EnvProcs.MANY:
        return parallel_many_process(env, done, trunc, logs)
    else:
        raise NotImplementedError


def train_with_latent(
    seed: int,
    base: Base,
    env: EnvLike,
    n_env_steps: int,
    env_type: EnvType,
    env_procs: EnvProcs,
    algo_type: AlgoType,
    *,
    start_step: int = 1,
    saver: Saver = None,
    callbacks: list[callback.Callback] = None,
):
    callbacks = callbacks if callbacks else []
    callbacks = [EpisodeReturnCallback(population_size=1)] + callbacks
    callback.on_train_start(callbacks, callback.CallbackData())

    if algo_type == AlgoType.ON_POLICY:
        buffer = OnPolicyBuffer(seed, base.config.update_cfg.max_buffer_size)
    else:
        buffer = OffPolicyBuffer(seed, base.config.update_cfg.max_buffer_size)

    observation, info = env.reset(seed=seed + 1)

    logger = Logger(callbacks, env_type=env_type, env_procs=env_procs)
    logger.init_logs(observation)

    rng = np.random.default_rng(seed)
    latents = random_ball_numpy(
        rng, base.config.env_cfg.n_envs, base.config.algo_params.latent_size
    )

    with SaverContext(saver, base.config.train_cfg.save_frequency) as s:
        for step in range(start_step, n_env_steps + 1):
            logger["step"] = step

            action, log_prob = base.explore(
                {"observation": observation, "latent": latents}
            )

            next_observation, reward, done, trunc, info = env.step(
                process_action(action, env_type, env_procs)
            )
            logger["episode_return"] += process_reward(reward, env_type, env_procs)

            termination = process_termination(
                step * base.config.env_cfg.n_envs,
                env,
                done,
                trunc,
                logger,
                env_type,
                env_procs,
                callbacks,
                rng,
                base.config.algo_params.latent_size,
                latents,
            )
            latents = termination[2]
            if termination[0] is not None and termination[1] is not None:
                next_observation, info = termination

            buffer.add(
                ExperienceLatent(
                    observation=observation,
                    latent=latents,
                    action=action,
                    reward=reward,
                    done=done,
                    next_observation=next_observation,
                    log_prob=log_prob,
                )
            )

            if base.should_update(step, buffer):
                callback.on_update_start(callbacks, callback.CallbackData())
                logger.update(base.update(buffer))
                callback.on_update_end(
                    callbacks, callback.CallbackData(logs=logger.get_logs())
                )

            s.update(step, base.state)

            observation = next_observation

    env.close()
    callback.on_train_end(callbacks, callback.CallbackData())
