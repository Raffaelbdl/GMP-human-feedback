from rl.base import AlgoType, Base, EnvProcs, EnvType
from rl.buffer import OffPolicyBuffer, OnPolicyBuffer
import rl.callbacks.callback as callback
from rl.callbacks.episode_return_callback import EpisodeReturnCallback
from rl.logging import Logger
from rl.save import Saver, SaverContext
from rl.train import process_action, process_reward, process_termination
from rl.types import EnvLike

from gmp.buffer import ExperienceLatent


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

    with SaverContext(saver, base.config.train_cfg.save_frequency) as s:
        for step in range(start_step, n_env_steps + 1):
            logger["step"] = step

            latent = base.latent if env_procs == EnvProcs.MANY else base.latent[0]
            action, log_prob = base.explore(
                {"observation": observation, "latent": latent}
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
            )
            if termination[0] is not None and termination[1] is not None:
                next_observation, info = termination

            buffer.add(
                ExperienceLatent(
                    observation=observation,
                    latent=latent,
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
