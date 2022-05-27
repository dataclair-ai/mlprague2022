from tensorflow import float32, Tensor, zeros, reshape
import tensorflow_probability as tfp

from tf_agents.agents.tf_agent import TFAgent
from tf_agents.bandits.environments.bandit_tf_environment import BanditTFEnvironment
from tf_agents.trajectories import TimeStep


def prep_reward_binary(
    rew_true_neg: float = 1,
    rew_false_neg: float = 0,
    rew_false_pos: float = 0,
    rew_true_pos: float = 1,
) -> tfp.distributions.Distribution:

    # probability of observing 1 (instead of zero)
    distr = tfp.distributions.Bernoulli(probs=[[0, 0], [0, 0]], dtype=float32)

    # shift the base value -> sample from {X; X+1} instead of {0; 1}
    reward_distr = tfp.bijectors.Shift(
        [[rew_true_neg, rew_false_neg], [rew_false_pos, rew_true_pos]]
    )(distr)

    return tfp.distributions.Independent(reward_distr, reinterpreted_batch_ndims=2)


def fake_timestep(obs: Tensor, spec: TimeStep, add_shape: bool = False) -> TimeStep:
    specvals = {
        i.name: zeros(i.shape, dtype=i.dtype) for i in spec if i.name is not None
    }
    if add_shape:
        add_shape = 1 if len(obs.shape) == 1 else obs.shape[0]
        obs = reshape(obs, [add_shape, spec.observation.shape[0]])
    return TimeStep(observation=obs, **specvals)


def predict(
    x: Tensor, agent: TFAgent, env: BanditTFEnvironment, add_shape: bool = False
):
    spec = env.time_step_spec()
    return agent.policy.action(fake_timestep(x, spec, add_shape))
