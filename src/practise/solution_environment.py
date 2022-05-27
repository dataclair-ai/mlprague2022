from typing import Optional

import tensorflow as tf
from tf_agents.bandits.environments.bandit_tf_environment import BanditTFEnvironment
from tf_agents.specs.tensor_spec import TensorSpec
from tf_agents.policies.utils import get_num_actions_from_tensor_spec
import tf_agents.trajectories.time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common, eager_utils


class SimpleEnvironment(BanditTFEnvironment):
    """SimpleEnvironment for book recommendations.

    Attributes:
        labels: tf.Variable(shape=(batch_size, num_actions), dtype=tf.int32)
            index of the optimal action
        num_actions: tf.constant(dtype=tf.int32)
            number of playable actions
        prev_labels: tf.Variable(shape=(batch_size, num_actions), dtype=tf.int32)
            index of the optimal action from past time_step
            used for calculation of optimal reward
        weights: tf.constant(shape=(num_actions), dtype=numeric)
            maximum reward that can be achieved by correctly playing each action
        rew_dtype: tf.DType
            dtype of the reward
    """

    def __init__(
        self,
        dataset: tf.data.Dataset,
        batch_size: int,
        action_spec: tf.TensorSpec,
        weights: Optional[tf.Tensor] = None,
    ):
        """
        Args:
            dataset: tf.data.Dataset
                unbatched dataset with two elements - (features, labels)
            batch_size: int
                dataset batch size to be used
            action_spec: discrete! BoundedTensorSpec
                discrete bounded TensorSpec specifiyng possible actions,
                it is used to extract number of actions and dtype of actions
            weights: tf.Tensor
                weight of each action, the maximum reward that can be achieved by plaing the action
        """

        self.num_actions = tf.constant(
            get_num_actions_from_tensor_spec(action_spec), dtype=tf.int32
        )

        # TimeStep spec
        context_spec = dataset.element_spec[0]
        observation_spec = TensorSpec(
            shape=context_spec.shape, dtype=context_spec.dtype
        )
        time_step_spec = ts.time_step_spec(observation_spec)
        self.rew_dtype = time_step_spec.reward.dtype

        # Dataset iterator
        dataset = dataset.batch(batch_size) if batch_size > 0 else dataset
        self._data_iterator = eager_utils.dataset_iterator(dataset)

        # Binary reward
        if weights is not None:
            if weights.shape != [
                self.num_actions,
            ]:
                raise ValueError("shape of param rewards must be `[num_actions,]`")
            self.weights = tf.cast(weights, dtype=self.rew_dtype)
        else:
            self.weights = tf.ones([self.num_actions], dtype=self.rew_dtype)

        size = batch_size if batch_size > 0 else 1

        self.prev_labels = common.create_variable(
            initial_value=0,
            shape=(size, self.num_actions),
            dtype=tf.int32,
            name="previous labels",
        )

        self.labels = common.create_variable(
            initial_value=0,
            shape=(size, self.num_actions),
            dtype=tf.int32,
            name="labels",
        )

        super(SimpleEnvironment, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            batch_size=batch_size,
            name="SimpleEnvironment",
        )

    def _observe(self) -> types.NestedArray:
        """Collects another batch of features and labels and prepares time_step.
        Updates current and previous labels.

        Returns:
            tf.Tensor(shape=(batch_size, num_features), dtype=self.time_step.obsevation.dtype)
                context
        """
        context, label = eager_utils.get_next(self._data_iterator)
        self.prev_labels.assign(self.labels)
        self.labels.assign(label)
        return context

    def _apply_action(self, action: types.NestedArray) -> types.Float:
        """Calculates rewards for current batch of actions.

        Args:
            action: tf.Tensor(shape=(batch_size, 1), dtype=self.action_spec.dtype)

        Returns:
            tf.Tensor(shape=(batch_size, 1), dtype=time_step_spec.reward.dtype)
            Rewards for each played action.
        """
        action_ohe = tf.one_hot(action, depth=self.num_actions)
        rews = tf.cast(self.labels, self.rew_dtype) * tf.reshape(self.weights, [1, -1])
        return tf.boolean_mask(rews, action_ohe)

    def optimal_reward_fn(self) -> tf.Tensor:
        """Leak optimal rewards.
        Typically used for calculating metrics such as `regret`.

        Returns:
            tf.Tensor(shape=(batch_size, 1), dtype=time_step_spec.reward.dtype)
        """
        return tf.math.reduce_max(
            tf.cast(self.prev_labels, dtype=self.rew_dtype) * self.weights, axis=1
        )
