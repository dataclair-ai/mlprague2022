import tensorflow as tf

from tf_agents.utils.common import create_variable
from tf_agents.metrics import tf_metric
from tf_agents.policies.utils import get_num_actions_from_tensor_spec
from tf_agents.specs import BoundedTensorSpec
from tf_agents.trajectories.trajectory import Trajectory


class ActionCountMetric(tf_metric.TFStepMetric):
    """Action count metric implementation for TF agents

    Attributes:
        action_count: int
            how many possible actions there are
        cnt: tf.Variable(shape=(action_count), dtype=tf.int32)
            count of how many times an action was taken
    """

    def __init__(
        self,
        action_spec: BoundedTensorSpec,
        name: str = "ActionCountMetric",
    ):
        """
        Args:
            action_spec: discrete! BoundedTensorSpec
                discrete bounded TensorSpec specifiyng possible actions,
                it is used to extract number and dtype of actions

            name: optional, name of the metric
        """
        super(tf_metric.TFStepMetric, self).__init__(name)
        self.action_count = get_num_actions_from_tensor_spec(action_spec)

        self.cnt = create_variable(
            initial_value=0,
            dtype=tf.int32,
            shape=(self.action_count),
            name="action count",
        )

    def reset(self):
        """
        Reset the metric.

        Reset removes all calculated values.
        """
        self.cnt.assign(tf.zeros_like(self.cnt))

    @tf.function
    def call(self, t: Trajectory) -> Trajectory:
        """Accumulates statistics from the provided trajectory.

        Each call expects a `trajectory` based on a new batch of data.

        Args:
            t: tf_agents.trajectories.trajectory.Trajectory
                trajectory to use for metric calculation
        Returns:
            The same trajectory that was passed as an argument
        """
        cnts = tf.math.bincount(t.action, minlength=self.action_count)
        self.cnt.assign(cnts)
        return t

    def result(self) -> tf.Tensor:
        """Calculates the final action counts.

        Returns:
            tf.Tensor(shape=(action_count), dtype=tf.int32) with final value of the metric
        """
        return self.cnt


class RMSEMetric(tf_metric.TFStepMetric):
    """RMSE metric implementation for TF agents

    Attributes:
        action_count: int
            how many possible actions are there
        action_spec: discrete! BoundedTensorSpec
                discrete bounded TensorSpec specifiyng possible actions
        rmse: tf.Variable(shape=(action_count), dtype=tf.float64)
            rmse value of each action on current batch
        type: str
            which field of policy info should be used as prediction
    """

    def __init__(
        self,
        action_spec: BoundedTensorSpec,
        type: str = "predicted_rewards_mean",
        name: str = "RMSEMetric",
    ):
        """
        Args:
            action_spec: discrete! BoundedTensorSpec
                discrete bounded TensorSpec specifiyng possible actions,
                it is used to extract number of actions and dtype of actions
            type: str
                which field of policy info should be used as prediction,
                possible values are [predicted_rewards_mean, predicted_rewards_optimistic]
            name: optional, name of the metric
        """
        super(tf_metric.TFStepMetric, self).__init__(name)
        self.action_spec = action_spec
        self.action_count = get_num_actions_from_tensor_spec(action_spec)

        if type not in ["predicted_rewards_mean", "predicted_rewards_optimistic"]:
            raise ValueError(
                "param `type` must be one of the following values: [predicted_rewards_mean, predicted_rewards_optimistic]"
            )

        self.type = type

        self.rmse = create_variable(
            initial_value=0,
            dtype=tf.float64,
            shape=(self.action_count),
            name="rmse",
        )

    def reset(self):
        """
        Reset the metric.

        Reset removes all calculated values.
        """
        self.rmse.assign(tf.zeros_like(self.rmse))

    @tf.function
    def call(self, t: Trajectory) -> Trajectory:
        """Accumulates statistics from the provided trajectory.

        Each call expects a `trajectory` based on a new batch of data.
        Each `trajectory` is expected to contain nonempty field `policy_info.predicted_rewards_mean`,

        Args:
            t: tf_agents.trajectories.trajectory.Trajectory
                trajectory to use for metric calculation

        Returns:
            The same trajectory that was passed as an argument
        """
        pred = tf.cast(getattr(t.policy_info, self.type), tf.float64)
        cnts = tf.math.bincount(t.action, minlength=self.action_count, dtype=tf.float64)
        acts = tf.range(self.action_count)

        def calc_sse(action):
            action = tf.cast(action, dtype=self.action_spec.dtype)
            mask = t.action == action
            pred_masked = tf.boolean_mask(
                pred[:, action],
                mask,
            )
            rew_masked = tf.boolean_mask(
                t.reward,
                mask,
            )
            rew_masked = tf.cast(rew_masked, dtype=pred_masked.dtype)

            return tf.reduce_sum((rew_masked - pred_masked) ** 2, axis=0)

        sse = tf.map_fn(calc_sse, tf.cast(acts, dtype=tf.float64))
        rmse = tf.cast(((sse / cnts) ** (1 / 2)), tf.float64)
        self.rmse.assign(
            tf.where(tf.math.is_nan(rmse), tf.constant(0, dtype=tf.float64), rmse)
        )
        return t

    def result(self) -> tf.Tensor:
        """Provides the RMSE value

        Returns:
            tf.Tensor(shape=(action_count), dtype=float64) with final value of the metric
        """
        return self.rmse
