from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import reverb
import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils


from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from Global_parameters import gp
from RL_environment import HandoverEnv

checkingdir = '/tmp'

class RL_agent:
  
    def __init__(self):        
        self.collect_env = HandoverEnv()
        self.eval_env = HandoverEnv(eval1=True)
        self.eval_env2 = HandoverEnv(eval2=True)
        self.strategy = strategy_utils.get_strategy(tpu=False, use_gpu=False)
        self.create_RL_agent()
        self.replay_buffer_creator()
        self.collector_evaluator_creator()
        self.learner_creator()

    def create_RL_agent(self):
        observation_spec, action_spec, time_step_spec = (
            spec_utils.get_tensor_specs(self.collect_env))

        with self.strategy.scope():
            self.critic_net = critic_network.CriticNetwork(
                    (observation_spec, action_spec),
                    observation_fc_layer_params=None,
                    action_fc_layer_params=None,
                    joint_fc_layer_params=gp.critic_joint_fc_layer_params,
                    kernel_initializer='glorot_uniform',
                    last_kernel_initializer='glorot_uniform')

        with self.strategy.scope():
            self.actor_net = actor_distribution_network.ActorDistributionNetwork(
                observation_spec,
                action_spec,
                fc_layer_params=gp.actor_fc_layer_params,
                continuous_projection_net=(
                    tanh_normal_projection_network.TanhNormalProjectionNetwork))
        
        with self.strategy.scope():
            self.train_step = train_utils.create_train_step()
            self.tf_agent = sac_agent.SacAgent(
                    time_step_spec,
                    action_spec,
                    actor_network=self.actor_net,
                    critic_network=self.critic_net,
                    actor_optimizer=tf.keras.optimizers.Adam(
                        learning_rate=gp.actor_learning_rate),
                    critic_optimizer=tf.keras.optimizers.Adam(
                        learning_rate=gp.critic_learning_rate),
                    alpha_optimizer=tf.keras.optimizers.Adam(
                        learning_rate=gp.alpha_learning_rate),
                    target_update_tau=gp.target_update_tau,
                    target_update_period=gp.target_update_period,
                    td_errors_loss_fn=tf.math.squared_difference,
                    gamma=gp.gamma,
                    reward_scale_factor=gp.reward_scale_factor,
                    train_step_counter=self.train_step,
                    debug_summaries = True,
                    summarize_grads_and_vars = True,
            )
            self.tf_agent.initialize()

    def replay_buffer_creator(self):
        table_name = 'uniform_table'
        table = reverb.Table(
            table_name,
            max_size=gp.replay_buffer_capacity,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1))

        reverb_server = reverb.Server([table])

        reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
            self.tf_agent.collect_data_spec,
            sequence_length=2,
            table_name=table_name,
            local_server=reverb_server)

        dataset = reverb_replay.as_dataset(
            sample_batch_size=gp.batch_size, num_steps=2, num_parallel_calls=5).prefetch(50)
        self.experience_dataset_fn = lambda: dataset

        self.observer = reverb_utils.ReverbAddTrajectoryObserver(
        reverb_replay.py_client,
        table_name,
        sequence_length=2,
        stride_length=1)

    def collector_evaluator_creator(self):
        self.tf_target_policy = self.tf_agent.policy
        self.target_policy = py_tf_eager_policy.PyTFEagerPolicy(
        self.tf_target_policy, use_tf_function=True)

        self.tf_collect_policy = self.tf_agent.collect_policy
        self.collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
        self.tf_collect_policy, use_tf_function=True)

        env_step_metric = py_metrics.EnvironmentSteps()
        self.collector = actor.Actor(
        self.collect_env,
        self.collect_policy,
        self.train_step,
        steps_per_run=1,
        metrics=actor.collect_metrics(10),
        summary_dir=os.path.join(checkingdir, learner.TRAIN_DIR),
        observers=[self.observer, env_step_metric])
        self.collector.run()

        self.evaluator = actor.Actor(
        self.eval_env,
        self.target_policy,
        self.train_step,
        steps_per_run=1,
        metrics=actor.eval_metrics(1),
        summary_dir=os.path.join(checkingdir, 'eval'),
        )

        self.evaluator2 = actor.Actor(
        self.eval_env2,
        self.target_policy,
        self.train_step,
        steps_per_run=1,
        metrics=actor.eval_metrics(1),
        summary_dir=os.path.join(checkingdir, 'eval2'),
        )

    def learner_creator(self):
        saved_model_dir = os.path.join(checkingdir, learner.POLICY_SAVED_MODEL_DIR)

        learning_triggers = [
            triggers.PolicySavedModelTrigger(
                saved_model_dir,
                self.tf_agent,
                self.train_step,
                interval=gp.policy_save_interval),
            triggers.StepPerSecondLogTrigger(self.train_step, interval=1000),
        ]

        self.agent_learner = learner.Learner(
        checkingdir,
        self.train_step,
        self.tf_agent,
        self.experience_dataset_fn,
        triggers=learning_triggers,
        strategy=self.strategy)