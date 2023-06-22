from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess


from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from RL_agent import RL_agent
from Global_parameters import gp

rl_agent = RL_agent()

def get_eval_metrics():
  rl_agent.evaluator.run()
  results = {}
  for metric in rl_agent.evaluator.metrics:
    results[metric.name] = metric.result()
  return results

def log_eval_metrics(step, metrics):
    eval_results = (', ').join(
      '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
    print('step = {0}: {1}'.format(step, eval_results))
    print('eval_step = {0}: eval_handovers = {1}: eval_av_return = {2}: eval_reward = {3}: eval_action = {4}'.format(step,
                gp.eval_handovers_count, metrics["AverageReturn"], gp.eval_reward_, gp.eval_action__))
    myoutput = open(gp.file_name, 'a')
    subprocess.run(["echo", 'eval_step = {0}: eval_results = {1}: eval_handovers = {2}: eval_av_return = {3}: eval_reward = {4}: eval_action = {5}:eval_total_throughputs = {6}: eval_total_rsrqs = {7}: eval_cell_throughputs = {8}: eval_cell_rsrqss = {9}\n'.format(step,
      eval_results, gp.eval_handovers_count, metrics["AverageReturn"], gp.eval_reward_, gp.eval_action__, gp.eval_throughput_to_save, gp.eval_rsrq_to_save, gp.eval_cell_throughputs, gp.eval_cell_rsrqs)], stdout=myoutput)
    

def get_eval2_metrics():
  rl_agent.evaluator2.run()
  results = {}
  for metric in rl_agent.evaluator2.metrics:
    results[metric.name] = metric.result()
  return results

def log_eval2_metrics(step, metrics):
    eval2_results = (', ').join(
      '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
    print('step = {0}: {1}'.format(step, eval2_results))
    print('eval2_step = {0}: eval2_handovers = {1}: eval2_av_return = {2}: eval2_reward = {3}: eval2_action = {4}'.format(step,
                gp.eval2_handovers_count, metrics["AverageReturn"], gp.eval2_reward_, gp.eval2_action__))
    myoutput = open(gp.file_name, 'a')
    # print('CALL ENV')
    subprocess.run(["echo", 'eval2_step = {0}: eval2_results = {1}: eval2_handovers = {2}: eval2_av_return = {3}: eval2_reward = {4}: eval2_action = {5}: eval2_total_throughputs = {6}: eval2_total_rsrqs = {7}: eval2_cell_throughputs = {8}: eval2_cell_rsrqss = {9}\n'.format(step,
      eval2_results, gp.eval2_handovers_count, metrics["AverageReturn"], gp.eval2_reward_, gp.eval2_action__, gp.eval2_throughput_to_save, gp.eval2_rsrq_to_save, gp.eval2_cell_throughputs, gp.eval2_cell_rsrqs)], stdout=myoutput)
    
rl_agent.tf_agent.train_step_counter.assign(0)
steps = []
loss = []
handovers = []
rewards = []
av_return = []
av_return2 = []

metrics = get_eval_metrics()
avg_return = get_eval_metrics()["AverageReturn"]
returns = [avg_return]

metrics2 = get_eval2_metrics()
avg_return2 = get_eval2_metrics()["AverageReturn"]
returns2 = [avg_return2]

for _ in range(gp.num_iterations):
    # Training.
    rl_agent.collector.run()
    loss_info = rl_agent.agent_learner.run(iterations=1)

    step = rl_agent.agent_learner.train_step_numpy

    if gp.eval_interval and step % gp.eval_interval == 0:
        metrics = get_eval_metrics()
        log_eval_metrics(step, metrics)
        returns.append(metrics["AverageReturn"])

    if gp.eval2_interval and step % gp.eval2_interval == 0:
        metrics2 = get_eval2_metrics()
        log_eval2_metrics(step, metrics2)
        returns2.append(metrics2["AverageReturn"])

    if gp.log_interval and step % gp.log_interval == 0:
        print(gp.handovers_count, gp.throughput_to_save)
        steps.append(step)
        loss.append(loss_info.loss.numpy())
        handovers.append(gp.handovers_count)
        av_return.append(metrics["AverageReturn"])
        av_return2.append(metrics2["AverageReturn"])
        rewards.append(gp.reward_)
        print('step = {0}: loss = {1}: handovers = {2}: av_return = {3}: reward = {4}: action = {5}'.format(step, loss_info.loss.numpy(), gp.handovers_count, metrics["AverageReturn"], gp.reward_, gp.action__))
        myoutput = open(gp.file_name, 'a')
        # print('CALL ENV')
        subprocess.run(["echo", 'step = {0}: loss = {1}: handovers = {2}: av_return = {3}: av_return2 = {4}: reward = {5}: action = {6}: total_throughputs = {7}: total_rsrqs = {8}: cell_throughputs = {9}: cell_rsrqs = {10}: max_speed = {11}: min_speed = {12}: duration = {13}: count = {14}\n'.format(step, loss_info.loss.numpy(), gp.handovers_count, metrics["AverageReturn"], metrics2["AverageReturn"], gp.reward_, gp.action__, gp.throughput_to_save, gp.rsrq_to_save, gp.cell_throughputs, gp.cell_rsrqs, gp.max_speed, gp.min_speed, gp.duration, gp.ues)], stdout=myoutput)


rl_agent.rb_observer.close()
rl_agent.reverb_server.stop()