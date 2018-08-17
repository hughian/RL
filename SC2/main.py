import os
import sys
import math
import time
import random
import numpy as np
import pandas as pd
from absl import app,flags
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from a3c_agent import A3CAgent
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for training.")
flags.DEFINE_string("model", "fcn", "atari or fcn.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("screen",  32, "feature screen resolution.")
flags.DEFINE_integer("minimap", 32, "feature minimap resolution.")
FLAGS(sys.argv)

LOG_PATH = './log/Flat32/fcn/'
SNAPSHOT = './snapshot/Flat32/fcn/'
if not os.path.exists(LOG_PATH):
  os.makedirs(LOG_PATH)
if not os.path.exists(SNAPSHOT):
  os.makedirs(SNAPSHOT)

def run_loop(agents, env):
  """
  A run loop to have agents and an environment interact.
  ------
  params:
    agents: list of agents(one agent or two agents,depending on the type of map)
    env: environment
  rtype:
    None, but yeild: [last_timesteps[0], actions[0], timesteps[0]], is_done
      last_timesteps[0]: last observation
      actions[0]: the function id of action taken last step
      timesteps[0]: observation after action
      is_done: is last step or reaches max_frames 
  """
  start_time = time.time()

  try:
    while True: 
      timesteps = env.reset()
      for a in agents: a.reset()
      while True: #steps
        last_timesteps = timesteps
        actions = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]
        timesteps = env.step(actions)
        is_done = timesteps[0].last()
        yield [last_timesteps[0], actions[0], timesteps[0]], is_done
        if is_done:
          break
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds" % elapsed_time)

def main(unused_argv):
    '''
    main func doc
    '''
    with sc2_env.SC2Env(
        map_name="DefeatRoaches",
        players=[sc2_env.Agent(sc2_env.Race.terran)], #map DefeatRoaches only need one player
                # sc2_env.Bot(sc2_env.Race.random,
                #            sc2_env.Difficulty.very_hard)],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=FLAGS.screen, minimap=FLAGS.minimap),
            use_feature_units=True),
        step_mul=FLAGS.step_mul,
        game_steps_per_episode=0,
        visualize=False) as env:
        try:
            agent = A3CAgent(FLAGS.training,FLAGS.screen,FLAGS.minimap)
            agent.build_model(False,'/gpu:0', FLAGS.model)
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True #pylint: disable=E1101
            sess = tf.Session(config=config)
            summary_writer = tf.summary.FileWriter(LOG_PATH)
            agent.setup(env.observation_spec(), env.action_spec(),sess,summary_writer)
            agent.initialize()
            if not FLAGS.training:
                agent.load_model(SNAPSHOT)
            replay_buffer = []
            for record, done in run_loop([agent],env):
                if FLAGS.training:
                    replay_buffer.append(record) #store_transition
                    if done:
                        lr = FLAGS.learning_rate * (1 - 0.9 * agent.episodes / 1e4)
                        agent.update(replay_buffer,0.99,lr,agent.episodes)
                        replay_buffer.clear() #clear replay buffer for next run_loop,
                                              #actually we should fix the buffer size and 
                                              #overwrite the old record with new ones.
                        if agent.episodes % 1000 == 0:
                            agent.save_model(SNAPSHOT,agent.episodes)
                        if agent.episodes >= 1e4:
                            break
            #if needed, save replay here by doing
            #env.save_replay(map_name)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
  app.run(main)
