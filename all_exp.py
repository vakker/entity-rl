from wrappers.train_eval import train_and_eval
import sys
import yaml

if __name__ == '__main__':

    for index_exp in [0, 1 ,2 ,3, 4, 5, 6, 7, 8, 9]:

        for environment in ['spg_endgoal_cue',
                            'spg_endgoal_9rooms',
                            'spg_dispenser_6rooms',
                            'spg_coinmaster_singleroom']:

            for sensor in ['rgb', 'rgb_depth rgb_touch', 'rgb_depth_touch']:

                for entropy in [0.05, 0.01, 0.005, 0.001]:

                    for multistep in [0, 2, 3 ,4]:

                        fname = environment + '_' + sensors + '_' + str(entropy) + '_' + str(multistep) + '_' + str(
                            index_exp)

                        agent_type = 'base'

                        results = train_and_eval(agent_type, sensors, total_timesteps_training=int(1e6),
                                                 n_multisteps=multistep, playground_name=environment, freq_eval=int(1e5),
                                                 episodes_eval=5, entropy=entropy)

                        log_fname = 'logs/' + fname + '.dat'

                        with open(log_fname, 'w') as f:
                            yaml.dump(results, f)

