from wrappers.train_eval import train_and_eval
import sys
import yaml

if __name__ == '__main__':

    # eval params
    environment = sys.argv[1]
    sensors = sys.argv[2]
    entropy = float(sys.argv[3])
    multistep = int(sys.argv[4])
    if multistep == 0: multistep = None
    index_exp = int(sys.argv[5])

    fname = environment + '_' + sensors + '_' + str(entropy) + '_' + str(multistep) + '_' + str(index_exp)

    agent_type = 'base'

    results = train_and_eval(agent_type, sensors, total_timesteps_training=int(1e6),
                    n_multisteps = multistep, playground_name = environment, freq_eval=int(1e5), episodes_eval=5, entropy = entropy)


    log_fname = 'logs/' + fname + '.dat'

    with open(log_fname, 'w') as f:
        yaml.dump(results, f)
