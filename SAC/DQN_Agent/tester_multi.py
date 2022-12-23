import os 
import sys
sys.dont_write_bytecode = True
os.chdir(sys.path[0])

import numpy as np
import agent_torch_trans as agent
import environment as env
#import tensorflow as tf
import parameters.setup as setup
from parameters import name_2_model

def get_checkpoints(model_name, model_ids):
    checkpoints = []
    if model_ids:
        for idx in model_ids:
            file_name = model_name + f'_{idx}.pt'
            model_path = os.path.join('models', model_name, file_name)
            checkpoints.append(model_path)
    else:
        checkpoints.append(os.path.join('models', model_name, model_name+'.pt'))
    return checkpoints

needs_to_run = {
                # 'DriveDQN_cnn_1f_Falseres': [True, None],
                # 'DriveDQN_simple_fusion2_gru_4f_Falseres': [True, [1900, 2000, 2100, 2900]],
                # 'DriveDQN_simple_fusion2_gru_8f_Falseres': [True, [1900, 2000, 2100, 2900]],
                # 'DriveDQN_simple_fusion2_gru_16f_Falseres': [True, [2000, 2100, 2200, 2900]],
                # 'DriveDQN_simple_fusion2_gru_8f_Trueres': [True, [2700, 2800, 2900]],
                # 'DriveDQN_simple_fusion2_gru_16f_Trueres': [True, [2700, 2800, 2900]],
                # 'DriveDQN_simple_fusion2_lstm_4f_Falseres': [True, [2200,2300,2400, 2900]],
                # 'DriveDQN_simple_fusion2_lstm_8f_Falseres': [True, [1900, 2000, 2100, 2900]],
                # 'DriveDQN_simple_fusion2_lstm_16f_Falseres': [True, [1800, 1900, 2000, 2900]],
                # 'DriveDQN_simple_fusion2_lstm_16f_Trueres': [True, [2700, 2800, 2900]],
                # 'DriveDQN_simple_fusion2_lstm_8f_Trueres': [True, [2700, 2800, 2900]],
                # 'DriveDQN_simple_fusion2_single_act_dec_4f_Falseres': [True, [1600,1700,1800,2900]],
                # 'DriveDQN_simple_fusion2_single_act_dec_8f_Falseres': [True, [800,900,2900]],
                'DriveDQN_simple_fusion2_single_act_dec_16f_Falseres': [True, [2700,2800,2900]],
                # 'DriveDQN_simple_fusion2_single_act_dec_8f_Trueres': [True, [2300,2400,2500,2900]],
                # 'DriveDQN_simple_fusion2_single_act_dec_16f_Trueres': [True, []],
                }

def eval_model_checkpoint(model_path, model_name, n_frames, residual):
    rewards_hist = []
    # Prepare env and agent
    car_racing_dict = setup.setup_dict_trans['car racing']
    agent_racing_dict = setup.setup_dict_trans['agent']

    agent_racing_dict['architecture'] = name_2_model[model_name]
    agent_racing_dict['architecture_args'].n_frames = n_frames
    agent_racing_dict['architecture_args'].residual = residual

    if model_name == 'DriveDQN_cnn':
        agent_racing_dict['process_state'] = False
        car_racing_dict['process_state'] = True
    else: 
        agent_racing_dict['process_state'] = True
        car_racing_dict['process_state'] = False
    # Build env
    environment = env.CarRacing(**car_racing_dict, render=RENDER)
    control = agent.DQN_Agent(environment=environment, **agent_racing_dict)
    # load model
    control.load(model_path)
    for i in range(NUM_EPS):
        # seed env
        environment.seed = [i]
        mean, std, rewards = control.test(1, RENDER)
        print(f"Model: {model_name} | N_frames: {n_frames} | Residual {residual} | Run: {i+1}/{NUM_EPS} | Reward: {rewards[0]}")
        rewards_hist.append(rewards[0])

    del environment
    del control 

    return np.mean(rewards_hist), np.std(rewards_hist)

def write_stats(path, model_ids, means, stds):
    print("Writing to ", path)
    with open(path, 'w') as fp:
        fp.write('model_id,mean reward,std\n')        
        for i in range(len(means)):
            model_id = str(model_ids[i]) if model_ids else str(model_ids)
            fp.write(f'{model_id},{str(means[i])},{str(stds[i])}\n')   

NUM_EPS = 100
RENDER = False
REMOTE = True
if __name__ == '__main__':
    for model_name, spec in needs_to_run.items():
        should_run, model_ids = spec 
        if should_run:
            print("Evaluating model: ", model_name)
            checkpoints = get_checkpoints(model_name, model_ids)
            # Run Model Checkpoint
            means, stds = [], []
            for checkpoint in checkpoints:
                n_frames = int(model_name.split('_')[-2][:-1])
                residual = model_name.split('_')[-1][0] == 'T'
                architecure_name = '_'.join(model_name.split('_')[:-2])
                mean, std = eval_model_checkpoint(checkpoint, architecure_name, n_frames, residual)
                means.append(mean)
                stds.append(std)
            write_stats(os.path.join('evaluation','test_runs', model_name+'.csv'), model_ids, means, stds)
        else: 
            print("Skipping model: ", model_name)