import argparse
from parameters import (model_params, name_2_model)

import parameters.setup as setup

parser = argparse.ArgumentParser(
                    prog = 'Run CarRacing Experiment',
                    description = 'Specify setup parameters, model type and model parameters')

# EXPERIMENT PARAMETERS
parser.add_argument('--process_state', action='store_true', help='Disables state processing and switches to default env. framestack processing')
parser.add_argument('--rm_frame_skip', action='store_false', help='Disables frame skipping.')
parser.add_argument('--accel', action='store_true', help='Pushes temporal states onto gpu on the same time. is faster if gpu has enough memory')
parser.add_argument('--use_all_timesteps', action='store_true', help='Whether to use all timesteps for Q loss')
parser.add_argument('--explore_frame_limit', type=int, default=250_000, help='The limit which the epsilon should decay towards.')
# MODEL PARAMETERS
parser.add_argument('--model', type=str, required=True, help='name of the model to run.')
parser.add_argument('--n_frames', type=int, required=True)
parser.add_argument('--residual', action='store_true', help='Whether to add residual connection at the end')
args = parser.parse_args()


def parse_args():
    # SETUP ENVIRONMENT (Assuming contraints are satifised)
    errors = ''
    agent_setup = setup.setup_dict_trans['agent']
    carracing_setup = setup.setup_dict_trans['car racing']

    # process_state
    agent_setup['process_state'] = args.process_state
    carracing_setup['process_state'] = not args.process_state

    # rm_frame_skip
    carracing_setup['use_frame_skip'] = args.rm_frame_skip


    # use_all_timesteps
    agent_setup['use_all_timesteps'] = args.use_all_timesteps

    # explore_frame_limit
    agent_setup['learning_rate_drop_frame_limit'] = args.explore_frame_limit

    # model
    if args.model not in name_2_model:
        errors += f'{args.model} is not a valid model name.\n'
    else:
        agent_setup['architecture'] = name_2_model[args.model]

    # accel 
    model_params.accel = args.accel

    # n_frames
    model_params.n_frames = args.n_frames

    # residual
    model_params.residual = args.residual

    # assign model_params
    agent_setup['architecture_args'] = model_params

    if errors:
        raise ValueError(errors)
    else:
        return agent_setup, carracing_setup