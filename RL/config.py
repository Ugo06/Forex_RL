import argparse


import argparse
import json
import os
import ast

def str2bool(v):
    print(type(v),v)
    return v.lower() in ("yes", "true", "t", "1")

def main(args):
    args.config_layer = ''.join(args.config_layer)
    args.config_layer = eval(args.config_layer)

    config = {
        "SAVE_DIR": args.save_dir,
        "RUN_ID": args.run_id,
        "NB_ACTION":args.nb_action,
        "WINDOW_SIZE": args.window_size,
        "EPISODE_SIZE": args.episode_size,
        "NB_EPISODE": args.nb_episode,
        "INITIAL_STEP": args.initial_step,
        "PAS":args.pas,
        "N_TRAIN": args.n_train,
        "N_TEST": args.n_test,
        "MODE": {
            'include_price': str2bool(args.include_price),
            'include_historic_position': str2bool(args.include_historic_position),
            'include_historic_action': str2bool(args.include_historic_action),
            'include_historic_wallet': str2bool(args.include_historic_wallet),
            'include_historic_orders': str2bool(args.include_historic_orders)
        },
        "WALLET":args.wallet,
        "REWARD_FUNCTION":args.reward_function,
        "ZETA":args.zeta,
        "BETA":args.beta,
        "SPLIT":args.split,
        "TYPE":args.type,
        "CONFIG_LAYER": args.config_layer,
        "EPSILON": args.epsilon,
        "EPSILON_MIN": args.epsilon_min,
        "EPSILON_DECAY": args.epsilon_min ** (1 / args.nb_episode) if args.epsilon_decay in 'default' else float(args.epsilon_decay),
        "BUFFER_SIZE": args.buffer_size,
        "GAMMA": args.gamma,
        "ALPHA": args.alpha,
        "BATCH_SIZE": args.batch_size,
        "ITER_SAVE_MODEL_SCORE": args.iter_save_model_score,
        "ITER_SAVE_TARGET_MODEL": args.iter_save_target_model,
        "ITER_TEST": args.iter_test,
        "FIGURE_TITLE": args.figure_title,
        "DATA_PATH": args.data_path,
        "MODEL_SAVE_PATH": os.path.join(args.save_dir, f"config_{args.run_id}", 'model.keras'),
        "SCORE_SAVE_PATH": os.path.join(args.save_dir, f"config_{args.run_id}", 'scores.npy'),
        "FIGURE_PATH": os.path.join(args.save_dir, f"config_{args.run_id}", 'figure.png')
    }
    
    run_dir = os.path.join(args.save_dir, f"config_{args.run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments from config.py
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--nb_action', type=int, default=3)
    parser.add_argument('--window_size', type=int, default=21)
    parser.add_argument('--episode_size', type=int, default=84)
    parser.add_argument('--nb_episode', type=int, default=200)
    parser.add_argument('--initial_step', type=None, default='random')
    parser.add_argument('--pas', type=int, default=1)
    parser.add_argument('--n_train', type=int, default=2)
    parser.add_argument('--n_test', type=int, default=1)
    parser.add_argument('--include_price', type=str, default='False')
    parser.add_argument('--include_historic_position', type=str, default='False')
    parser.add_argument('--include_historic_action', type=str, default='False')
    parser.add_argument('--include_historic_wallet', type=str, default='False')
    parser.add_argument('--include_historic_orders', type=str, default='False')
    parser.add_argument('--reward_function',type=str,default='default')
    parser.add_argument('--zeta',type=float,default=1)
    parser.add_argument('--beta',type=float,default=1)
    parser.add_argument('--split',type=int,efault=250)
    parser.add_argument('--wallet',type=int,default=0)
    parser.add_argument('--type', type=str, default='lstm')
    parser.add_argument('--config_layer', type=None, default=[64, 8])
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--epsilon_min', type=float, default=0.01)
    parser.add_argument('--epsilon_decay', type=None, default='default')
    parser.add_argument('--buffer_size', type=int, default=15000)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--iter_save_model_score', type=int, default=25)
    parser.add_argument('--iter_save_target_model', type=int, default=10)
    parser.add_argument('--iter_test', type=int, default=1)
    parser.add_argument('--figure_title', type=str, default='Values of portfolio function of episodes')
    parser.add_argument('--data_path', type=str, default='C:/Users/Ugo/Documents/AI/Forex_ML/RL/DATA/FAKE_DATA_TRAIN.csv')

    args = parser.parse_args()
    main(args)
