'''
Main program file.

func:
- train_ppo
- train_td3
- test
- baseline
'''

import os, shutil
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandapower as pp
import pandapower.timeseries as ts
from pandapower.timeseries.data_sources.frame_data import DFData
from pandapower.timeseries.output_writer import OutputWriter

import utils
from cigre_mv_microgrid import create_cigre_mv_microgrid
from controllers.baseline_controller import RandomControl, SimpleControl
from controllers.td3_controller import TD3Agent
from controllers.ppo_controller import PPOAgent
from setting import *

def train_ppo(n_runs, n_epochs, start, train_length, pv_profile, wt_profile, load_profile, price_profile,
    sequence_model_type='none', noise_type='action'):
    # env
    assert(start >= 0 and start < pv_profile.shape[0])
    assert(train_length >= 0 and train_length <= pv_profile.shape[0] - start)
    time_steps = range(start, start + train_length)
    pv_ds = DFData(pv_profile.iloc[start: start+train_length])
    wt_ds = DFData(wt_profile.iloc[start: start+train_length])
    load_ds = DFData(load_profile.iloc[start: start+train_length])

    # history
    history_dir = os.path.join('.', 'history', 'train', 'PPO')
    if os.path.isdir(history_dir):
        shutil.rmtree(history_dir)

    # run
    ep_return_list = np.zeros((n_runs, n_epochs))
    ep_cost_list = np.zeros((n_runs, n_epochs))
    for run in range(n_runs):
        net, ids = create_cigre_mv_microgrid(pv_ds, wt_ds, load_ds)
        agent = PPOAgent(net, ids, pv_profile, wt_profile, load_profile, price_profile, sequence_model_type)

        best_cost = train_length * MAX_COST
        for epoch in range(n_epochs):
            agent.training = True
            ts.run_timeseries(net, time_steps=time_steps, continue_on_divergence=False)
            ep_return_list[run, epoch] = np.sum(agent.rewards)
            np.save(os.path.join('.', 'plot', 'ep_return_list.npy'), ep_return_list)
            ep_cost_list[run, epoch] = np.sum(agent.costs)
            np.save(os.path.join('.', 'plot', 'ep_cost_list.npy'), ep_cost_list)
            print(f'Run: {run + 1}, epoch: {epoch + 1}, return = {ep_return_list[run, epoch]:.3f}, cost = {ep_cost_list[run, epoch]:.3f}')

            # history & best models
            if epoch >= 20:
                cost = np.sum(agent.costs)
                if cost < best_cost or (epoch % 20 == 0):
                    # log history
                    dir = os.path.join(history_dir,str(run+1), 'best_avg_cost')
                    if not os.path.isdir(dir):
                        os.makedirs(dir)
                    pd.DataFrame(agent.history).to_csv(os.path.join(dir, f'[{epoch}]_cost{cost:.3f}.csv'))
                    
                    # save best cost model
                    if cost < best_cost:
                        agent.save(run)
                        best_cost = cost
            agent.reset()
        
    # plot
    print(f'Epoch return: \n {np.mean(ep_return_list, axis=0)}')
    print(f'Epoch cost: \n {np.mean(ep_cost_list, axis=0)}')
    utils.plot_ep_values(ep_return_list, train_length, n_epochs, ylabel='Return')
    utils.plot_ep_values(ep_cost_list, train_length, n_epochs, ylabel='Cost')


def train_td3(n_runs, n_epochs, start, train_length, pv_profile, wt_profile, load_profile, price_profile,
    verbose=True, sequence_model_type='rnn', use_pretrained_sequence_model=False, 
    noise_type='action', retrain=False, run=1):
    # env
    assert(start >= 0 and start < pv_profile.shape[0])
    assert(train_length >= 0 and train_length <= pv_profile.shape[0] - start)
    time_steps = range(start, start + train_length)
    pv_ds = DFData(pv_profile.iloc[start: start+train_length])
    wt_ds = DFData(wt_profile.iloc[start: start+train_length])
    load_ds = DFData(load_profile.iloc[start: start+train_length])

    # history
    history_dir = os.path.join('.', 'history', 'train', 'TD3')
    if os.path.isdir(history_dir):
        shutil.rmtree(history_dir)

    # run
    ep_return_list = np.zeros((n_runs, n_epochs))
    ep_cost_list = np.zeros((n_runs, n_epochs))
    for run in range(n_runs):
        net, ids = create_cigre_mv_microgrid(pv_ds, wt_ds, load_ds)
        agent = TD3Agent(net, ids, pv_profile, wt_profile, load_profile, price_profile,
            training=True, n_epochs=n_epochs,
            sequence_model_type=sequence_model_type, use_pretrained_sequence_model=use_pretrained_sequence_model, 
            buffer_size=BUFFER_SIZE, noise_type=noise_type, batch_size=BATCH_SIZE)
        if retrain:
            agent.load_models(run=run)

        # run
        best_cost = train_length * MAX_COST
        for epoch in range(n_epochs):
            # train
            agent.training = True
            ts.run_timeseries(net, time_steps=time_steps, verbose=verbose, continue_on_divergence=False)
            ep_return_list[run, epoch] = np.sum(agent.rewards)
            ep_cost_list[run, epoch] = np.sum(agent.costs)
            print(f'Run: {run + 1}, episode: {epoch + 1}, return = {ep_return_list[run, epoch]:.3f}, cost = {ep_cost_list[run, epoch]:.3f}')
            # agent.reset()

            # test
            # agent.training = False
            # ts.run_timeseries(net, time_steps=time_steps, verbose=verbose, continue_on_divergence=False)
            
            test_cost = np.sum(agent.costs)
            if (epoch >= 20) and ((epoch % 20 == 0) or test_cost < best_cost):
                # log history
                dir = os.path.join(history_dir,str(run+1), 'best_avg_cost')
                if not os.path.isdir(dir):
                    os.makedirs(dir)
                pd.DataFrame(agent.history).to_csv(os.path.join(dir, f'[{epoch}]_cost{ep_cost_list[run, epoch]:.3f}.csv'))
                
                # save best cost model
                # if test_cost < best_cost:
                #     agent.save_models(run=run+1)
                #     best_cost = test_cost
            agent.reset()        

    # plot
    print(f'Episode return: \n {np.mean(ep_return_list, axis=0)}')
    print(f'Episode cost: \n {np.mean(ep_cost_list, axis=0)}')
    utils.plot_ep_values(ep_return_list, train_length, n_epochs, ylabel='Return')
    utils.plot_ep_values(ep_cost_list, train_length, n_epochs, ylabel='Cost')

def test(n_runs, start, test_length, pv_profile, wt_profile, load_profile, price_profile, run, sequence_model_type='rnn', log=False, log_path=None):
    assert(start >= 0 and start < pv_profile.shape[0])
    assert(test_length >= 0 and test_length <= pv_profile.shape[0] - start)
    time_steps=range(start, start+test_length)
    
    # env
    pv_ds = DFData(pv_profile.iloc[start: start+test_length])
    wt_ds = DFData(wt_profile.iloc[start: start+test_length])
    load_ds = DFData(load_profile.iloc[start: start+test_length])
    net, ids = create_cigre_mv_microgrid(pv_ds, wt_ds, load_ds)

    # log pf results
    if log:
        n_runs = 1
        ow = OutputWriter(net, time_steps, output_path=log_path, output_file_type='.csv', csv_separator=',')
        ow.log_variable('res_sgen', 'p_mw')
        ow.log_variable('res_load', 'p_mw')
        ow.log_variable('res_storage', 'p_mw')
        ow.log_variable('res_trafo', 'p_lv_mw', index=[ids['trafo0']])

    # agent
    agent = PPOAgent(net, ids, pv_profile, wt_profile, load_profile, price_profile, sequence_model_type, training=False)
    agent.load(run)
    # agent = TD3Agent(net, ids, pv_profile, wt_profile, load_profile, price_profile,
    #         training=False, sequence_model_type=sequence_model_type)
    # agent.load_models(run=run)

    # run
    ep_cost_list = []
    for _ in range(n_runs):
        ts.run_timeseries(net, time_steps=time_steps, verbose=False, continue_on_divergence=False)
        ep_cost_list.append(np.sum(agent.costs))
        agent.reset()
    print(f'Avg cost = {np.mean(ep_cost_list)}')

def baseline(n_runs, start, test_length, pv_profile, wt_profile, load_profile, price_profile, Control, log=False, log_path=None):
    assert(start >= 0 and start < pv_profile.shape[0])
    assert(test_length >= 0 and test_length <= pv_profile.shape[0] - start)
    time_steps=range(start, start+test_length)
    
    # env
    pv_ds = DFData(pv_profile.iloc[start: start+test_length])
    wt_ds = DFData(wt_profile.iloc[start: start+test_length])
    load_ds = DFData(load_profile.iloc[start: start+test_length])
    net, ids = create_cigre_mv_microgrid(pv_ds, wt_ds, load_ds)

    # log pf results
    if log:
        n_runs = 1
        ow = OutputWriter(net, time_steps, output_path=log_path, output_file_type='.csv', csv_separator=',')
        ow.log_variable('res_sgen', 'p_mw')
        ow.log_variable('res_load', 'p_mw')
        ow.log_variable('res_storage', 'p_mw')
        ow.log_variable('res_trafo', 'p_lv_mw', index=[ids['trafo0']])

    # controller
    controller = Control(net, ids, price_profile=price_profile)

    # run
    ep_cost_list = []
    for _ in range(n_runs):
        ts.run_timeseries(net, time_steps=time_steps, continue_on_divergence=False, verbose=True)
        ep_cost_list.append(np.sum(controller.costs))
        controller.reset()
    print(f'Avg cost = {np.mean(ep_cost_list)}')

if __name__ == '__main__':
    # --- configurations ---
    logging.basicConfig(level=logging.INFO)
    algo = 'ppo'
    sequence_model_type = 'rnn' # ['none', 'conv1d', 'rnn', 'transformer']
    sequence_length = 1 if (sequence_model_type == 'none') else SEQ_LENGTH

    # train configs
    n_train_runs = 10
    n_epochs = 500
    train_start = 0
    train_length = 30 * 24
    noise_type = 'action' # ['action', 'param']
    use_pretrained_sequence_model = False

    # test configs
    n_test_runs = 1
    # test_start = train_start + train_length
    # test_length = 7 * 24
    test_start = train_start
    test_length = train_length
    log = True
    log_path = os.path.join('.', 'pf_res', algo, sequence_model_type)
    log_path_baseline = os.path.join('.', 'pf_res', 'baseline', 'simple')

    # --- profile ---
    pv_profile = pd.read_csv('./data/profile/pv_profile.csv')
    wt_profile = pd.read_csv('./data/profile/wt_profile.csv')
    load_profile = pd.read_csv('./data/profile/load_profile.csv')
    price_profile = pd.read_csv('./data/profile/price_profile.csv')

    # --- train, test ---
    train_ppo(n_train_runs, n_epochs, train_start, train_length, 
        pv_profile, wt_profile, load_profile, price_profile, sequence_model_type)

    # train_td3(n_runs=n_train_runs, n_epochs=n_epochs, start=train_start, train_length=train_length,
    #     pv_profile=pv_profile, wt_profile=wt_profile, load_profile=load_profile, price_profile=price_profile,
    #     sequence_model_type=sequence_model_type, use_pretrained_sequence_model=use_pretrained_sequence_model, noise_type=noise_type)

    # test(n_runs=n_test_runs, start=test_start, test_length=test_length, 
    #     pv_profile=pv_profile, wt_profile=wt_profile, load_profile=load_profile, price_profile=price_profile,
    #     run=0, sequence_model_type=sequence_model_type, log=log, log_path=log_path)

    # baseline(n_runs=n_test_runs, start=test_start, test_length=test_length, 
    #     pv_profile=pv_profile, wt_profile=wt_profile, load_profile=load_profile, price_profile=price_profile,
    #     Control=SimpleControl, 
    #     log=log, log_path=log_path_baseline)

    # --- plot pf results ---
    # utils.plot_pf_results(log_path, test_start, test_length)
    # utils.plot_pf_results(dir=log_path_baseline)