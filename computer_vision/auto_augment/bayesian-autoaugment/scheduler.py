# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:51:56 2019

The scheduler of Bayesian Optimized

This program uses Bayesian Optimized to automatically find the best data augmentation method.

@author: zyb_as
"""
import sys
import time
import skopt
from tcp_communicate import Command_Sender, Command_Listener
from augmentation import *


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--base_estimator', type=str, default='RF',
                    help='the base agent model to approximate the mapping between augment polices and model performance. Can choose RF, GBRT and ET.')
parser.add_argument('--configure_path', type=str, default='configure.txt',
                    help='path of configure file.')
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--mode', type=str, default='run',
                    help='execute mode: run or debug')
args = parser.parse_args()


import pdb


def get_optimizer(rand_init_num = 20):

    aug_name_list = get_augment_name_list()
    
    # define the skopt optimizer
    opt = skopt.Optimizer(
                [
                    skopt.space.Categorical(aug_name_list, name="A_aug_type"),
                    skopt.space.Real(0.0, 1.0, name="A_aug_probability"),
                    skopt.space.Real(0.0, 1.0, name="A_aug_magnitude"),
                    skopt.space.Categorical(aug_name_list, name="B_aug_type"),
                    skopt.space.Real(0.0, 1.0, name="B_aug_probability"),
                    skopt.space.Real(0.0, 1.0, name="B_aug_magnitude"),
                ],
                n_initial_points=rand_init_num,
                base_estimator='RF',
                acq_func='EI',
                acq_optimizer='auto',
                random_state=int(time.time()),
            )
    return opt
    


def restore_history_record(record_path):
    history_record = []
    with open(record_path, 'r') as reader:
        for line in reader:
            items = line.rstrip().split('|')
            cost = float(items[0])
            polices = [items[1], float(items[2]), float(items[3]), items[4], float(items[5]), float(items[6])]
            history_record.append((cost, polices))
    return history_record



def scheduler(command_sender, command_listener, record_path, paras):
    
    # get the skopt optimizer
    opt = get_optimizer(rand_init_num = 20)

    # restore the history record file
    print("restore history record file: {}".format(record_path))
    if os.path.exists(record_path):
        history_record  = restore_history_record(record_path)
        for record in history_record:
            print(record)
            opt.tell(record[1], record[0])
    
    total_epochs = paras.epochs
    for i in range(total_epochs):
        print("\nepoch {} starting...".format(i))
        sys.stdout.flush()
        
        # ask a new hyperparams
        cur_hyperparams = opt.ask()

        cur_police = polices2str(cur_hyperparams)
        cur_police_list = [cur_police]

        send_record_list = command_sender.send_command(cur_police_list, mode=paras.mode)
        
        receive_msg_list = command_listener.receive_result(send_record_list)

        print("analysis receive msg...")
        sys.stdout.flush()
        cost_record_list = []
        polices_record_list = []
        for send_record in send_record_list:
            cmd_id = send_record[0]
            success_status = send_record[1]
            polices_str = send_record[2]
            if success_status == False:
                continue
            
            for msg in receive_msg_list:
                items = msg.split('|')
                if items[0] == 'success' and int(items[1]) == cmd_id:
                    # find a success result
                    cost = float(items[2]) 
         
                    print("cost: {}, polices: {}".format(cost, polices_str)) 
                    sys.stdout.flush()
                    cur_hyperparams = str2polices(polices_str)

                    cost_record_list.append(cost)
                    polices_record_list.append(cur_hyperparams)
                    
                    # tell the optimizer the cost
                    opt.tell(cur_hyperparams, cost)
                    break

        print("save polices search result in record file")
        sys.stdout.flush()
        with open(record_path, 'a+') as writer:
            for idx, cost in enumerate(cost_record_list):
                write_line = str(cost) + '|'
                hyperparams_pair = polices_record_list[idx]
                write_line += hyperparams_pair[0] + '|' + str(hyperparams_pair[1]) + '|' + str(hyperparams_pair[2]) + '|'
                write_line += hyperparams_pair[3] + '|' + str(hyperparams_pair[4]) + '|' + str(hyperparams_pair[5]) + '\n'
                writer.write(write_line)


    print("\ntell all the evaluator to stop")
    sys.stdout.flush()
    command_sender.send_stop_command()
    
            




def parallel_scheduler1(command_sender, command_listener, parallel_level, record_path, paras):


    # get the skopt optimizer
    opt = get_optimizer(rand_init_num = 160)

    # restore the history record file
    print("restore history record file: {}".format(record_path))
    if os.path.exists(record_path):
        history_record  = restore_history_record(record_path)
        for record in history_record:
            print(record)
            opt.tell(record[1], record[0])
    
    total_epochs = paras.epochs
    for i in range(total_epochs):
        print("\nepoch {} starting...".format(i))
        sys.stdout.flush()
        
        # ask for a list of new hyperparams
        new_hyperparams_list = opt.ask(parallel_level)

        # convert data augment polices to str
        cur_police_list = []
        for new_hyperparams in new_hyperparams_list:
            cur_police_str = polices2str(new_hyperparams)
            cur_police_list.append(cur_police_str)

        send_record_list = command_sender.send_command(cur_police_list, mode=paras.mode)
        
        receive_msg_list = command_listener.receive_result(send_record_list)

        print("analysis receive msg...")
        sys.stdout.flush()
        cost_record_list = []
        polices_record_list = []
        for send_record in send_record_list:
            cmd_id = send_record[0]
            success_status = send_record[1]
            polices_str = send_record[2]
            if success_status == False:
                continue
            
            for msg in receive_msg_list:
                items = msg.split('|')
                if items[0] == 'success' and int(items[1]) == cmd_id:
                    # find a success result
                    cost = float(items[2]) 
         
                    print("cost: {}, polices: {}".format(cost, polices_str)) 
                    sys.stdout.flush()
                    cur_hyperparams = str2polices(polices_str)

                    cost_record_list.append(cost)
                    polices_record_list.append(cur_hyperparams)
                    
                    # tell the optimizer the cost
                    opt.tell(cur_hyperparams, cost)
                    break

        print("save polices search result in record file")
        sys.stdout.flush()
        with open(record_path, 'a+') as writer:
            for idx, cost in enumerate(cost_record_list):
                write_line = str(cost) + '|'
                hyperparams_pair = polices_record_list[idx]
                write_line += hyperparams_pair[0] + '|' + str(hyperparams_pair[1]) + '|' + str(hyperparams_pair[2]) + '|'
                write_line += hyperparams_pair[3] + '|' + str(hyperparams_pair[4]) + '|' + str(hyperparams_pair[5]) + '\n'
                writer.write(write_line)

    print("\ntell all the evaluator to stop")
    sys.stdout.flush()
    command_sender.send_stop_command()





def parallel_scheduler2(command_sender, command_listener, parallel_level, record_path, paras):


    # get the skopt optimizer
    opt = get_optimizer(rand_init_num = 20)
    time.sleep(1)
    rand_opt = get_optimizer(rand_init_num = 1e10)

    # restore the history record file
    print("restore history record file: {}".format(record_path))
    if os.path.exists(record_path):
        history_record  = restore_history_record(record_path)
        for record in history_record:
            print(record)
            opt.tell(record[1], record[0])
    
    total_epochs = paras.epochs
    for i in range(total_epochs):
        print("\nepoch {} starting...".format(i))
        sys.stdout.flush()
        
        # ask for a list of new hyperparams
        rand_hyperparams_list = rand_opt.ask(parallel_level - 1)
        new_hyperparams = opt.ask()
        new_hyperparams_list = rand_hyperparams_list
        new_hyperparams_list.append(new_hyperparams)

        # convert data augment polices to str
        cur_police_list = []
        for new_hyperparams in new_hyperparams_list:
            cur_police_str = polices2str(new_hyperparams)
            cur_police_list.append(cur_police_str)

        send_record_list = command_sender.send_command(cur_police_list, mode=paras.mode)
        
        receive_msg_list = command_listener.receive_result(send_record_list)

        print("analysis receive msg...")
        sys.stdout.flush()
        cost_record_list = []
        polices_record_list = []
        for send_record in send_record_list:
            cmd_id = send_record[0]
            success_status = send_record[1]
            polices_str = send_record[2]
            if success_status == False:
                continue
            
            for msg in receive_msg_list:
                items = msg.split('|')
                if items[0] == 'success' and int(items[1]) == cmd_id:
                    # find a success result
                    cost = float(items[2]) 
         
                    print("cost: {}, polices: {}".format(cost, polices_str)) 
                    sys.stdout.flush()
                    cur_hyperparams = str2polices(polices_str)

                    cost_record_list.append(cost)
                    polices_record_list.append(cur_hyperparams)
                    
                    # tell the optimizer the cost
                    rand_opt.tell(cur_hyperparams, cost)
                    opt.tell(cur_hyperparams, cost)
                    break

        print("save polices search result in record file")
        sys.stdout.flush()
        with open(record_path, 'a+') as writer:
            for idx, cost in enumerate(cost_record_list):
                write_line = str(cost) + '|'
                hyperparams_pair = polices_record_list[idx]
                write_line += hyperparams_pair[0] + '|' + str(hyperparams_pair[1]) + '|' + str(hyperparams_pair[2]) + '|'
                write_line += hyperparams_pair[3] + '|' + str(hyperparams_pair[4]) + '|' + str(hyperparams_pair[5]) + '\n'
                writer.write(write_line)

    print("\ntell all the evaluator to stop")
    sys.stdout.flush()
    command_sender.send_stop_command()
    



def load_configure(configure_path):
    server_info = None
    client_info_list = []
    with open(configure_path, 'r') as reader:
        for line in reader:
            items = line.rstrip().split('|')
            status = items[0]
            if status.startswith('#'):
                continue
            if status == 'Server':
                ip = items[1].split(':')[1]
                port = items[2].split(':')[1]
                record_path = items[3].split(':')[1][1:-1]
                server_info = (ip, port, record_path)
            elif status == 'Client':
                ip = items[1].split(':')[1]
                port = items[2].split(':')[1]
                gpu = items[3].split(':')[1]
                client_info = (ip, port, gpu)
                client_info_list.append(client_info)
            else:
                raise RuntimeError("parse configure.py error")

    print("server info:")
    print(server_info)
    print("\nclient info list:")
    for client_info in client_info_list:
        print(client_info)
    print(" ")
    return server_info, client_info_list
                


if __name__ == "__main__":
    # load configure file
    server_info, client_info_list = load_configure(args.configure_path)

    parallel_level = len(client_info_list) 
    print("parallel level: {}".format(parallel_level))
    sys.stdout.flush()
    assert parallel_level >= 1
    
    record_path = server_info[2]

    command_sender = Command_Sender(client_info_list)
    command_listener = Command_Listener(server_info)

    if parallel_level == 1:
        # normal mode
        scheduler(command_sender, command_listener, record_path, args)
    else:
        # parallel mode
        parallel_scheduler1(command_sender, command_listener, parallel_level, record_path, args)
