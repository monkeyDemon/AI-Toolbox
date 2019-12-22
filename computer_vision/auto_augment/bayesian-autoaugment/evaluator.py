# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:51:56 2019

The evaluate client.

1. receive command from the server(schduler)
2. do the evaluate process(this step is time-consuming)
3. send the evaluate result to the server

@author: zyb_as
"""
import os
import sys
import time
import shutil
import socket
import random
import subprocess
from subprocess import Popen
from augmentation import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--client_id', type=int, default=0)
parser.add_argument('--configure_path', type=str, default='configure.txt',
                    help='path of configure file.')
args = parser.parse_args()



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
                server_info = (ip, port)
            elif status == 'Client':
                ip = items[1].split(':')[1]
                port = items[2].split(':')[1]
                gpu = items[3].split(':')[1]
                record_path = items[4].split(':')[1][1:-1]
                client_info = (ip, port, gpu, record_path)
                client_info_list.append(client_info)
            else:
                raise RuntimeError("parse configure.py error")
    return server_info, client_info_list


def get_cost(record_dir, cmd_id):
    cost_save_path = record_dir + '/cost.txt'
    with open(cost_save_path, 'r') as reader:
        for line in reader:
            items = line.rstrip().split('|')
            cur_cmd_id = int(items[0])
            if cur_cmd_id == cmd_id:
                cost = float(items[1])
                return cost
    raise RuntimeError("can not find the evaluate cost.")


def perform_evaluate_command(command, gpu, record_dir):
    eval_status = 'success' 
    eval_result = None
    exit_status = False

    items = command.split('|')
    cmd_type = items[0]

    if cmd_type == 'Debug':
        cmd_id = int(items[1])
        polices = items[2]

        print("evaluating...")
        sys.stdout.flush()
        time.sleep(10)
        eval_result = random.uniform(0, 1)
    elif cmd_type == 'Run':
        try:
            cmd_id = int(items[1])
            polices = items[2]

            #if os.path.exists(record_dir):
            #    shutil.rmtree(record_dir)
            #os.mkdir(record_dir)

            print("evaluating...")
            sys.stdout.flush()
            eval_cmd = "./evaluate.sh {} {} {} {}".format(cmd_id, gpu, record_dir, polices)
            # 调用方案1 
            os.system(eval_cmd) 
            
            # 调用方案2
            #p = Popen([eval_cmd],stdout=subprocess.PIPE, shell=True)
            #result = p.stdout.read().rstrip()
            
            # get the cost
            print("os.system")
            sys.stdout.flush()
            eval_result = get_cost(record_dir, cmd_id)
        except:
            print("ERROR!!! error when evaluate")
            sys.stdout.flush()
            eval_status = 'false'
    elif cmd_type == 'Stop':
        exit_status = True 
        return exit_status, None
    else:
        raise RuntimeError("get wrong command")

    eval_result_str = eval_status + '|' + str(cmd_id) + '|' + str(eval_result)
    return exit_status, eval_result_str



if __name__ == "__main__":

    # load configure file
    server_info, client_info_list = load_configure(args.configure_path)

    server_ip = server_info[0]
    server_port = int(server_info[1])

    client_info = client_info_list[args.client_id]
    ip = client_info[0]
    port = int(client_info[1])
    gpu = client_info[2]
    record_dir = client_info[3]

    if os.path.exists(record_dir):
        shutil.rmtree(record_dir)
    os.mkdir(record_dir)

    # set socket
    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
    listen_sock.bind((ip, port))  
    listen_sock.listen(1)  
    while True:  
        # wait command
        try:
            connection, address = listen_sock.accept()
            connection.settimeout(5)
            command = connection.recv(1024)

            print("\nreceive msg:")
            print(command)
        except:
            print("error when listen or receive!!!")
            continue
        finally:
            connection.close()
            sys.stdout.flush()
        
        # perform the evaluate command
        exit_status, eval_result_str = perform_evaluate_command(command, gpu, record_dir)
       
        if exit_status == True:
            break 

        # send the evaluate result
        try:
            send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
            send_sock.connect((server_ip, server_port))
            time.sleep(0.1)
            send_sock.send(eval_result_str)

            print("send the evaluate result success")
        except:
            print("error when send evaluate result!!!")
            continue
        finally:
            send_sock.close()
            sys.stdout.flush()
    
    # close socket
    listen_sock.close()
