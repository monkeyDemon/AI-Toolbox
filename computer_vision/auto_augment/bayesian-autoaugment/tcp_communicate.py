# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 19:05:56 2019

@author: zyb_as
"""
import sys
import time
import socket


class Command_Sender(object):
    def __init__(self, client_info_list):
        self.cmd_id = 0
        self.client_num = len(client_info_list)
        self.client_info_list = client_info_list

    def send_command(self, polices_list, mode='run'):
        # send parallel evaluate command to different devices
        record_list = []
        for i in range(self.client_num):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.cmd_id += 1
            client_info = self.client_info_list[i]
            ip = client_info[0]
            port = int(client_info[1])
            polices = polices_list[i]
            if mode == 'run':
                command = 'Run|' + str(self.cmd_id) + '|' + polices
            elif mode == 'debug':
                command = 'Debug|' + str(self.cmd_id) + '|' + polices
            else:
                raise RuntimeError("error use of parameter mode")

            record = [self.cmd_id, True, polices]

            # send message
            try:
                # send message
                sock.connect((ip, port))
                sock.send(command)
                sock.close()
                print("send command success. ip {}, port {}, cmd_id {}".format(ip, port, self.cmd_id))
                sys.stdout.flush()
            except:
                print("ERROR!!! send command failed. ip {}, port {}, cmd_id {}".format(ip, port, self.cmd_id))
                sys.stdout.flush()
                record[1] = False
            record_list.append(record)

        # return the send record info
        return record_list

    def send_stop_command(self):
        for i in range(self.client_num):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.cmd_id += 1
            client_info = self.client_info_list[i]
            ip = client_info[0]
            port = int(client_info[1])
            command = 'Stop|' + str(self.cmd_id) + '|' + 'None'

            # send message
            try:
                sock.connect((ip, port))
                sock.send(command)
                sock.close()
                print("send stop command success. ip {}, port {}, cmd_id {}".format(ip, port, self.cmd_id))
                sys.stdout.flush()
            except:
                print("ERROR!!! send stop command failed. ip {}, port {}, cmd_id {}".format(ip, port, self.cmd_id))
                sys.stdout.flush()



class Command_Listener(object):
    def __init__(self, server_info):
        self.server_info = server_info

    def receive_result(self, send_record_list):
        server_ip = self.server_info[0]
        server_port = int(self.server_info[1])

        client_num = 0
        for record in send_record_list:
            success_status = record[1]
            if success_status == True: 
                client_num += 1

        listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)  # allow multiplexing
        listen_sock.bind((server_ip, server_port))
        listen_sock.listen(client_num)
        receive_num = 0
        receive_msg_list = []
        while receive_num < client_num:
            connection, address = listen_sock.accept()
            try:
                connection.settimeout(5)
                msg = connection.recv(1024)

                # analysis msg
                print("receive msg {}".format(msg))
                receive_msg_list.append(msg) 
            except:
                print("ERROR!!! receive msg failed")
            finally:
                connection.close()
                sys.stdout.flush()
                receive_num += 1
        listen_sock.close()
        return receive_msg_list
