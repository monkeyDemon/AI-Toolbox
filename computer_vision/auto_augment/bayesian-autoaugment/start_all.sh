#!/bin/bash


localip=$(ifconfig | grep 'inet'| grep -v '127.0.0.1' | cut -d: -f2 | awk '{ print $2}' )
echo $localip

while read line
do
    if [[ $line == \#* ]]
    then
        continue
    fi

    items=(${line//|/ })
    terminal=${items[0]}
    
    if [[ $terminal == 'Server' ]]
    then
        # Server
        echo $line
        log_path="record/sch.out"
        nohup python scheduler.py --epochs=40 --configure_path='configure.txt' > $log_path 2>&1 &

    else
        # Client

        ip=${items[1]}
        ip=(${ip//:/ })
        ip=${ip[1]}

        # judge if the ip is local
        is_local=$(echo $localip | grep "${ip}")
        if [[ "$is_local" != "" ]]
        then
            echo $line
            record_dir=${items[4]}
            record_dir=(${record_dir//:/ })
            record_dir=${record_dir[1]}
            record_dir=(${record_dir//\'/})

            client_id=(${record_dir//'client'/ })
            client_id=${client_id[1]}
            echo $client_id
            log_path=$record_dir".out"

            nohup python evaluator.py --client_id=${client_id} > ${log_path} 2>&1 &
        fi
    fi
done < configure.txt
