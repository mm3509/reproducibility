# Call with
#   paper/deploy_azure.sh --ip=51.140.32.47
# or, with options:
#   paper/deploy_azure.sh --ip=51.140.32.47 --user=name -k=~/.ssh/azure

KEY="~/.ssh/azure"  # SSH key
RG=PB  # Resource Group

#!/bin/bash
for i in "$@"
do
    case $i in
	-k=*|--key=*)
	    KEY="${i#*=}"

	    ;;
	-i=*|--ip=*)
	    IP="${i#*=}"
	    ;;
	-r=*|--rg=*)
	    RG="${i#*=}"
	    ;;
	-u=*|--user=*)
	    USER="${i#*=}"
	    ;;
	*)
            # unknown option
	    ;;
    esac
done

if [ -z "$IP" ]; then
    NEW=true
else
    NEW=false
fi

echo KEY = ${KEY}
echo IP = ${IP}
echo NEW = ${NEW}
echo USER = ${USER}
echo RG = ${RG}

KEY_FLAG="-i $KEY"
echo KEYFLAG = ${KEY_FLAG}

if $NEW; then
    
    # Create new virtual machine
    az_cmd="az vm create
       --resource-group ${RG}
       --name MM
       --image microsoft-dsvm:linux-data-science-vm-ubuntu:linuxdsvmubuntu:19.04.00
       --size Standard_NV6_Promo
       --admin-username ${USER}
       --ssh-key-value ${KEY}.pub"

    # Parse the result to get the IP address
    result=$($az_cmd)
    echo $result
    ip_line=$(echo $result | grep -o '"publicIpAddress": "[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}"')
    echo $ip_line
    IP=$(echo $ip_line | grep -o '[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}')

    # Wait a bit before connecting
    sleep 60
fi

# Copy local files
ssh -o StrictHostKeyChecking=no ${KEY_FLAG} ${USER}@${IP} "rm -r repro2020; mkdir repro2020"
scp ${KEY_FLAG} -r * ${USER}@${IP}:repro2020

# reset the GPU
ssh ${KEY_FLAG} ${USER}@${IP} "sudo nvidia-smi -rac"

# This heredoc activates conda
ssh ${KEY_FLAG} ${USER}@${IP} bash -l <<HERE
cd repro2020
cp /data/anaconda/envs/py35/lib/python3.5/site-packages/keras/engine/training.py training.py_backup
cp /data/anaconda/envs/py35/lib/python3.5/site-packages/keras/engine/training_arrays.py training_arrays.py_backup
cp training.py /data/anaconda/envs/py35/lib/python3.5/site-packages/keras/engine/
cp training_arrays.py /data/anaconda/envs/py35/lib/python3.5/site-packages/keras/engine/
conda activate py35
python control.py
HERE
