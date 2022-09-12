#!/bin/bash

#SBATCH --job-name={{JOB_NAME}}
#SBATCH --output=slurm-logs/%x_%j.log
#SBATCH --time=48:00:00

### Head node
#SBATCH --ntasks-per-node=1
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1

{{CPU_RESOURCES}}

{{GPU_RESOURCES}}

# Load modules or your own conda environment here
# module load pytorch/v1.4.0-gpu
# conda activate {{CONDA_ENV}}
{{LOAD_ENV}}

set -ex

################# DON NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=($nodes)

node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w $node_1 scripts/get-ip {{NET_INTERFACE}}) # making redis-address

if [[ $ip == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$ip"
  if [[ ${#ADDR[0]} > 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "We detect space in ip! You are using IPV6 address. We split the IPV4 address as $ip"
fi

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --het-group=0 \
  scripts/sing-exec \
  ray start --head --node-ip-address=$ip --port=6379 --redis-password=$redis_password \
  --block --resources='{"NO-GPU": 1}' &
sleep 10

{{CPU_LAUNCH}}
{{GPU_LAUNCH}}

sleep 30

##############################################################################################

#### call your code below
node_ip=$(scripts/get-ip {{NET_INTERFACE}})
scripts/sing-exec {{COMMAND_PLACEHOLDER}} {{COMMAND_SUFFIX}} --head-ip auto --node-ip $node_ip

echo "Cleaning up $SLURM_JOB_ID"
scripts/sing-exec ray stop &> /dev/null
wait

# scancel $SLURM_JOB_ID
