# pylint: disable=consider-using-with,subprocess-run-check

import argparse
import subprocess
from os import path as osp

TEMPLATE_FILE = osp.join(osp.dirname(__file__), "sbatch-template.sh")


CPU_LAUNCH = """
echo "STARTING WORKER CPU nodes"
srun --het-group={{CPU_HET_GROUP}} \\
  scripts/sing-exec \\
  ray start --address $ip_head --redis-password=$redis_password \\
  --block --resources='{"NO-GPU": 1}' &
"""

CPU_RESOURCES = """
### CPU workers
#SBATCH hetjob
#SBATCH --ntasks-per-node=1
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=48
#SBATCH --ntasks={{NUM_CPU_NODES}}
"""

GPU_LAUNCH = """
echo "STARTING WORKER GPU nodes"
srun --het-group={{GPU_HET_GROUP}} \\
  scripts/sing-exec \\
  ray start --address $ip_head --redis-password=$redis_password \\
  --block &
"""

GPU_RESOURCES = """
### GPU workers
#SBATCH hetjob
#SBATCH --partition=gengpu
#SBATCH --cpus-per-task={{GPU_CPUS_PER_TASK}}
#SBATCH --ntasks={{NUM_GPU_NODES}}
#SBATCH --gres=gpu:{{NUM_GPUS_PER_NODE}}
#SBATCH --mem-per-cpu=8G
"""


def replace(placeholder, replace_with):
    return text.replace(placeholder, str(replace_with))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="The job name and path to logging file (exp_name.log).",
    )
    parser.add_argument(
        "--num-cpu-nodes",
        type=int,
        default=1,
        help="Number of nodes to use.",
    )
    parser.add_argument(
        "--num-gpu-nodes",
        type=int,
        default=0,
        help="Number of nodes to use.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use in each node. (Default: 0)",
    )
    parser.add_argument(
        "--time",
        type=int,
        default=72,
        help="Time limit for the job (hours)",
    )
    parser.add_argument(
        "--load-env",
        type=str,
        default="",
        help="The script to load your environment, e.g. 'module load cuda/10.1'",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="slurm-logs",
        help="Work directory for logs.",
    )
    parser.add_argument(
        "--net-interface",
        type=str,
        default="ib0",
    )
    parser.add_argument(
        "--command",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--command-suffix",
        type=str,
        default="",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    args = parser.parse_args()

    # job_name = "{}_{}".format(
    #     args.exp_name, time.strftime("%m%d-%H%M", time.localtime())
    # )
    job_name = args.exp_name

    # ===== Modified the template script =====
    with open(TEMPLATE_FILE, "r") as f:
        text = f.read()

    # The head is also a full CPU node
    num_cpu_nodes = args.num_cpu_nodes - 1
    if num_cpu_nodes > 0:
        text = replace("{{CPU_RESOURCES}}", CPU_RESOURCES)
        text = replace("{{CPU_LAUNCH}}", CPU_LAUNCH)
        text = replace("{{CPU_HET_GROUP}}", 1)
    else:
        text = replace("{{CPU_RESOURCES}}", "")
        text = replace("{{CPU_LAUNCH}}", "")

    if args.num_gpu_nodes:
        text = replace("{{GPU_RESOURCES}}", GPU_RESOURCES)
        text = replace("{{GPU_LAUNCH}}", GPU_LAUNCH)
        if num_cpu_nodes > 0:
            text = replace("{{GPU_HET_GROUP}}", 2)
        else:
            text = replace("{{GPU_HET_GROUP}}", 1)
    else:
        text = replace("{{GPU_RESOURCES}}", "")
        text = replace("{{GPU_LAUNCH}}", "")

    text = replace("{{TIME}}", args.time)
    text = replace("{{WORK_DIR}}", args.work_dir)
    text = replace("{{JOB_NAME}}", job_name)
    text = replace("{{NUM_CPU_NODES}}", num_cpu_nodes)
    text = replace("{{NUM_GPU_NODES}}", args.num_gpu_nodes)
    text = replace("{{NUM_GPUS_PER_NODE}}", args.num_gpus)
    text = replace("{{GPU_CPUS_PER_TASK}}", args.num_gpus * 6)
    text = replace("{{COMMAND_PLACEHOLDER}}", args.command)
    text = replace("{{NET_INTERFACE}}", args.net_interface)
    text = replace("{{LOAD_ENV}}", args.load_env)
    text = replace("{{COMMAND_SUFFIX}}", args.command_suffix)

    script_file = f"{job_name}.sh"
    script_file = osp.join(args.work_dir, script_file)
    print(f"Saving to {script_file}")
    with open(script_file, "w") as f:
        f.write(text)

    if not args.dry_run:
        print("Start to submit job!")
        proc = subprocess.run(["sbatch", script_file], capture_output=True, shell=True)
        if proc.returncode == 0:
            print(f"Job submitted! Script file is at: <{script_file}>")
            print(f"Log file is at: <{job_name}.log>")
        else:
            print("Job submission failed")
            print("STDOUT:", proc.stdout.decode())
            print("STDERR:", proc.stderr.decode())
