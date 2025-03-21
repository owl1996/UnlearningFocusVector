#!/bin/sh
#OAR -n my_array_jobs
#OAR -l /nodes=1/core=1/gpu=1,walltime=01:00:00
#OAR -O output_%jobid%_%arrayid%.log
#OAR -E error_%jobid%_%arrayid%.err
#OAR -q production
#OAR -p cluster='cluster_name'
#OAR --array-param-file path/to/params.txt
#OAR -O oar_job.%jobid%.output
#OAR -E oar_job.%jobid%.error
set -x -v

/path/to/python_script.sh $*