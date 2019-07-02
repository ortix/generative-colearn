#!/bin/bash

# $1 is the environment (pendulum, 2dof, 3dof)
# $2 is the learning agent (ga, nongan, knn) use knn only for pendulum

counter=1

while [ $counter -le 10 ]
do
  python3 main.py --learner=$2 --experiment=$1 --folder="$1 /$2"
  ((counter++))
done

python3 main.py --post-process --folder="$1 /$2"

echo "These results where for the $1 system, with the $2-model"
