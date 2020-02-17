#!/bin/bash

run_python()
{ 
        python cmp_wrapper.py \
        -method $3 \
        --d_min $1 \
        --d_max $(( $1 + 1 ))\
        --num_node $2 \
        --graph_build $3 \
        --cmp_policy bp \
        --rand_seed $4 \
        --pkl_name "$3_$2_$1_$4"
}

for num_node in {2..20}
do 
    echo "num_node $num_node"
    for num_seed in {1..5}
    do
        run_python 10 $num_node cgm $num_seed &
    done
    wait
done
