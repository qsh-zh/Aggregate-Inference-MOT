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

time_length=20

for num_grid in {5..5}
do 
    echo "num_grid $num_grid"
    for num_seed in {1..2}
    do
        run_python $num_grid $time_length itsbp $num_seed &
    done
    wait
done
