python dataset/d3plot_to_h5.py \
    --src /home/kong/datasets/barrier/fem/T_lok_F_shape_barrier_9_3_100km \
    --tmp /home/kong/datasets/barrier/tmp \
    --out /home/kong/datasets/barrier/h5/T_lok_F_shape_barrier_9_3_100km_50_2/output.h5 \
    --required-config configs/data/required_parts.config \
    --node-stride 50 \
    --frame-stride 2    