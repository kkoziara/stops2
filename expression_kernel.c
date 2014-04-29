#define ROW_SIZE %(row_size)d

/*
Kernel implementing picking of gene to express next
*/
__kernel void choice(__global float *pop, __global const float *tokens,
__global const float *random, __global const float *bounds) {
    float sum = 0;
    int my_id = get_global_id(0);
    int block_start = ROW_SIZE * my_id;
    for (int i = 0; i < ROW_SIZE; i++) {
        sum += fabs(tokens[block_start + i]);
    }
    float cur_cum_sum = 0.0f;
    float my_rand = random[my_id];

    /* iterate over the vector until you exceed given random threshold */
    for (int i = 0; i < ROW_SIZE; i++) {
        cur_cum_sum += fabs(tokens[block_start + i]);
        if (cur_cum_sum / sum > my_rand) {
            float d = tokens[block_start + i];
            /* normalize influence on cell's vector and decide if it is enhancive or repressive */
            d = d > 0.0f ? 1.0f : -1.0f;
            /* apply the change wrt lower and upper limit */
            pop[block_start + i] = max(0.0f, min(bounds[i], pop[block_start + i] + d));
            break;
        }
    }
}