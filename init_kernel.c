#define POP_SIZE %(pop_size)d

/*
Kernel which initializes array with multipliers for calculating ligand absorption probabilities
*/
__kernel void init_mul_mat(__global float* mul_mat, __global float* adj_mat, __private float max_dist)
{
    size_t cell_idx = get_global_id(0);
    float sum = 0.0f;
    for (int i = 0; i < POP_SIZE; ++i)
    {
        float dist = adj_mat[cell_idx * POP_SIZE + i];
        if (dist > 0 && dist < max_dist)
        {
            float p = 1.0f / dist;
            mul_mat[cell_idx * POP_SIZE + i] = p;
            sum += p;
        }
        else
        {
            mul_mat[cell_idx * POP_SIZE + i] = 0.0f;
        }
    }
    if (sum > 0.0f)
    {
        for (int i = 0; i < POP_SIZE; ++i)
        {
            mul_mat[cell_idx * POP_SIZE + i] /= sum;
        }
    }
}