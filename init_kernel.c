//#define ENV_SIZE %(env_size)d

/*
Kernel which initializes array with multipliers for calculating ligand absorption probabilities
*/
__kernel void init_mul_mat(__global float* mul_mat, __global float* adj_mat, __private float max_dist)
{
    size_t env_idx = get_global_id(0);
    for (int i = 0; i < ENV_SIZE; ++i)
    {
        float dist = adj_mat[env_idx * ENV_SIZE + i];
        if (dist > 0 && dist <= max_dist)
        {
            mul_mat[env_idx * ENV_SIZE + i] = 1.0f / dist;
        }
        else
        {
            mul_mat[env_idx * ENV_SIZE + i] = 0.0f;
        }
    }
}