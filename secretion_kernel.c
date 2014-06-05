//#define ROW_SIZE %(row_size)d
//#define MAX_LIG %(max_lig)d

/*
Kernel implementing ligands secretion.
*/
__kernel void secretion(__global float* pop, __global float* env, __global int* env_map, __global int* secr,
                        __private float MAX_CON, __private float SECR_AMOUNT)
{
    size_t cell_idx = get_global_id(0);
    int env_idx = env_map[cell_idx];
    for (int j = 0; j < MAX_LIG; ++j)
    {
        int k = secr[j];
        float gene_inf = pop[cell_idx * ROW_SIZE + k];
        if (gene_inf > 0)
        {
            /* secretion of ligand and gene expression drop */
            env[env_idx * MAX_LIG + j] = min(MAX_CON, env[env_idx * MAX_LIG + j] + SECR_AMOUNT); //TODO:race condition ? map/reduce ?
            pop[cell_idx * ROW_SIZE + k] = gene_inf - 1.0f;
        }
    }
}

/*
Leaking
*/
__kernel void leaking(__global float* env, __private float LEAK)
{
    size_t env_idx = get_global_id(0);
    for (int j = 0; j < MAX_LIG; ++j)
    {
        env[env_idx * MAX_LIG + j] = max(0.0f, env[env_idx * MAX_LIG + j] - LEAK);
    }
}