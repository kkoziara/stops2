//#define MAX_LIG %(max_lig)d
//#define ROW_SIZE %(row_size)d
//#define ENV_SIZE %(env_size)d

__kernel void calculate_env_muls(__global float* env_muls, __global float* mul_mat, __global int* env_count)
 {
    size_t env_idx = get_global_id(0);
    float con_sum = 0.0f;
    float con_prd = 1.0f;

    for (int i = 0; i < ENV_SIZE; ++i)
    {
        con_sum += env_count[i] * mul_mat[env_idx * ENV_SIZE + i];
    }
    for (int i = 0; i < ENV_SIZE && con_prd > 0.0f; ++i)
    {

        con_prd *= pown(1.0f - mul_mat[env_idx * ENV_SIZE + i], env_count[i]);
    }
    env_muls[env_idx] = (1.0f / con_sum) * (1.0f - con_prd);
 }

/*
Kernel implementing ligand reception. Its result is pop_hit - array [1...POP_SIZE, 1...MAX_LIG] with values {0, 1},
where 1 at pop_hit[i,k] means cell i absorbed ligand k.
*/
__kernel void reception(__global int* pop_hit, __global float* env, __global float* mul_mat,
                        __global float* env_muls, __global int* env_map,
                        __global float* pop, __global int* receptors, __private int POP_SIZE,
                        __global ranluxcl_state_t *ranluxcltab)
{
    size_t env_idx = get_global_id(0);

    ranluxcl_state_t ranluxclstate;
    ranluxcl_download_seed(&ranluxclstate, ranluxcltab);

    for (int lig_idx = 0; lig_idx < MAX_LIG; ++lig_idx)
    {

        float lig = env[env_idx * MAX_LIG + lig_idx];

        int rec = receptors[lig_idx];

        if (lig > 0)
        {
            float diff = 0.0f;
            for (int i = 0; i < POP_SIZE; ++i)
            /* for each cell we check if it gets the ligand */
            {
                if (rec < 0 || pop[i * ROW_SIZE + rec] > 0)
                /* if cell can accept the ligand */
                {

                    float prob = lig * mul_mat[env_map[i] + ENV_SIZE * env_idx] * env_muls[lig_idx];
                    if (prob > ranluxcl32(&ranluxclstate).x)
                    /* if ligand is accepted */
                    {
                        pop_hit[lig_idx + i * MAX_LIG] = 1;
                        diff += 1.0f;
                    }
                }
            }
            env[env_idx * MAX_LIG + lig_idx] = max(0.0f, lig - diff);
        }
    }
    ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);
}

/*
Kernel updating population state according to pop_hit array calculated by reception kernel.
*/

__kernel void update_pop_with_reception(__global float* pop, __global int* pop_hit,
                                        __global int* rec_gene, __global float* bound)
{
    size_t cell_idx = get_global_id(0);
    for (int lig_idx = 0; lig_idx < MAX_LIG; ++lig_idx)
    {
        if (pop_hit[cell_idx * MAX_LIG + lig_idx] > 0)
        {
            int k = rec_gene[lig_idx];
            pop[cell_idx * ROW_SIZE + k] = min(pop[cell_idx * ROW_SIZE + k] + 1.0f, bound[k]);
        }
    }
}

/*
Kernel filling given buffer with specified value.
*/

__kernel void fill_buffer(__global int* buf, __private int x)
{
    buf[get_global_id(0)] = x;
}