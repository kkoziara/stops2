#define ROW_SIZE %(row_size)d
#define POP_SIZE %(pop_size)d
#define MAX_LIG %(max_lig)d

/*
Kernel implementing ligands secretion.
*/
__kernel void secretion(__global float* pop, __global float* env, __global int* secr,
                        __private float MAX_CON, __private float LEAK, __private float SECR_AMOUNT)
{
    size_t cell_idx = get_global_id(0);
    for (int j = 0; j < MAX_LIG; ++j)
    {
        int k = secr[j];
        float gene_inf = pop[cell_idx * ROW_SIZE + k];
        float lig_con = env[cell_idx * MAX_LIG + j];
        if (gene_inf > 0)
        {
            /* secretion of ligand and gene expression drop */
            lig_con = min(MAX_CON, lig_con + SECR_AMOUNT);
            pop[cell_idx * ROW_SIZE + k] = gene_inf - 1.0f;
        }
        /* leaking */
        env[cell_idx * MAX_LIG + j] = max(0.0f, lig_con - LEAK);
    }

}