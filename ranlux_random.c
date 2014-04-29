#include <pyopencl-ranluxcl.cl>

/*
Kernel which must be run before using ranlux.
*/
__kernel void init_ranlux(__private uint ins,
    __global ranluxcl_state_t *ranluxcltab)
{
    ranluxcl_initialization(ins, ranluxcltab);
}

/*
Kernel for getting vector of random numbers using ranlux.
*/
__kernel void get_random_vector(__global float* output, __private uint out_size,
    __global ranluxcl_state_t *ranluxcltab)
{
    ranluxcl_state_t ranluxclstate;
    ranluxcl_download_seed(&ranluxclstate, ranluxcltab);

    unsigned long work_items = get_global_size(0);
    unsigned long idx = get_global_id(0)*4;
    while (idx + 4 <= out_size)
    {
        vstore4(ranluxcl32(&ranluxclstate),
            idx >> 2, output);
        idx += 4 * work_items;
    }

    float4 tail;
    if (idx < out_size)
    {
        tail = ranluxcl32(&ranluxclstate);
        output[idx] = tail.x;
        if (idx + 1 < out_size)
        {
            output[idx + 1] = tail.y;
            if (idx + 2 < out_size)
                output[idx + 2] = tail.z;
        }
    }

    ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);
}