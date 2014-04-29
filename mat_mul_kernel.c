/*
Simple matrix multiplication.
*/

__kernel void mat_mul(__global float* C, __global float* A, __global float* B, __private unsigned int w)
{

   int idy = get_global_id(0);
   int idx = get_global_id(1);
   int wo = get_global_size(1);
   // value stores the element
   // that is computed by the thread
   float value = 0;
   for (int k = 0; k < w; ++k)
   {
      float eA = A[idy * w + k];
      float eB = B[k * wo + idx];
      value += eA * eB;
   }

   // Write the matrix to device memory each
   // thread writes one element
   C[idy * wo + idx] = value;
}