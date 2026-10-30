#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#if defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#else
#define REQD_SUBGROUP_SIZE_128
#endif

#define OPWM 64
#define OPWN 64
#define CPWK 16
#define OPTM 4
#define OPTN 8

#define WG_M (OPWM / OPTM)
#define WG_N (OPWN / OPTN)
#define VEC_K (CPWK / 4)

REQD_SUBGROUP_SIZE_128
__kernel void mul_mat_f16_f32(
    const int M, const int N, const int K,
    __global const void* A_void, ulong A_offset,
    __global const void* B_void, ulong B_offset,
    __global       void* C_void, ulong C_offset) {

    __global const half*  A = (__global const half* )((__global const char*)A_void + A_offset);
    __global const float* B = (__global const float*)((__global const char*)B_void + B_offset);
    __global       float* C = (__global       float*)((__global       char*)C_void + C_offset);

    const int lidm = get_local_id(0);
    const int lidn = get_local_id(1);
    const int lid = lidn * WG_M + lidm;
    const int num_threads = WG_M * WG_N;

    const int offsetM = get_group_id(0) * OPWM;
    const int offsetN = get_group_id(1) * OPWN;

    __local half4  Alocal[OPWM][VEC_K];
    __local float4 Blocal[OPWN][VEC_K];

    float sum[OPTM][OPTN];

    for (int wm = 0; wm < OPTM; wm++) {
        for (int wn = 0; wn < OPTN; wn++) {
            sum[wm][wn] = 0.0f;
        }
    }

    const int numTiles = (K + CPWK - 1) / CPWK;

    for (int t = 0; t < numTiles; t++) {
        const int k_start = t * CPWK;

        for (int l = lid; l < OPWM * VEC_K; l += num_threads) {
            const int l_row_a = l % OPWM;
            const int l_vec_a = l / OPWM;
            const int g_row_a = offsetM + l_row_a;
            const int k_vec_a = k_start + l_vec_a * 4;

            if (g_row_a < M && k_vec_a < K) {
                if (k_vec_a + 3 < K) {
                    Alocal[l_row_a][l_vec_a] = vload4(0, A + g_row_a * K + k_vec_a);
                } else {
                    half4 tempA = (half4)(0.0h);
                    if (k_vec_a < K) tempA.s0 = A[g_row_a * K + k_vec_a];
                    if (k_vec_a + 1 < K) tempA.s1 = A[g_row_a * K + k_vec_a + 1];
                    if (k_vec_a + 2 < K) tempA.s2 = A[g_row_a * K + k_vec_a + 2];
                    Alocal[l_row_a][l_vec_a] = tempA;
                }
            } else {
                Alocal[l_row_a][l_vec_a] = (half4)(0.0h);
            }
        }

        for (int l = lid; l < OPWN * VEC_K; l += num_threads) {
            const int l_row_b = l % OPWN;
            const int l_vec_b = l / OPWN;
            const int g_row_b = offsetN + l_row_b;
            const int k_vec_b = k_start + l_vec_b * 4;

            if (g_row_b < N && k_vec_b < K) {
                if (k_vec_b + 3 < K) {
                    Blocal[l_row_b][l_vec_b] = vload4(0, B + g_row_b * K + k_vec_b);
                } else {
                    float4 tempB = (float4)(0.0f);
                    if (k_vec_b < K) tempB.s0 = B[g_row_b * K + k_vec_b];
                    if (k_vec_b + 1 < K) tempB.s1 = B[g_row_b * K + k_vec_b + 1];
                    if (k_vec_b + 2 < K) tempB.s2 = B[g_row_b * K + k_vec_b + 2];
                    Blocal[l_row_b][l_vec_b] = tempB;
                }
            } else {
                Blocal[l_row_b][l_vec_b] = (float4)(0.0f);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k_vec = 0; k_vec < VEC_K; k_vec++) {
            float4 a_fvecs[OPTM];
            int current_row_a = lidm;
            for (int wm = 0; wm < OPTM; wm++) {
                a_fvecs[wm] = convert_float4(Alocal[current_row_a][k_vec]);
                current_row_a += WG_M;
            }

            float4 b_fvecs[OPTN];
            int current_row_b = lidn;
            for (int wn = 0; wn < OPTN; wn++) {
                b_fvecs[wn] = Blocal[current_row_b][k_vec];
                current_row_b += WG_N;
            }

            for (int wm = 0; wm < OPTM; wm++) {
                for (int wn = 0; wn < OPTN; wn++) {
                    sum[wm][wn] += dot(a_fvecs[wm], b_fvecs[wn]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int wm = 0; wm < OPTM; wm++) {
        int globalRow = offsetM + lidm + wm * WG_M;
        if (globalRow < M) {
            for (int wn = 0; wn < OPTN; wn++) {
                int globalCol = offsetN + lidn + wn * WG_N;
                if (globalCol < N) {
                    C[globalCol * M + globalRow] = sum[wm][wn];
                }
            }
        }
    }
}
