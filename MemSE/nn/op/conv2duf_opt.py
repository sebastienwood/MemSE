from numba import njit, prange, cuda
import numba

_WARPSIZE = 32
_NUMWARPS = 4
max_blocksize = _NUMWARPS * _WARPSIZE
inner_sm_size = _WARPSIZE + 1
@cuda.jit(device=True, inline=True)
def warpReduceSum(val):
    offset = _WARPSIZE // 2
    while offset:
        val += cuda.shfl_down_sync(cuda.activemask(), val, offset)
        offset //= 2
    return val

@cuda.jit(device=True, inline=True)
def blockReduceSum(val, sm_partial):
    shared = sm_partial[cuda.threadIdx.x]
    lane = cuda.threadIdx.x % _WARPSIZE
    wid = cuda.threadIdx.x // _WARPSIZE
    val = warpReduceSum(val)
    if lane == 0:
        shared[wid] = val
    cuda.syncthreads()
    val = shared[lane] if cuda.threadIdx.x < cuda.blockDim.x / _WARPSIZE else 0.
    if wid == 0:
        val = warpReduceSum(val)
    return val


@cuda.jit
def op_numba_w(input, gamma, c, weight_shape_1, weight_shape_2, weight_shape_3, padding, stride):
    sharedC = cuda.shared.array(shape=512, dtype=numba.float32)
    sm_partials = cuda.shared.array((_NUMWARPS, inner_sm_size), dtype=numba.float32)
    xi, bi = cuda.grid(2)
    x_gridsize, y_gridsize = cuda.gridsize(2)
    threadX = cuda.threadIdx.x
    threadB = cuda.threadIdx.y

    if threadB == 0:
        tx = cuda.threadIdx.x
        while tx < c.shape[0]:
            sharedC[tx] = c[tx]
            tx += cuda.blockDim.x
    cuda.syncthreads()

    if bi > input.shape[0] or xi > gamma.shape[5]:
        return

    grid_bi = input.shape[0] * input.shape[2] * input.shape[3] * input.shape[5] * input.shape[6]
    while bi < grid_bi:
        bi_ = bi % input.shape[0]
        i0 = bi // input.shape[0] % input.shape[2]
        j0 = bi // (input.shape[0] * input.shape[2]) % input.shape[3]
        i0p = bi // (input.shape[0] * input.shape[2] * input.shape[3]) % input.shape[5]
        j0p = bi // (input.shape[0] * input.shape[2] * input.shape[3] * input.shape[5]) % input.shape[6]

        strided_i0 = i0 * stride[0]
        strided_i0p = i0p * stride[0]
        strided_j0 = j0 * stride[1]
        strided_j0p = j0p * stride[1]

        v = numba.float32(0.)
        for ii in range(weight_shape_2):
            # Virtual padding
            i0ii = strided_i0 + ii
            i0pii = strided_i0p + ii
            oob_0 = i0ii < padding[0] or i0ii >= gamma.shape[1] + padding[0]
            oob_0p = i0pii < padding[0] or i0pii >= gamma.shape[1] + padding[0]
            if oob_0 or oob_0p:
                continue
            for ji in range(weight_shape_3):
                j0ji = strided_j0 + ji
                j0pji = strided_j0p + ji
                oob_0j = oob_0 or j0ji < padding[1] or j0ji >= gamma.shape[2] + padding[1]
                oob_0pj = oob_0p or j0pji < padding[1] or j0pji >= gamma.shape[2] + padding[1]
                
                if not oob_0j and not oob_0pj:
                    # Recenter on actual coords
                    i0ii_padded = i0ii - padding[0]
                    j0ji_padded = j0ji - padding[1]
                    i0pii_padded = i0pii - padding[0]
                    j0pji_padded = j0pji - padding[1]
                    i = xi
                    while i < weight_shape_1:
                        v += gamma[bi_, i0ii_padded, j0ji_padded, i0pii_padded, j0pji_padded, i] #+ gamma[bi, i0ii_padded, j0ji_padded, i0pii_padded, j0pji_padded, i+cuda.blockDim.x]
                        i += x_gridsize
                
        v = blockReduceSum(v, sm_partials)
        # For each c0 update input
        tix = threadX
        while tix < input.shape[1]:
            cuda.atomic.add(input, (bi_, tix, i0, j0, tix, i0p, j0p), v * sharedC[tix])
            tix += cuda.blockDim.x
        bi += y_gridsize