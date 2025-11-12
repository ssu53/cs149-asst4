import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    # print()
    # print(f"{X.shape=}")
    # print(f"{W.shape=}")

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == out_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax

    # Maximum free dimension of the stationary operand of general matrix multiplication on tensor engine
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128

    # Maximum partition dimension of a tile
    TILE_K = nl.tile_size.pmax  # 128

    # Maximum free dimension of the moving operand of general matrix multiplication on tensor engine
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512
    TILE_N = min(TILE_N, out_width)  # Adjust TILE_N if out_width is smaller

    # Partition dimension, to put W on sbuf
    PARTITION_DIM = 128

    # Reshape X for loading
    X_reshaped = X.reshape((batch_size, in_channels, input_height * input_width))

    K = in_channels
    M = out_channels

    # Reshape W and transpose and put it onto sbuf
    W_sbuf = nl.ndarray(
        shape=(TILE_K, TILE_M, in_channels // TILE_K, out_channels // TILE_M, filter_height, filter_width),
        dtype=W.dtype,
        buffer=nl.sbuf,
    )

    for m in nl.affine_range(M // TILE_M):
        W_sbuf_temp = nl.ndarray(
            shape=(TILE_M, in_channels, filter_height, filter_width),
            dtype=W.dtype,
            buffer=nl.sbuf,
        )
        nisa.dma_copy(src=W[m * TILE_M:(m + 1) * TILE_M, :, :, :], dst=W_sbuf_temp)

        for k in nl.affine_range(K // TILE_K):
            for fh in nl.affine_range(filter_height):
                for fw in nl.affine_range(filter_width):

                    # Fetch the tile to transpose (must be at most 128x128 sized tile)
                    a_tile = W_sbuf_temp[:, k * TILE_K:(k + 1) * TILE_K, fh, fw] # [TILE_M, TILE_K]

                    # Transpose the tile
                    a_tile_transposed = nisa.nc_transpose(a_tile)

                    # Store the result tile into HBM
                    res_sbuf = nl.copy(a_tile_transposed, dtype=a_tile_transposed.dtype)
                    nisa.dma_copy(src=res_sbuf, dst=W_sbuf[:, :, k, m, fh, fw])

    # Process the images in batches
    for b in nl.affine_range(batch_size):

        # Iterate over the tiles of out_channels
        for m in nl.affine_range(M // TILE_M): # chunking over out_channels

            # Move the chunk of bias to sbuf
            bias_sbuf = nl.ndarray(shape=(TILE_M, 1, 1), dtype=bias.dtype, buffer=nl.sbuf)
            nisa.dma_copy(src=bias[m * TILE_M:(m + 1) * TILE_M], dst=bias_sbuf[:,0,0])

            for pool_n in nl.affine_range(out_pool_height): # take one row at a time

                # Allocate a tile in psum for the final result of the convolution
                # This accumulates over the iteration over K and over the iteration of the filters
                res_conv_psum = nl.zeros((TILE_M, pool_size, TILE_N), dtype=nl.float32, buffer=nl.psum)
                    
                if pool_size == 2:
                    res_conv_sbuf = nl.ndarray(
                        shape=(TILE_M, pool_size, TILE_N),
                        dtype=X.dtype,
                        buffer=nl.sbuf,
                    )
                
                for pool_idx in nl.affine_range(pool_size):
                    n = pool_n * pool_size + pool_idx

        
                    # Iterate over the filters
                    for fh in nl.affine_range(filter_height):
                        for fw in nl.affine_range(filter_width):
                            for k in nl.affine_range(K // TILE_K): # chunking over in_channels
                                
                                # Index into W_sbuf (already on sbuf) for the lhsT tile
                                lhsT_tile = W_sbuf[:, :, k, m, fh, fw] # [TILE_M, TILE_K]
                                
                                # Declare the rhs tile on sbuf
                                rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=X.dtype, buffer=nl.sbuf) # [TILE_K, TILE_N]
                                # Load the tile from X_reshaped onto rhs_tile. It should be shifted to be appropriate for the filter location
                                # effectively X_shift = X[b, :, fh : fh + out_height, fw : fw + out_width]
                                nisa.dma_copy(
                                    dst=rhs_tile,
                                    src=X_reshaped[b, k * TILE_K:(k+1) * TILE_K, ((n + fh) * input_width) + fw : ((n + fh) * input_width) + fw + TILE_N],
                                )

                                # Accumulate partial-sums into PSUM
                                res_conv_psum[:, pool_idx, :] += nisa.nc_matmul(lhsT_tile[...], rhs_tile[...])

                    # Copy the result from PSUM back to SBUF, and cast to expected output data-type
                    res_sbuf = nl.copy(res_conv_psum, dtype=X_out.dtype)

                    # Add the bias
                    res_sbuf = nisa.tensor_tensor(res_sbuf, bias_sbuf, op=nl.add) # [TILE_M, pool_size, TILE_N]

                    if pool_size == 1: 
                        # Copy to hbm
                        nisa.dma_copy(
                            dst=X_out[b, TILE_M * m:TILE_M * (m + 1), n, 0:TILE_N],
                            src=res_sbuf[:, 0, :])
                    if pool_size == 2:
                        # Copy to res_conv_sbuf
                        nisa.dma_copy(
                            dst=res_conv_sbuf[:, :, :],
                            src=res_sbuf,
                        )

                # Pooling
                # res_conv_sbuf is shape [TILE_M, pool_size, TILE_N] -> [TILE_M, TILE_N // pool_size]
                if pool_size == 2:
                    res_conv_sbuf = res_conv_sbuf.reshape((TILE_M, pool_size, TILE_N // pool_size, pool_size))
                    res_conv_sbuf = nisa.tensor_reduce(nl.max, res_conv_sbuf, axis=3) # [TILE_M, 2, TILE_N // 2]
                    res_conv_sbuf = nisa.tensor_reduce(nl.max, res_conv_sbuf, axis=1) # [TILE_M, TILE_N // 2]

                    nisa.dma_copy(
                        dst=X_out[b, TILE_M * m:TILE_M * (m + 1), pool_n, :], # [1, 128, 1, out_width // 2]
                        src=res_conv_sbuf,
                    )


    return X_out

