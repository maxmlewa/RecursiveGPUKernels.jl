
# ======================
# === Recursive TRSM ===
# ======================

"""
Top-level TRSM entry function (in-place): A / B → B
"""
function trsm!(A_tbm::TriangularBlockMatrix, B::AbstractMatrix)
    # Compatibility check
    if size(B, 1) != A_tbm.n
        error("Matrix B and A not compatible for TRSM.")
    end
    _trsm_recursive!(A_tbm.blocks, 0, A_tbm.triangular_type, B, 1, A_tbm.n, 0, A_tbm.l)
    return B
end


"""
Recursive block-level TRSM. Solves A * X = B.
"""
function _trsm_recursive!(
    A_blocks,
    current_level::Int,
    triangular_type::Symbol,
    B::AbstractMatrix, 
    b_i_start::Int, 
    b_size::Int,
    diag_path::Int, 
    max_levels::Int
)
    if current_level == max_levels
        A_diag_block = A_blocks[max_levels + 1][diag_path + 1]
        B_sub = @view B[b_i_start:b_i_start + b_size - 1, :]

        # Copy B_sub into a full buffer and solve
        B_full = copy(B_sub)

        # TRSM on GPU using cuBLAS
        CUDA.CUBLAS.trsm!('L', 'L', 'N', 'N', 1.0f0, A_diag_block, B_full)
        copyto!(B_sub, B_full)
        return
    end

    mid = cld(b_size, 2)
    b_i1, b_i2 = b_i_start, b_i_start + mid
    A_off_diag_block = A_blocks[current_level + 1][diag_path + 1]

    if triangular_type == :lower
        _trsm_recursive!(A_blocks, current_level + 1, triangular_type, B, b_i1, mid, 2 * diag_path, max_levels)

        # B2 -= A21 * X1
        B1 = @view B[b_i1:b_i1 + mid - 1, :]
        B2 = @view B[b_i2:b_i1 + b_size - 1, :]
        mul!(B2, A_off_diag_block, B1, -1, 1)

        _trsm_recursive!(A_blocks, current_level + 1, triangular_type, B, b_i2, b_size - mid, 2 * diag_path + 1, max_levels)
    end

    return
end