

# =======================
# === Recursive TRMM ===
# ======================
"""
Top-level TRMM entry function (in-place): A * B → B
"""
function trmm!(A_tbm::TriangularBlockMatrix, B::AbstractMatrix)
    # B should be compatible to A
    if size(B, 1) != A_tbm.n
        error("Matrix B and A not compatible for multiplication.")
    end
    _trmm_recursive!(A_tbm.blocks, 0, A_tbm.triangular_type, B, 1, A_tbm.n, 0, A_tbm.l)
    return B
end


"""
Recursive block-level TRMM.

A * B_sub = B_sub (in-place).
"""
function _trmm_recursive!(
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
        # Leaf-level base case
        A_diag_block = A_blocks[max_levels + 1][diag_path + 1]
        B_sub = @view B[b_i_start:b_i_start + b_size - 1, :]

        # Safe temporary multiplication to avoid aliasing
        b_TEMP = similar(B_sub)
        mul!(b_TEMP, A_diag_block, B_sub)
        copyto!(B_sub, b_TEMP)
        return
    end

    # Recursive case: split block and proceed
    mid = cld(b_size, 2)
    b_i1, b_i2 = b_i_start, b_i_start + mid
    # b_j1, b_j2 = b_j_start, b_j_start + mid
    A_off_diag_block = A_blocks[current_level + 1][diag_path + 1]

    if triangular_type == :lower
        # Recurse into the bottom-right diagonal child
        _trmm_recursive!(A_blocks, current_level + 1, triangular_type, B, b_i2, b_size - mid, 2 * diag_path + 1, max_levels)

        # Apply off-diagonal block multiplication: B2 += A21 * B1
        B2 = @view B[b_i2:b_i1 + b_size - 1, :]
        B1 = @view B[b_i1:b_i1 + mid - 1, :]
        mul!(B2, A_off_diag_block, B1, 1, 1)

        # Recurse into the top-left diagonal child
        _trmm_recursive!(A_blocks, current_level + 1, triangular_type, B, b_i1, mid, 2 * diag_path, max_levels)

    # elseif triangular_type == :upper
    #   TO_DO
    end

    return
end