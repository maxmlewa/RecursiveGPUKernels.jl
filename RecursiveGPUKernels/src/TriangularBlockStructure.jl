

# ======================================
# === Triangular Block Matrix Struct ===
# ======================================

"""
TriangularBlockMatrix stores the main recursive triangular matrix structure.

# Fields:
- blocks: Nested vector of matrix blocks (currently views of the original matrix)
- n: Original matrix size
- l: Recursion depth (number of block levels)
- triangular_type: :lower or :upper
"""
struct TriangularBlockMatrix{T}
    blocks::
    n::Int
    l::Int
    triangular_type::Symbol
end



# ======================================
# === GPU Adaption of Block Matrix ===
# ======================================

"""
Converts CPU-based block matrix into GPU-ready (CuArray for now) blocks.
"""
function adapt_blocks_to_gpu(B::TriangularBlockMatrix{T}) where {T}
    blocks_gpu = Vector{Vector{AbstractMatrix{T}}}(undef, B.l + 1)
    for k in 1:(B.l + 1)
        blocks_gpu[k] = Vector{AbstractMatrix{T}}(undef, length(B.blocks[k]))
        for i in 1:length(B.blocks[k])
            # Change line below to adapt to the available device
            blocks_gpu[k][i] = CuArray(B.blocks[k][i])
        end
    end
    return TriangularBlockMatrix(blocks_gpu, B.n, B.l, B.triangular_type)
end



# =======================================
# === Triangular Block Matrix Builder ===
# =======================================

"""
Builds a TriangularBlockMatrix from full matrix A and recursion depth l.
"""
function TriangularBlockMatrix(A::AbstractMatrix{T}, l::Int; triangular_type::Symbol = :lower) where {T}
    if size(A, 1) != size(A, 2)
        error("Input matrix A must be square.")
    end
    blocks_cpu = build_block_structure(A, l; triangular_type)
    return TriangularBlockMatrix{T}(blocks_cpu, size(A, 1), l, triangular_type)
end


"""
Internal recursive block structure builder.
Divides matrix into triangular blocks recursively.
"""
function build_block_structure(A::AbstractMatrix{T}, l::Int; triangular_type::Symbol = :lower) where {T}
    n = size(A, 1)
    blocks = Vector{Vector{AbstractMatrix{T}}}(undef, l + 1)

    # Initialize leaf-level blocks
    for k in 1:(l+1)
        blocks[k] = Vector{AbstractMatrix{T}}(undef, 2^(k - 1))
    end

    function subdivide_and_assign(i_start, j_start, current_size, level, diag_path)
        if level == l + 1
            blocks[level][diag_path + 1] = @view A[i_start:i_start + current_size - 1, j_start:j_start + current_size - 1]
            return
        end

        mid = cld(current_size, 2)
        i1, i2 = i_start, i_start + mid
        j1, j2 = j_start, j_start + mid

        if triangular_type == :lower
            blocks[level][diag_path + 1] = @view A[i2:i_start + current_size - 1, j1:j1 + mid - 1]
        elseif triangular_type == :upper
            blocks[level][diag_path + 1] = @view A[i1:i1 + mid - 1, j2:j_start + current_size - 1]
        end

        subdivide_and_assign(i1, j1, mid, level + 1, 2 * diag_path)
        subdivide_and_assign(i2, j2, current_size - mid, level + 1, 2 * diag_path + 1)
    end

    subdivide_and_assign(1, 1, n, 1, 0)
    return blocks
end