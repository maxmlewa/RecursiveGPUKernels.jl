# File to facilitate easy benchmarking on cluster

# Loading packages
using LinearAlgebra
using CUDA,Adapt
using KernelAbstractions
using Random


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



# ======================================
# === Benchmarking Helper ===
# ======================================


# using KernelAbstractions

"""
    run_manual_benchmark(func_to_benchmark(), backend=CUDADevice(), min_time_s::Float64 = 1.0, min_iters::Int = 5)

Times a GPU function using KernelAbstractions' synchronization for accuracy.
Returns best time (ms)
"""
function run_manual_benchmark(func_to_benchmark(), backend=CUDADevice(), min_time_s::Float64 = 1.0, min_iters::Int = 5)
    # warm up
    func()
    KernelAbstractions.synchronize(backend)
    
    best_time_ns = 1e12 
    elapsed_time_ns = 0.0
    i = 0

    while elapsed_time_ns < min_time_s * 1e9 || i < min_iters
        KernelAbstractions.synchronize(backend)
        start_time = time_ns()

        func_to_benchmark()

        KernelAbstractions.synchronize(backend)
        end_time = time_ns()

        this_duration = end_time - start_time
        elapsed_time_ns += this_duration
        best_time_ns = min(best_time_ns, this_duration)
        i += 1
    end

    return best_time_ns
end

function run_single_benchmark(func_to_benchmark, backend)
    KernelAbstractions.synchronize(backend)
    start_time = time_ns()

    func_to_benchmark()

    KernelAbstractions.synchronize(backend)
    end_time = time_ns()

    return end_time - start_time
end



# ===========================
# === ACCURACY BENCHMARKS ===
# ===========================
"""
Benchmark the recursive and standard TRMM implementations on the GPU.
    Example usage: benchmark_trmm_accuracy(512, 256)
"""
function benchmark_trmm_accuracy(N::Int, block_threshold::Int)
    L = max(0, floor(Int, log2(N / block_threshold)))

    if has_cuda_gpu()
        println("Using CUDA GPU for computation...")
    else
        println("No GPU found.")
        return
    end

    println("Benchmarking TRMM for size $N with block threshold $block_threshold (L = $L)...")

    # CPU inputs
    A_cpu = tril(rand(Float32, N, N))
    B_cpu = rand(Float32, N, N)

    # GPU versions
    A_gpu = CuArray(A_cpu)
    B_gpu_recursive = CuArray(copy(B_cpu))
    B_gpu_standard = CuArray(copy(B_cpu))

    # Recursive TRMM
    A_tbm = TriangularBlockMatrix(A_cpu, L)
    A_tbm_gpu = adapt_blocks_to_gpu(A_tbm)
    trmm!(A_tbm_gpu, B_gpu_recursive)
    sync_device()

    # Standard cuBLAS TRMM
    CUDA.CUBLAS.trmm!('L', 'L', 'N', 'N', 1.0, A_gpu, B_gpu_standard, B_gpu_standard)
    sync_device()

    # Check result
    result_recursive = Array(B_gpu_recursive)
    result_standard = Array(B_gpu_standard)

    if isapprox(result_recursive, result_standard, atol=1e-4)
        println("SUCCESS: Results match.")
    else
        println("FAILURE: Mismatch. Max diff = ", maximum(abs.(result_recursive - result_standard)))
    end
end


"""
Benchmark the recursive and standard TRSM implementations on the GPU for accuracy.
    Example usage:  benchmark_trsm_accuracy(512, 256)

"""
function benchmark_trsm_accuracy(N::Int, block_threshold::Int)
    L = max(0, floor(Int, log2(N / block_threshold)))

    if has_cuda_gpu()
        println("Using CUDA GPU for computation...")
    else
        println("No GPU found.")
        return
    end

    println("Benchmarking TRSM for size $N with block threshold $block_threshold (L = $L)...")

    # ----------------------------
    # CPU inputs
    # ----------------------------
    A_cpu = tril(rand(Float32, N, N)) + I * 10f0  # Ensure well-conditioned
    B_cpu = rand(Float32, N, N)

    # Expected result: Solve A * X = B on CPU
    expected_cpu = A_cpu \ B_cpu

    # ----------------------------
    # Move to GPU
    # ----------------------------
    A_gpu = CuArray(A_cpu)
    B_gpu_recursive = CuArray(copy(B_cpu))
    B_gpu_standard = CuArray(copy(B_cpu))

    # ----------------------------
    # Recursive TRSM
    # ----------------------------
    A_tbm = TriangularBlockMatrix(A_cpu, L; triangular_type = :lower)
    A_tbm_gpu = adapt_blocks_to_gpu(A_tbm)
    trsm!(A_tbm_gpu, B_gpu_recursive)
    sync_device()

    # ----------------------------
    # Standard cuBLAS TRSM
    # ----------------------------
    CUDA.CUBLAS.trsm!('L', 'L', 'N', 'N', 1.0f0, A_gpu, B_gpu_standard)
    sync_device()

    # ----------------------------
    # Accuracy Check
    # ----------------------------
    result_recursive = Array(B_gpu_recursive)
    result_standard = Array(B_gpu_standard)

    println("\n--- Accuracy Results ---")

    max_diff_recursive = maximum(abs.(expected_cpu - result_recursive))
    max_diff_standard = maximum(abs.(expected_cpu - result_standard))

    rel_err_recursive = max_diff_recursive / maximum(abs.(expected_cpu))
    rel_err_standard = max_diff_standard / maximum(abs.(expected_cpu))

    println("Recursive TRSM:  Max diff = $(round(max_diff_recursive, sigdigits=4)),  Rel error = $(round(rel_err_recursive, sigdigits=4))")
    println("Standard TRSM:   Max diff = $(round(max_diff_standard, sigdigits=4)),  Rel error = $(round(rel_err_standard, sigdigits=4))")

    if isapprox(result_recursive, expected_cpu; atol=1e-4, rtol=1e-4)
        println("Recursive TRSM PASS")
    else
        println("Recursive TRSM FAIL")
    end
 
end




# =========================
# === TIMING BENCHMARKS ===
# =========================
"""
Benchmark runtime of recursive TRMM vs cuBLAS TRMM on GPU using manual timer.
"""
function benchmark_trmm_timing(N::Int, block_threshold::Int)
    L = max(0, floor(Int, log2(N / block_threshold)))

    if !has_cuda_gpu()
        println("No GPU found.")
        return
    end

    println("\nBenchmarking TRMM timing for size $N with block threshold $block_threshold (L = $L)...")

    # -------------------------------
    # Prepare test data (Float32) for uniformity
    # -------------------------------
    A_cpu = tril(rand(Float32, N, N))
    B_cpu = rand(Float32, N, N)

    # CuArrays
    A_gpu = CuArray(A_cpu)
    B_gpu_recursive = CuArray(copy(B_cpu))
    B_gpu_standard = CuArray(copy(B_cpu))

    # -------------------------------
    # Prepare recursive TRMM call
    # -------------------------------
    A_tbm = TriangularBlockMatrix(A_cpu, L; triangular_type = :lower)
    A_tbm_gpu = adapt_blocks_to_gpu(A_tbm)

    recursive_func = () -> trmm!(A_tbm_gpu, B_gpu_recursive)

    # -------------------------------
    # Prepare cuBLAS TRMM call
    # -------------------------------
    standard_func = () -> CUDA.CUBLAS.trmm!(
        'L',  # Left side
        'L',  # Lower triangular
        'N',  # No transpose
        'N',  # Non-unit diagonal
        1.0f0, A_gpu, B_gpu_standard, B_gpu_standard
    )

    backend = KernelAbstractions.CUDADevice()

    # -------------------------------
    # Benchmark each implementation
    # -------------------------------
    time_recursive_ns = run_manual_benchmark(recursive_func, backend)
    time_standard_ns  = run_manual_benchmark(standard_func, backend)

    # Convert to ms
    time_recursive_ms = time_recursive_ns / 1e6
    time_standard_ms  = time_standard_ns / 1e6

    # -------------------------------
    # Print timing results
    # -------------------------------
    println("\n--- Timing Results (Best of runs) ---")
    println("Recursive TRMM: $(round(time_recursive_ms, digits=3)) ms")
    println("cuBLAS   TRMM:  $(round(time_standard_ms,  digits=3)) ms")

    speedup = time_standard_ns / time_recursive_ns
    println("Recursive is $(round(speedup, digits=2)) faster than cuBLAS.")
end


"""
Benchmark runtime of recursive TRSM vs cuBLAS TRSM on GPU using manual timer.
Example usage:  benchmark_trsm_timing(512, 256), 
            or loop: for i in 7:12
                        benchmark_trsm_timing(2^i, 256)
                     end
"""
function benchmark_trsm_timing(N::Int, block_threshold::Int)
    L = max(0, floor(Int, log2(N / block_threshold)))

    if !has_cuda_gpu()
        println("No GPU found.")
        return
    end

    println("\nBenchmarking TRSM timing for size $N with block threshold $block_threshold (L = $L)...")

    # -------------------------------
    # Prepare test data (Float32)
    # -------------------------------
    A_cpu = tril(rand(Float32, N, N)) + I  # Add identity to ensure invertibility
    B_cpu = rand(Float32, N, N)

    # CuArrays
    A_gpu = CuArray(A_cpu)
    B_gpu_recursive = CuArray(copy(B_cpu))
    B_gpu_standard = CuArray(copy(B_cpu))

    # -------------------------------
    # Prepare recursive TRSM call
    # -------------------------------
    A_tbm = TriangularBlockMatrix(A_cpu, L; triangular_type = :lower)
    A_tbm_gpu = adapt_blocks_to_gpu(A_tbm)

    recursive_func = () -> trsm!(A_tbm_gpu, B_gpu_recursive)

    # -------------------------------
    # Prepare cuBLAS TRSM call
    # -------------------------------
    standard_func = () -> CUDA.CUBLAS.trsm!(
        'L',  # Left side
        'L',  # Lower triangular
        'N',  # No transpose
        'N',  # Non-unit diag (because we added I above)
        1.0f0, A_gpu, B_gpu_standard
    )

    backend = KernelAbstractions.CUDADevice()

    # -------------------------------
    # Benchmark each implementation
    # -------------------------------
    time_recursive_ns = run_manual_benchmark(recursive_func, backend)
    time_standard_ns  = run_manual_benchmark(standard_func, backend)

    # Convert to ms
    time_recursive_ms = time_recursive_ns / 1e6
    time_standard_ms  = time_standard_ns / 1e6

    # -------------------------------
    # Print timing results
    # -------------------------------
    println("\n--- Timing Results (Best of runs) ---")
    println("Recursive TRSM: $(round(time_recursive_ms, digits=3)) ms")
    println("cuBLAS   TRSM:  $(round(time_standard_ms,  digits=3)) ms")

    speedup = time_standard_ns / time_recursive_ns
    println("Recursive is $(round(speedup, digits=2))× faster/slower than cuBLAS.")
end





"""
Synchronizes the CUDA device to ensure timing is accurate.
"""
function sync_device()
    if has_cuda_gpu()
        CUDA.synchronize()
    end
end




"""
Main entry point to run benchmarks for increasing sizes.
"""
function main()
    for i in 2:10
        N = 2^i
        block_threshold = 256
        println("\n==============================")
        println("Benchmarking for N = $N")
        println("==============================")
        benchmark_trmm(N, block_threshold)
    end
end

# Uncomment to run:
# main()