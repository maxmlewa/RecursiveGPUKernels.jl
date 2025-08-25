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