import time
import jax
import jax.numpy as jnp

# Simple synthetic data
N = 10000
X = jnp.linspace(-1, 1, N)
Y = 3.0 * X + 1.0  # y = 3x + 1

STEPS = 1_000_000

# Model: simple linear weights
def loss_fn(params, x, y):
    pred = params[0] * x + params[1]
    return jnp.mean((pred - y) ** 2)

@jax.jit
def sgd_step(params, x, y, lr=1e-3):
    grads = jax.grad(loss_fn)(params, x, y)
    return params - lr * grads

# ---------------------
# Pattern A: Single scan, log afterwards
# ---------------------
@jax.jit
def train_scan_all(params, xs, ys):
    def step_fn(carry, _):
        p = carry
        new_p = sgd_step(p, xs, ys)
        return new_p, loss_fn(new_p, xs, ys)
    
    final, history = jax.lax.scan(step_fn, params, None, length=STEPS)
    return final, history

# ---------------------
# Pattern B: Scan with host callback
# ---------------------
def train_scan_with_callback_impl(params, xs, ys, log_fn, log_frequency):
    def step_fn(carry, i):
        p, step_count = carry
        new_p = sgd_step(p, xs, ys)
        loss_val = loss_fn(new_p, xs, ys)
        
        # Only log every log_frequency steps
        should_log = (step_count % log_frequency) == 0
        jax.lax.cond(
            should_log,
            lambda: jax.debug.callback(log_fn, loss_val),
            lambda: None
        )
        
        return (new_p, step_count + 1), None
    
    final, _ = jax.lax.scan(step_fn, (params, 0), jnp.arange(STEPS))
    return final[0]

# Create JIT version with static_argnames
train_scan_with_callback = jax.jit(
    train_scan_with_callback_impl, 
    static_argnames=['log_fn', 'log_frequency']
)

# ---------------------
# Pattern C: Chunked training
# ---------------------
def train_chunk_impl(params, xs, ys, chunk_steps):
    def step_fn(carry, _):
        p = carry
        new_p = sgd_step(p, xs, ys)
        return new_p, loss_fn(new_p, xs, ys)
    
    final, history = jax.lax.scan(step_fn, params, None, length=chunk_steps)
    return final, history

# JIT with chunk_steps as static argument
train_chunk = jax.jit(train_chunk_impl, static_argnames=['chunk_steps'])

def train_chunked(params, xs, ys, chunk_steps):
    num_chunks = STEPS // chunk_steps
    all_metrics = []
    p = params
    
    for i in range(num_chunks):
        p, hist = train_chunk(p, xs, ys, chunk_steps)
        # Log metrics after each chunk
        all_metrics.append(float(hist.mean()))
    
    return p, all_metrics

# ---------------------
# Run benchmarks
# ---------------------

def benchmark_pattern(name, fn, *args, **kwargs):
    """Helper to properly benchmark including compilation time"""
    print(f"\nRunning {name}...")
    
    # Warmup run (compilation)
    print("  Warmup (compilation)...")
    t0 = time.time()
    result = fn(*args, **kwargs)
    jax.block_until_ready(result[0] if isinstance(result, tuple) else result)
    compile_time = time.time() - t0
    print(f"  Compile time: {compile_time:.4f}s")
    
    # Actual benchmark run
    print("  Benchmark run...")
    t0 = time.time()
    result = fn(*args, **kwargs)
    jax.block_until_ready(result[0] if isinstance(result, tuple) else result)
    run_time = time.time() - t0
    print(f"  Run time: {run_time:.4f}s")
    
    return result, compile_time, run_time

# Initial parameters
params0 = jnp.array([0.0, 0.0])

results = {}

# Pattern A
(final_A, histA), compile_A, run_A = benchmark_pattern(
    "Pattern A (full scan, log afterwards)",
    train_scan_all,
    params0, X, Y
)
results["A"] = {
    "compile_time": compile_A,
    "run_time": run_A,
    "total_time": compile_A + run_A,
    "metrics_count": len(histA),
    "final_loss": float(histA[-1])
}

# Pattern B - various logging frequencies
for log_freq in [1, 1000, 10000, 100000]:
    logs_B = []
    def log_fn(x):
        logs_B.append(float(x))
    
    logs_B.clear()
    final_B, compile_B, run_B = benchmark_pattern(
        f"Pattern B (callback, freq={log_freq})",
        train_scan_with_callback,
        params0, X, Y, log_fn, log_freq
    )
    results[f"B_{log_freq}"] = {
        "compile_time": compile_B,
        "run_time": run_B,
        "total_time": compile_B + run_B,
        "metrics_count": len(logs_B),
        "final_loss": float(loss_fn(final_B, X, Y))
    }

# Pattern C - various chunk sizes
for chunk_size in [1, 1000, 10000, 100000]:
    (final_C, metrics_C), compile_C, run_C = benchmark_pattern(
        f"Pattern C (chunked, size={chunk_size})",
        train_chunked,
        params0, X, Y, chunk_size
    )
    results[f"C_{chunk_size}"] = {
        "compile_time": compile_C,
        "run_time": run_C,
        "total_time": compile_C + run_C,
        "metrics_count": len(metrics_C),
        "final_loss": float(loss_fn(final_C, X, Y))
    }

# Report
print("\n" + "="*80)
print("BENCHMARK RESULTS")
print("="*80)
print(f"{'Pattern':<30} {'Compile (s)':<12} {'Run (s)':<12} {'Total (s)':<12} {'Metrics':<10} {'Final Loss':<12}")
print("-"*90)

for name, data in results.items():
    print(f"{name:<30} {data['compile_time']:>11.4f} {data['run_time']:>11.4f} "
          f"{data['total_time']:>11.4f} {data['metrics_count']:>9} {data['final_loss']:>11.6f}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print(f"Pattern A: Single scan, minimal overhead, returns and logs full history")
print("\nPattern B: Callbacks with different frequencies")
for log_freq in [1, 1000, 10000, 100000]:
    key = f"B_{log_freq}"
    slowdown = results[key]['run_time'] / results['A']['run_time']
    print(f"  - log_freq={log_freq:>6}: {slowdown:.2f}x slower")

print("\nPattern C: Chunking adds Python loop overhead")
for chunk_size in [1, 1000, 10000, 100000]:
    key = f"C_{chunk_size}"
    slowdown = results[key]['run_time'] / results['A']['run_time']
    print(f"  - chunk_size={chunk_size:>6}: {slowdown:.2f}x slower")