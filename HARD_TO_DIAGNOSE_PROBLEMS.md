# Hard-to-Diagnose Problems in WeatherFlow

**Generated:** 2026-01-03
**Analysis Scope:** Core modules (data, models, training, physics)

This document catalogues subtle bugs and issues that are difficult to diagnose because they:
- Cause silent failures or corrupted results
- Only manifest under specific conditions
- Produce misleading error messages
- Involve race conditions or state management

---

## 🔴 CRITICAL ISSUES

### 1. **SYNTAX ERROR in `weatherflow/__init__.py` - BREAKS ALL IMPORTS**

**Location:** `weatherflow/__init__.py:1-9`

**Problem:**
The docstring structure is malformed:
```python
"""
# weatherflow/__init__.py
"""WeatherFlow: A Deep Learning Library for Weather Prediction.

This module is intentionally lightweight on import. Heavy submodules and optional
features are loaded lazily when accessed to avoid ImportError on simple imports
in minimal environments (e.g., Google Colab, Lambda Labs) that don't have all
optional dependencies installed.
"""
```

The first `"""` on line 1 opens a string, line 2 is inside that string, then line 3's `"""` CLOSES it, leaving the rest as unquoted code. The `don't` on line 7 starts an unterminated string literal.

**Impact:**
- **The entire package cannot be imported** - `import weatherflow` fails with SyntaxError
- This is a showstopper bug that makes the package completely unusable
- The error message is misleading (points to line 7 instead of line 1)

**Why It's Hard to Diagnose:**
- File appears correct in most editors
- Syntax highlighters may show it correctly
- The error points to line 7, not the root cause on lines 1-3
- No smart quotes or encoding issues (which would be the first suspicion)

**Test:**
```bash
python3 -c "import weatherflow"  # Fails with SyntaxError
```

---

### 2. **Division by Near-Zero Creating NaN Gradients**

**Location:** `weatherflow/models/flow_matching.py:418`

**Code:**
```python
v_target = (x1 - x0) / (1 - t).view(-1, 1, 1, 1)
```

**Problem:**
When `t` approaches 1.0 (which happens during training), the denominator `(1 - t)` approaches zero, causing:
- Division by very small numbers → numerical overflow
- NaN or Inf values in gradients
- Silent gradient corruption that degrades training

**Impact:**
- Training becomes unstable near t=1
- Model fails to learn the final state of the flow
- Gradients become NaN but training continues (silent failure)
- Loss curves show sudden spikes or become NaN

**Why It's Hard to Diagnose:**
- Only occurs when `t` is sampled near 1.0 (probabilistically rare early in training)
- NaN/Inf checking is not performed after this computation
- Error doesn't raise an exception, just corrupts the loss
- Works fine for most of the training, fails intermittently

**Evidence:**
```python
# weatherflow/training/flow_trainer.py:279
t = torch.rand(x0.size(0), device=self.device)  # Can sample 0.999, 0.9999, etc.
```

No clamping or epsilon is added to prevent t=1.

---

### 3. **Division by Zero in Spatial Derivatives**

**Location:** `weatherflow/physics/losses.py:99-100`

**Code:**
```python
dlat = np.pi / (lat_dim - 1)  # Division by zero if lat_dim=1
dlon = 2 * np.pi / lon_dim    # Division by zero if lon_dim=0
```

**Problem:**
- If `lat_dim=1`: Division by zero → `dlat = inf`
- If `lon_dim=0`: Division by zero → crash
- No input validation on grid dimensions

**Impact:**
- Entire physics loss becomes NaN/Inf
- Training silently fails with corrupted physics constraints
- Could crash with ZeroDivisionError if lon_dim=0

**Why It's Hard to Diagnose:**
- Only happens with edge-case grid sizes (single latitude, zero longitudes)
- Error occurs deep in physics calculations
- May not crash if wrapped in try/except elsewhere
- Test suite doesn't test boundary grid sizes

---

### 4. **TimeEncoder Division by Zero**

**Location:** `weatherflow/models/flow_matching.py:51-52`

**Code:**
```python
half_dim = self.dim // 2
embeddings = np.log(10000) / (half_dim - 1)  # If dim=2, half_dim=1, divides by 0
```

**Problem:**
When `dim=2`, `half_dim=1`, resulting in division by zero.

**Impact:**
- Model initialization fails with dim=2
- Error is cryptic: `RuntimeWarning: divide by zero encountered`
- Creates Inf embeddings that propagate through the network

**Why It's Hard to Diagnose:**
- dim=2 is an edge case but not invalid
- No validation that dim must be > 2
- Error manifests as NaN loss later in training, not at initialization

---

### 5. **Race Condition in Background Simulation Thread**

**Location:** `app.py:112-121`

**Code:**
```python
sim_id = f"sim_{int(time.time())}_{np.random.randint(1000)}"  # Line 112
runner = SimulationRunner(sim_id, config)
active_simulations[sim_id] = runner  # Line 116

thread = threading.Thread(target=runner.run)
thread.daemon = True
thread.start()  # Line 121
```

**Problems:**
1. **Non-thread-safe random number generation:** `np.random.randint(1000)` uses global state, can cause collisions
2. **Unsynchronized dictionary access:** `active_simulations` dict modified from main thread and background threads without locks
3. **Race condition in status checks:** Lines 132-147 read `active_simulations` without synchronization

**Impact:**
- Simulation ID collisions under concurrent requests
- Dictionary corruption from concurrent modifications
- `RuntimeError: dictionary changed size during iteration` possible
- Intermittent failures that only occur under load

**Why It's Hard to Diagnose:**
- Only manifests under concurrent requests
- Works fine in single-threaded testing
- Timing-dependent failures
- Python GIL provides false sense of thread safety for dict operations

---

### 6. **Silent Data Loading Failures**

**Location:** `weatherflow/data/datasets.py:280-286`

**Code:**
```python
except Exception as e:
    last_exception = e
    logging.warning(f"Method failed with error: {str(e)}")
    continue
```

**Problem:**
- All methods can fail silently
- Only the LAST exception is kept, losing context from earlier failures
- If all methods fail, downstream code receives incomplete data

**Location 2:** `weatherflow/data/datasets.py:38`
```python
if file_path.exists():
    with h5py.File(file_path, "r") as f:
        self.data[var] = np.array(f[var])
else:
    print(f"Warning: File {file_path} not found.")  # Only prints, doesn't raise
```

**Impact:**
- Training proceeds with empty or partial data
- Model trains but produces nonsense results
- Error only discovered hours into training when checking results
- `__len__` returns 0, but DataLoader doesn't fail until iteration

**Why It's Hard to Diagnose:**
- Warnings are easy to miss in logs
- Datasets appear to load successfully
- DataLoader creation succeeds
- Failure only occurs during training epoch
- By the time error is noticed, the root cause (missing file) is lost

---

### 7. **Negative/Zero Dataset Length**

**Location:** `weatherflow/data/datasets.py:288-290`

**Code:**
```python
def __len__(self) -> int:
    return len(self.times) - 1  # Can return -1 or 0
```

**Problem:**
- If `self.times` has 0 or 1 elements, returns -1 or 0
- No validation that `times` has at least 2 elements
- DataLoader silently creates empty batches or raises cryptic IndexError

**Impact:**
- `IndexError` during iteration with misleading traceback
- Or DataLoader returns 0 batches, training loop never executes
- Error is far from the root cause

**Why It's Hard to Diagnose:**
- Error occurs during DataLoader iteration, not dataset creation
- Error message doesn't mention the `__len__` method
- Root cause is invalid time slice selection earlier

---

### 8. **No Error Handling on ERA5 Download**

**Location:** `weatherflow/data/era5.py:110-123`

**Code:**
```python
client.retrieve(
    "reanalysis-era5-pressure-levels",
    {...},
    path,
)  # No try/except, no validation
```

**Problem:**
- Network failures: timeout, connection reset → crash with unclear error
- Rate limiting: API returns error → silent partial file write
- Invalid credentials: fails but error is masked
- Disk full: partial file written, no cleanup
- File corruption: downloaded file not validated

**Impact:**
- Corrupted data files silently created
- Training proceeds with garbage data
- Errors only discovered when data is loaded (much later)
- No automatic retry or fallback

**Why It's Hard to Diagnose:**
- Download appears successful (file exists)
- No checksum or validation
- Error manifests as NaN loss in training, not download failure
- Time gap between download and use makes correlation difficult

---

## 🟡 HIGH SEVERITY ISSUES

### 9. **EMA State Dict Type Mismatch**

**Location:** `weatherflow/training/flow_trainer.py:251`

**Code:**
```python
if self._ema_params is not None:
    eval_state = self.model.state_dict()
    self.model.load_state_dict(self._ema_state_dict())  # type: ignore
```

**Problem:**
- `_ema_state_dict()` returns `Optional[Dict]`, but can return `None`
- `load_state_dict()` doesn't accept `None`
- Type checker ignored with `# type: ignore`

**Impact:**
- Runtime error if EMA initialization fails silently
- Model state corrupted during evaluation
- Evaluation uses training weights instead of EMA weights

**Why It's Hard to Diagnose:**
- Protected by `if self._ema_params is not None` check
- But `_ema_state_dict()` can still return None if dict is empty
- Type ignore comment hides the issue

---

### 10. **Missing Model Eval Mode Restoration**

**Location:** `weatherflow/training/flow_trainer.py:247-251`

**Code:**
```python
eval_state = None
if self._ema_params is not None:
    eval_state = self.model.state_dict()
    self.model.load_state_dict(self._ema_state_dict())  # type: ignore

# ... evaluation happens ...
# Missing: self.model.load_state_dict(eval_state)
```

**Problem:**
- Training state is saved to `eval_state`
- Model is switched to EMA weights
- After evaluation, training state is NEVER restored
- Subsequent training uses EMA weights instead of training weights

**Impact:**
- EMA updates corrupt because they're applied to EMA params, not training params
- Training diverges after first validation
- Loss increases after validation epochs

**Why It's Hard to Diagnose:**
- Only affects training AFTER validation
- Works fine for first epoch before validation
- Effect is gradual degradation, not immediate failure
- Requires understanding EMA mechanics to notice

---

### 11. **Device Mismatch in Flow Visualization**

**Location:** `weatherflow/utils/flow_visualization.py:27-33`

**Code:**
```python
states = [x0.cpu()]

for t in times[1:]:
    t_batch = t.expand(x0.size(0))
    v_t = model(states[-1].to(device), t_batch)  # states[-1] copied to device
    next_state = states[-1].to(device) + v_t * (1/n_steps)  # states[-1] copied AGAIN
    states.append(next_state.cpu())
```

**Problem:**
- `states[-1]` is on CPU
- Copied to device TWICE per iteration (lines 31 and 32)
- Unnecessary memory allocations and data transfers
- Inefficient and can cause CUDA OOM on long rollouts

**Impact:**
- 2x memory usage during visualization
- Significant slowdown from repeated CPU↔GPU transfers
- May OOM even when visualization should fit in memory

**Why It's Hard to Diagnose:**
- Still produces correct results
- Only causes performance issues, not correctness issues
- OOM error doesn't clearly indicate redundant transfers

---

### 12. **Type Confusion: Tensor vs NumPy Array**

**Location:** `weatherflow/physics/losses.py:122-124` (example usage)

**Code:**
```python
if pressure_levels.dim() == 1:  # Assumes tensor
    pressure_levels = pressure_levels.view(1, n_levels, 1, 1) * 100
```

**Problem:**
- `pressure_levels` can be NumPy array or torch Tensor
- NumPy arrays don't have `.dim()` method
- Crashes with `AttributeError: 'numpy.ndarray' object has no attribute 'dim'`

**Impact:**
- Crashes when using NumPy array input
- Error message doesn't mention type expectation
- Unclear which type is expected

**Why It's Hard to Diagnose:**
- No type hints indicate expected type
- Works if you pass tensors, fails for NumPy
- Error is deep in call stack

---

### 13. **Spectral Loss Division by Zero**

**Location:** `weatherflow/physics/losses.py:226-228`

**Code:**
```python
denominator = ((log_k - log_k_mean)**2).sum()
slope = numerator / (denominator + 1e-8)
```

**Problem:**
- If all `log_k` values are identical, `denominator ≈ 0`
- Epsilon `1e-8` is arbitrary and may not prevent numerical issues
- Can happen with small grids or uniform spectra

**Impact:**
- Division by near-zero → numerical instability
- Loss becomes very large or Inf
- Gradient explosion

**Why It's Hard to Diagnose:**
- Depends on data statistics
- Only happens with specific spectral characteristics
- Epsilon appears to protect against it but doesn't scale with magnitude

---

### 14. **No NaN/Inf Checking in Metrics**

**Location:** `weatherflow/training/metrics.py:6`

**Code:**
```python
def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))
```

**Problem:**
- If `pred` or `target` contain NaN/Inf, result is NaN
- No validation or clipping
- NaN propagates silently through training

**Impact:**
- Training continues with NaN loss
- Optimizer updates become NaN
- All weights become NaN
- Training appears to run but produces garbage

**Why It's Hard to Diagnose:**
- Loss value may still print as a number (depending on framework)
- Gradients silently become NaN
- Only noticed when checking model outputs, not during training
- By then, entire training run is wasted

---

### 15. **Unsafe Optimizer zero_grad()**

**Location:** Multiple locations (e.g., `weatherflow/training/flow_trainer.py:186`)

**Code:**
```python
self.optimizer.zero_grad()
```

**Problem:**
- Should use `zero_grad(set_to_none=True)` for better performance
- Current implementation sets gradients to zero tensors (memory allocation)
- `set_to_none=True` avoids allocation overhead

**Impact:**
- Slower training (minor)
- Higher memory usage
- Gradient accumulation bugs if mixed with code expecting None

**Why It's Hard to Diagnose:**
- Works correctly, just inefficient
- Performance difference is subtle
- Not a correctness issue, just a performance issue

---

## 🟠 MEDIUM SEVERITY ISSUES

### 16. **Coriolis Parameter Zero at Equator**

**Location:** `weatherflow/physics/losses.py:356-357`

**Code:**
```python
u_g = -dPhi_dy / (f + 1e-8)
v_g = dPhi_dx / (f + 1e-8)
```

**Problem:**
- Coriolis parameter `f` is zero at equator
- Epsilon `1e-8` is arbitrary
- Geostrophic approximation is invalid at equator anyway

**Impact:**
- Numerically unstable near equator
- Loss values are meaningless for equatorial regions
- Should mask out equator instead of adding epsilon

**Why It's Hard to Diagnose:**
- Only affects specific latitudes
- Error is domain-specific (physics knowledge required)
- Appears to work but produces wrong physics

---

### 17. **DataLoader num_workers=0 in Examples**

**Location:** `examples/flow_matching/era5_strict_training_loop.py:42`

**Code:**
```python
num_workers: int = 0
```

**Problem:**
- Single-threaded data loading
- Blocks training on data loading
- GPU underutilized

**Impact:**
- 2-4x slower training
- GPU utilization <50%

**Why It's Hard to Diagnose:**
- Still produces correct results
- Users may not notice inefficiency
- Common in examples for simplicity

---

### 18. **Missing Gradient Clipping Validation**

**Location:** `weatherflow/training/flow_trainer.py:189-191`

**Code:**
```python
if self.grad_clip is not None:
    self.scaler.unscale_(self.optimizer)
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
```

**Problem:**
- `grad_clip` can be negative or zero (no validation)
- Negative value crashes
- Zero disables clipping silently

**Impact:**
- Crashes with unclear error for negative values
- Silent behavior change for zero

**Why It's Hard to Diagnose:**
- Input validation missing
- Error is in PyTorch internals

---

### 19. **Resource Leak: Unclosed Datasets**

**Location:** `weatherflow/data/datasets.py:31`

**Code:**
```python
self.ds = xr.open_zarr(self.data_path, ...)  # Never closed
```

**Problem:**
- No `__del__` or context manager
- File handles remain open
- Memory not released

**Impact:**
- Memory leak over time
- File handle exhaustion with many datasets
- "Too many open files" error

**Why It's Hard to Diagnose:**
- Only manifests after many dataset creations
- Error message doesn't mention original allocation
- Memory profiling required to detect

---

### 20. **Hardcoded Configuration URLs**

**Location:** `weatherflow/data/datasets.py:206`

**Code:**
```python
DEFAULT_URL = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
```

**Problem:**
- No fallback if URL changes
- No validation that URL is accessible
- Fails silently if bucket permissions change

**Impact:**
- Dataset loading fails with unclear error
- No indication that URL is the problem

**Why It's Hard to Diagnose:**
- Error is network/permissions related
- Looks like user configuration issue

---

### 21. **Missing train/eval Mode Switches**

**Location:** Various (e.g., `weatherflow/utils/flow_visualization.py:22`)

**Code:**
```python
model.eval()
# ... visualization ...
# Missing: model.train() restoration
```

**Problem:**
- Model mode changed during visualization
- Never restored to original mode
- Affects subsequent training if called mid-training

**Impact:**
- Batch norm and dropout behavior changes
- Training diverges after visualization
- Requires deep understanding of PyTorch mode semantics

**Why It's Hard to Diagnose:**
- Effect is subtle behavior change
- Works fine if only doing inference
- Only problematic when mixed with training

---

### 22. **Arbitrary Numerical Constants**

**Location:** Multiple locations

**Examples:**
- `weatherflow/training/flow_trainer.py:29`: `min_velocity=5.0` - why 5.0?
- `weatherflow/physics/losses.py:103`: `cos_lat.clamp(min=1e-8)` - why 1e-8?
- `weatherflow/models/flow_matching.py:434`: `energy_x0 + 1e-6` - why 1e-6?

**Problem:**
- Magic numbers without justification
- Don't scale with data magnitude
- May cause issues with different data ranges

**Impact:**
- Numerical instability with different data scales
- Unclear how to tune these values

**Why It's Hard to Diagnose:**
- Works for default data range
- Fails mysteriously with scaled data
- No documentation on choosing values

---

## 📊 TESTING GAPS

### 23. **No Tests for Edge Cases**

**Missing Test Coverage:**
- Grid sizes: 0, 1, 2 dimensions
- Empty datasets (len=0, len=1)
- NaN/Inf in inputs
- Device mismatches (CPU vs CUDA)
- Time values at boundaries (t=0, t=1)
- Concurrent access patterns
- Large batch sizes (OOM scenarios)
- Missing files and network failures

**Impact:**
- Edge case bugs only found in production
- No regression tests for fixes

---

## 🔧 RECOMMENDATIONS

### Immediate Fixes (Critical)

1. **Fix `weatherflow/__init__.py` docstring** - blocking all imports
2. **Add epsilon to flow matching denominator** - preventing NaN gradients
3. **Validate grid dimensions in physics losses** - preventing division by zero
4. **Add thread locks to `app.py`** - preventing race conditions
5. **Validate TimeEncoder dim >= 4** - preventing division by zero

### High Priority

1. **Add NaN/Inf checking after every physics loss computation**
2. **Restore model state after EMA evaluation**
3. **Add input validation to all dataset classes**
4. **Implement proper error handling for ERA5 downloads**

### Medium Priority

1. **Add type hints and runtime type checking**
2. **Implement context managers for dataset cleanup**
3. **Add configuration validation with clear error messages**
4. **Use `zero_grad(set_to_none=True)` for efficiency**

### Long Term

1. **Comprehensive edge case test suite**
2. **Input fuzzing for physics losses**
3. **Memory leak detection in CI**
4. **Load testing for concurrent API access**
5. **Documentation of all numerical constants with justification**

---

## 🔍 DETECTION STRATEGIES

### How to Find Similar Issues

1. **Numerical Stability:**
   ```bash
   grep -r "/ (.*)" --include="*.py" | grep -v "1e-"  # Division without epsilon
   grep -r "torch.sqrt\|torch.log\|torch.div" --include="*.py"  # Ops that can produce NaN
   ```

2. **Type Safety:**
   ```bash
   mypy --strict weatherflow/  # Strict type checking
   grep -r "# type: ignore" --include="*.py"  # Ignored type errors
   ```

3. **Resource Management:**
   ```bash
   grep -r "open(" --include="*.py" | grep -v "with "  # Unclosed files
   grep -r "\.to(device)" --include="*.py"  # Device transfers
   ```

4. **Threading Issues:**
   ```bash
   grep -r "threading\|multiprocessing" --include="*.py"  # Thread usage
   grep -r "global " --include="*.py"  # Global state
   ```

5. **Silent Failures:**
   ```bash
   grep -r "except.*pass" --include="*.py"  # Caught but ignored
   grep -r "print(" --include="*.py" | grep -i "warn\|error"  # Should use logging
   ```

---

## 📝 VERIFICATION

To verify these issues:

```bash
# Issue 1: Import failure
python3 -c "import weatherflow"

# Issue 2: NaN gradients
python3 -c "import torch; t=torch.tensor([1.0]); print((10-5)/(1-t))"

# Issue 3: Division by zero
python3 -c "import numpy as np; print(np.pi / (1-1))"

# Issue 4: TimeEncoder
python3 -c "from weatherflow.models.flow_matching import TimeEncoder; TimeEncoder(2)"

# Issue 5: Race condition (requires load testing)
ab -n 100 -c 10 http://localhost:5000/api/run
```

---

**Document End**
