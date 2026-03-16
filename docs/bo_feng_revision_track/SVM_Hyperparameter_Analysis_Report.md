# Nonparametric Identification Analysis: SVM Hyperparameter Sensitivity Report

**Date:** Based on runtime logs from October 2025  
**Objective:** Evaluate sensitivity of identified sets to SVM hyperparameters across different support sizes (s=5, s=10, s=20) and iteration counts

---

## Executive Summary

This report documents the implementation and results of post-estimation SVM hyperparameter tuning for nonparametric identification analysis. The analysis focuses on:

1. **SVM hyperparameter sweep** without re-running estimation
2. **Visualization of full parameter grid** (10k evaluated points) to assess coverage
3. **Comparison across different s values** (s=5, s=10, s=20) to examine how identified sets change with support size

**Key implementation changes:**
- Explicit PGx evaluation domain derived from evaluated points (data-driven bounds)
- Post-estimation SVM hyperparameter sweep: kernels (gaussian, linear), scales (auto, 1), box constraints (0.1, 1, 10)
- Full-grid visualization showing all 10k evaluated parameter combinations
- SVM settings embedded in figure titles and filenames for easy comparison

---

## Methodology

### Code Modifications

The main script `II_MAIN_nonparam_simul.m` was updated to:

1. **Decouple PGx domain from iteration count**: Bounds for the evaluation grid are now derived from the actual evaluated points using percentiles (1-99) with 10% padding, rather than being hardcoded per iteration count.

2. **Full-grid visualization**: Added scatter plot of all evaluated parameter combinations, colored by membership in the identified set.

3. **SVM hyperparameter sweep**: After estimation, systematically evaluate different SVM configurations:
   - **Kernels**: `gaussian`, `linear`
   - **Kernel scales**: `'auto'`, `1`
   - **Box constraints**: `0.1`, `1`, `10`
   - Total: 2 × 2 × 3 = **12 configurations per (s, iteration)**

4. **Figure naming convention**: 
   - Full grid: `nonparam_FullGrid_s{S}_{ITER}k.png`
   - SVM results: `nonparam_IdSet_simul_s{S}_{ITER}k_kernel={K}_scale={SCALE}_box={BOX}.png`

### Experiment Configuration

- **Support sizes tested**: s = 5, 10, 20
- **Iteration counts**: 500k, 1000k, 2000k, 4000k
- **Grid size**: NGrid = 10,000 evaluated parameter combinations
- **PGx evaluation density**: 200,000 points (Halton sequence)

---

## Code Changes Summary

The following changes were made to `II_MAIN_nonparam_simul.m` to implement the SVM hyperparameter analysis and visualization improvements:

### 1. Explicit PGx Evaluation Domain
- **Location**: Lines ~276-292
- **Change**: Replaced hardcoded, iteration-dependent PGx bounds with data-driven bounds
  - **Before**: PGx bounds were hardcoded per `maxiter_index` (different scaling factors: 7.5×10 for 500k, 7×6 for 1000k, 6.5×3.5 for 2000k/4000k)
  - **After**: Bounds derived from evaluated parameter grid using percentiles (1-99) with 10% padding
  - Computes `mu_min`, `mu_max`, `sigma_min`, `sigma_max` from `ddpars` (evaluated parameters)
  - Ensures consistent evaluation domain across different iteration counts for comparable visualizations
- **Code example**:
  ```matlab
  mu_vals = ddpars(1,:);
  sig_vals = ddpars(2,:);
  mu_lo = prctile(mu_vals,1); mu_hi = prctile(mu_vals,99);
  sg_lo = prctile(sig_vals,1); sg_hi = prctile(sig_vals,99);
  mu_rng = max(mu_hi - mu_lo, eps);
  sg_rng = max(sg_hi - sg_lo, eps);
  mu_min = mu_lo - 0.1*mu_rng; mu_max = mu_hi + 0.1*mu_rng;
  sigma_min = max(0, sg_lo - 0.1*sg_rng); sigma_max = sg_hi + 0.1*sg_rng;
  ```

### 2. Full Parameter Grid Visualization
- **Location**: Lines ~294-304
- **Change**: Added new figure showing all evaluated parameter combinations
  - Scatter plot of all 10k evaluated points colored by membership in identified set
  - Black dots: Parameter combinations NOT in the identified set
  - Cyan dots: Parameter combinations IN the identified set
  - Includes true parameter marker at (μ≈3, σ≈1)
  - Saved as `nonparam_FullGrid_s{S}_{ITER}k.png/eps`
  - Enables assessment of parameter space coverage and identification of sparse regions

### 3. SVM Hyperparameter Sweep Implementation
- **Location**: Lines ~306-380
- **Change**: Implemented systematic post-estimation SVM hyperparameter sweep
  - **Nested loops** over: kernels (`gaussian`, `linear`), kernel scales (`'auto'`, `1`), box constraints (`0.1`, `1`, `10`)
  - **Total configurations**: 2 × 2 × 3 = **12 per (s, iteration)**
  - **Training**: Uses same `Xtrain = ddpars'` and `YTrain = id_set_index` (no re-estimation required)
  - **Prediction**: Evaluates on explicit PGx domain for each configuration
  - **Visualization**: Each configuration generates a separate figure with SVM-predicted identified set
  - Overlays evaluated points (gray) on SVM prediction (cyan region)
  - Includes range markers (red horizontal bar for μ range, green vertical bar for σ range)

### 4. Enhanced Figure Titles and Filenames
- **Location**: Lines ~370-376
- **Change**: Added SVM hyperparameter information to figure titles and filenames
  - **Title format**: `Id set via SVM (kernel={K},scale={SCALE},box={BOX}) | s={S}, iter={ITER}k`
  - **Filename format**: `nonparam_IdSet_simul_s{S}_{ITER}k_kernel={K}_scale={SCALE}_box={BOX}.png/eps`
  - Uses `regexprep` to sanitize hyperparameter strings for valid filenames (replaces special characters with underscores)
  - Enables easy identification and comparison of figures across different SVM settings

### 5. Removed Legacy Code
- **Location**: Lines ~465-466 (now removed)
- **Change**: Removed duplicate saveas calls that created generic figure filenames without hyperparameter information
  - **Before**: Both generic (`nonparam_IdSet_simul_s{S}_{ITER}k.png`) and hyperparameter-specific figures were saved
  - **After**: Only hyperparameter-specific figures are saved for clarity and consistency

### Code Structure Notes
- **Estimation phase** (lines ~130-250): Unchanged — all modifications occur post-estimation
- **Visualization phase** (lines ~254-380): New code added for full-grid visualization and SVM sweep
- **No re-estimation**: All new functionality operates on existing results (`VV_all`, `distribution_parameters_all`)
- **Modularity**: The SVM sweep can be easily extended (e.g., adding polynomial kernels or additional scale/box constraint values) without modifying the estimation code

---

## Results by Support Size

### s = 5

#### Iteration: 500k

**Full Parameter Grid:**
![Full Grid s=5, 500k](nonparam_FullGrid_s5_500k.png)

**How to Read These Figures:**

1. **Full Parameter Grid Figure (above)**: This scatter plot shows all 10,000 evaluated parameter combinations in the (μ, σ) space:
   - **Black dots**: Parameter combinations that do NOT belong to the identified set (rejected by the moment/test conditions)
   - **Cyan dots**: Parameter combinations that DO belong to the identified set (consistent with observed data)
   - **Black marker at (μ≈3, σ≈1)**: The true/reference parameter value used in the simulation
   - This figure helps assess **coverage** - whether the 10k parameter grid adequately explores the space and whether there are sparse regions

2. **SVM IdSet Figures (below)**: Each figure shows the SVM-predicted identified set for a specific hyperparameter configuration:
   - **Cyan shaded region**: The continuous region where the SVM predicts parameter combinations belong to the identified set
   - **Gray dots**: The dense evaluation grid (PGx) showing where we evaluated the SVM
   - Compare across different SVM settings to see which produce more "sensible" (convex, connected) identified sets

**SVM Hyperparameter Comparison:**

| Kernel | Scale | Box Constraint | Figure |
|--------|-------|---------------|--------|
| gaussian | auto | 0.1 | ![kernel=gaussian,scale=auto,box=0.1](nonparam_IdSet_simul_s5_500k_kernel=gaussian_scale=_auto__box=0.1.png) |
| gaussian | auto | 1 | ![kernel=gaussian,scale=auto,box=1](nonparam_IdSet_simul_s5_500k_kernel=gaussian_scale=_auto__box=1.png) |
| gaussian | auto | 10 | ![kernel=gaussian,scale=auto,box=10](nonparam_IdSet_simul_s5_500k_kernel=gaussian_scale=_auto__box=10.png) |
| gaussian | 1 | 0.1 | ![kernel=gaussian,scale=1,box=0.1](nonparam_IdSet_simul_s5_500k_kernel=gaussian_scale=1_box=0.1.png) |
| gaussian | 1 | 1 | ![kernel=gaussian,scale=1,box=1](nonparam_IdSet_simul_s5_500k_kernel=gaussian_scale=1_box=1.png) |
| gaussian | 1 | 10 | ![kernel=gaussian,scale=1,box=10](nonparam_IdSet_simul_s5_500k_kernel=gaussian_scale=1_box=10.png) |
| linear | auto | 0.1 | ![kernel=linear,scale=auto,box=0.1](nonparam_IdSet_simul_s5_500k_kernel=linear_scale=_auto__box=0.1.png) |
| linear | auto | 1 | ![kernel=linear,scale=auto,box=1](nonparam_IdSet_simul_s5_500k_kernel=linear_scale=_auto__box=1.png) |
| linear | auto | 10 | ![kernel=linear,scale=auto,box=10](nonparam_IdSet_simul_s5_500k_kernel=linear_scale=_auto__box=10.png) |
| linear | 1 | 0.1 | ![kernel=linear,scale=1,box=0.1](nonparam_IdSet_simul_s5_500k_kernel=linear_scale=1_box=0.1.png) |
| linear | 1 | 1 | ![kernel=linear,scale=1,box=1](nonparam_IdSet_simul_s5_500k_kernel=linear_scale=1_box=1.png) |
| linear | 1 | 10 | ![kernel=linear,scale=1,box=10](nonparam_IdSet_simul_s5_500k_kernel=linear_scale=1_box=10.png) |

---

#### Iteration: 1000k

**Full Parameter Grid:**
![Full Grid s=5, 1000k](nonparam_FullGrid_s5_1000k.png)

**How to Read These Figures:**

1. **Full Parameter Grid Figure (above)**: This scatter plot shows all 10,000 evaluated parameter combinations in the (μ, σ) space:
   - **Black dots**: Parameter combinations that do NOT belong to the identified set (rejected by the moment/test conditions)
   - **Cyan dots**: Parameter combinations that DO belong to the identified set (consistent with observed data)
   - **Black marker at (μ≈3, σ≈1)**: The true/reference parameter value used in the simulation
   - This figure helps assess **coverage** - whether the 10k parameter grid adequately explores the space and whether there are sparse regions

2. **SVM IdSet Figures (below)**: Each figure shows the SVM-predicted identified set for a specific hyperparameter configuration:
   - **Cyan shaded region**: The continuous region where the SVM predicts parameter combinations belong to the identified set
   - **Gray dots**: The dense evaluation grid (PGx) showing where we evaluated the SVM
   - Compare across different SVM settings to see which produce more "sensible" (convex, connected) identified sets

**SVM Hyperparameter Comparison:**

| Kernel | Scale | Box Constraint | Figure |
|--------|-------|---------------|--------|
| gaussian | auto | 0.1 | ![kernel=gaussian,scale=auto,box=0.1](nonparam_IdSet_simul_s5_1000k_kernel=gaussian_scale=_auto__box=0.1.png) |
| gaussian | auto | 1 | ![kernel=gaussian,scale=auto,box=1](nonparam_IdSet_simul_s5_1000k_kernel=gaussian_scale=_auto__box=1.png) |
| gaussian | auto | 10 | ![kernel=gaussian,scale=auto,box=10](nonparam_IdSet_simul_s5_1000k_kernel=gaussian_scale=_auto__box=10.png) |
| gaussian | 1 | 0.1 | ![kernel=gaussian,scale=1,box=0.1](nonparam_IdSet_simul_s5_1000k_kernel=gaussian_scale=1_box=0.1.png) |
| gaussian | 1 | 1 | ![kernel=gaussian,scale=1,box=1](nonparam_IdSet_simul_s5_1000k_kernel=gaussian_scale=1_box=1.png) |
| gaussian | 1 | 10 | ![kernel=gaussian,scale=1,box=10](nonparam_IdSet_simul_s5_1000k_kernel=gaussian_scale=1_box=10.png) |
| linear | auto | 0.1 | ![kernel=linear,scale=auto,box=0.1](nonparam_IdSet_simul_s5_1000k_kernel=linear_scale=_auto__box=0.1.png) |
| linear | auto | 1 | ![kernel=linear,scale=auto,box=1](nonparam_IdSet_simul_s5_1000k_kernel=linear_scale=_auto__box=1.png) |
| linear | auto | 10 | ![kernel=linear,scale=auto,box=10](nonparam_IdSet_simul_s5_1000k_kernel=linear_scale=_auto__box=10.png) |
| linear | 1 | 0.1 | ![kernel=linear,scale=1,box=0.1](nonparam_IdSet_simul_s5_1000k_kernel=linear_scale=1_box=0.1.png) |
| linear | 1 | 1 | ![kernel=linear,scale=1,box=1](nonparam_IdSet_simul_s5_1000k_kernel=linear_scale=1_box=1.png) |
| linear | 1 | 10 | ![kernel=linear,scale=1,box=10](nonparam_IdSet_simul_s5_1000k_kernel=linear_scale=1_box=10.png) |

---

#### Iteration: 2000k

**Full Parameter Grid:**
![Full Grid s=5, 2000k](nonparam_FullGrid_s5_2000k.png)

**How to Read These Figures:**

1. **Full Parameter Grid Figure (above)**: This scatter plot shows all 10,000 evaluated parameter combinations in the (μ, σ) space:
   - **Black dots**: Parameter combinations that do NOT belong to the identified set (rejected by the moment/test conditions)
   - **Cyan dots**: Parameter combinations that DO belong to the identified set (consistent with observed data)
   - **Black marker at (μ≈3, σ≈1)**: The true/reference parameter value used in the simulation
   - This figure helps assess **coverage** - whether the 10k parameter grid adequately explores the space and whether there are sparse regions

2. **SVM IdSet Figures (below)**: Each figure shows the SVM-predicted identified set for a specific hyperparameter configuration:
   - **Cyan shaded region**: The continuous region where the SVM predicts parameter combinations belong to the identified set
   - **Gray dots**: The dense evaluation grid (PGx) showing where we evaluated the SVM
   - Compare across different SVM settings to see which produce more "sensible" (convex, connected) identified sets

**SVM Hyperparameter Comparison:**

| Kernel | Scale | Box Constraint | Figure |
|--------|-------|---------------|--------|
| gaussian | auto | 0.1 | ![kernel=gaussian,scale=auto,box=0.1](nonparam_IdSet_simul_s5_2000k_kernel=gaussian_scale=_auto__box=0.1.png) |
| gaussian | auto | 1 | ![kernel=gaussian,scale=auto,box=1](nonparam_IdSet_simul_s5_2000k_kernel=gaussian_scale=_auto__box=1.png) |
| gaussian | auto | 10 | ![kernel=gaussian,scale=auto,box=10](nonparam_IdSet_simul_s5_2000k_kernel=gaussian_scale=_auto__box=10.png) |
| gaussian | 1 | 0.1 | ![kernel=gaussian,scale=1,box=0.1](nonparam_IdSet_simul_s5_2000k_kernel=gaussian_scale=1_box=0.1.png) |
| gaussian | 1 | 1 | ![kernel=gaussian,scale=1,box=1](nonparam_IdSet_simul_s5_2000k_kernel=gaussian_scale=1_box=1.png) |
| gaussian | 1 | 10 | ![kernel=gaussian,scale=1,box=10](nonparam_IdSet_simul_s5_2000k_kernel=gaussian_scale=1_box=10.png) |
| linear | auto | 0.1 | ![kernel=linear,scale=auto,box=0.1](nonparam_IdSet_simul_s5_2000k_kernel=linear_scale=_auto__box=0.1.png) |
| linear | auto | 1 | ![kernel=linear,scale=auto,box=1](nonparam_IdSet_simul_s5_2000k_kernel=linear_scale=_auto__box=1.png) |
| linear | auto | 10 | ![kernel=linear,scale=auto,box=10](nonparam_IdSet_simul_s5_2000k_kernel=linear_scale=_auto__box=10.png) |
| linear | 1 | 0.1 | ![kernel=linear,scale=1,box=0.1](nonparam_IdSet_simul_s5_2000k_kernel=linear_scale=1_box=0.1.png) |
| linear | 1 | 1 | ![kernel=linear,scale=1,box=1](nonparam_IdSet_simul_s5_2000k_kernel=linear_scale=1_box=1.png) |
| linear | 1 | 10 | ![kernel=linear,scale=1,box=10](nonparam_IdSet_simul_s5_2000k_kernel=linear_scale=1_box=10.png) |

---

#### Iteration: 4000k

**Full Parameter Grid:**
![Full Grid s=5, 4000k](nonparam_FullGrid_s5_4000k.png)

**How to Read These Figures:**

1. **Full Parameter Grid Figure (above)**: This scatter plot shows all 10,000 evaluated parameter combinations in the (μ, σ) space:
   - **Black dots**: Parameter combinations that do NOT belong to the identified set (rejected by the moment/test conditions)
   - **Cyan dots**: Parameter combinations that DO belong to the identified set (consistent with observed data)
   - **Black marker at (μ≈3, σ≈1)**: The true/reference parameter value used in the simulation
   - This figure helps assess **coverage** - whether the 10k parameter grid adequately explores the space and whether there are sparse regions

2. **SVM IdSet Figures (below)**: Each figure shows the SVM-predicted identified set for a specific hyperparameter configuration:
   - **Cyan shaded region**: The continuous region where the SVM predicts parameter combinations belong to the identified set
   - **Gray dots**: The dense evaluation grid (PGx) showing where we evaluated the SVM
   - Compare across different SVM settings to see which produce more "sensible" (convex, connected) identified sets

**SVM Hyperparameter Comparison:**

| Kernel | Scale | Box Constraint | Figure |
|--------|-------|---------------|--------|
| gaussian | auto | 0.1 | ![kernel=gaussian,scale=auto,box=0.1](nonparam_IdSet_simul_s5_4000k_kernel=gaussian_scale=_auto__box=0.1.png) |
| gaussian | auto | 1 | ![kernel=gaussian,scale=auto,box=1](nonparam_IdSet_simul_s5_4000k_kernel=gaussian_scale=_auto__box=1.png) |
| gaussian | auto | 10 | ![kernel=gaussian,scale=auto,box=10](nonparam_IdSet_simul_s5_4000k_kernel=gaussian_scale=_auto__box=10.png) |
| gaussian | 1 | 0.1 | ![kernel=gaussian,scale=1,box=0.1](nonparam_IdSet_simul_s5_4000k_kernel=gaussian_scale=1_box=0.1.png) |
| gaussian | 1 | 1 | ![kernel=gaussian,scale=1,box=1](nonparam_IdSet_simul_s5_4000k_kernel=gaussian_scale=1_box=1.png) |
| gaussian | 1 | 10 | ![kernel=gaussian,scale=1,box=10](nonparam_IdSet_simul_s5_4000k_kernel=gaussian_scale=1_box=10.png) |
| linear | auto | 0.1 | ![kernel=linear,scale=auto,box=0.1](nonparam_IdSet_simul_s5_4000k_kernel=linear_scale=_auto__box=0.1.png) |
| linear | auto | 1 | ![kernel=linear,scale=auto,box=1](nonparam_IdSet_simul_s5_4000k_kernel=linear_scale=_auto__box=1.png) |
| linear | auto | 10 | ![kernel=linear,scale=auto,box=10](nonparam_IdSet_simul_s5_4000k_kernel=linear_scale=_auto__box=10.png) |
| linear | 1 | 0.1 | ![kernel=linear,scale=1,box=0.1](nonparam_IdSet_simul_s5_4000k_kernel=linear_scale=1_box=0.1.png) |
| linear | 1 | 1 | ![kernel=linear,scale=1,box=1](nonparam_IdSet_simul_s5_4000k_kernel=linear_scale=1_box=1.png) |
| linear | 1 | 10 | ![kernel=linear,scale=1,box=10](nonparam_IdSet_simul_s5_4000k_kernel=linear_scale=1_box=10.png) |

---

### s = 10

#### Iteration: 500k

**Full Parameter Grid:**
![Full Grid s=10, 500k](nonparam_FullGrid_s10_500k.png)

**How to Read These Figures:**

1. **Full Parameter Grid Figure (above)**: This scatter plot shows all 10,000 evaluated parameter combinations in the (μ, σ) space:
   - **Black dots**: Parameter combinations that do NOT belong to the identified set (rejected by the moment/test conditions)
   - **Cyan dots**: Parameter combinations that DO belong to the identified set (consistent with observed data)
   - **Black marker at (μ≈3, σ≈1)**: The true/reference parameter value used in the simulation
   - This figure helps assess **coverage** - whether the 10k parameter grid adequately explores the space and whether there are sparse regions

2. **SVM IdSet Figures (below)**: Each figure shows the SVM-predicted identified set for a specific hyperparameter configuration:
   - **Cyan shaded region**: The continuous region where the SVM predicts parameter combinations belong to the identified set
   - **Gray dots**: The dense evaluation grid (PGx) showing where we evaluated the SVM

**SVM Hyperparameter Comparison:**

| Kernel | Scale | Box Constraint | Figure |
|--------|-------|---------------|--------|
| gaussian | auto | 0.1 | ![kernel=gaussian,scale=auto,box=0.1](nonparam_IdSet_simul_s10_500k_kernel=gaussian_scale=_auto__box=0.1.png) |
| gaussian | auto | 1 | ![kernel=gaussian,scale=auto,box=1](nonparam_IdSet_simul_s10_500k_kernel=gaussian_scale=_auto__box=1.png) |
| gaussian | auto | 10 | ![kernel=gaussian,scale=auto,box=10](nonparam_IdSet_simul_s10_500k_kernel=gaussian_scale=_auto__box=10.png) |
| gaussian | 1 | 0.1 | ![kernel=gaussian,scale=1,box=0.1](nonparam_IdSet_simul_s10_500k_kernel=gaussian_scale=1_box=0.1.png) |
| gaussian | 1 | 1 | ![kernel=gaussian,scale=1,box=1](nonparam_IdSet_simul_s10_500k_kernel=gaussian_scale=1_box=1.png) |
| gaussian | 1 | 10 | ![kernel=gaussian,scale=1,box=10](nonparam_IdSet_simul_s10_500k_kernel=gaussian_scale=1_box=10.png) |
| linear | auto | 0.1 | ![kernel=linear,scale=auto,box=0.1](nonparam_IdSet_simul_s10_500k_kernel=linear_scale=_auto__box=0.1.png) |
| linear | auto | 1 | ![kernel=linear,scale=auto,box=1](nonparam_IdSet_simul_s10_500k_kernel=linear_scale=_auto__box=1.png) |
| linear | auto | 10 | ![kernel=linear,scale=auto,box=10](nonparam_IdSet_simul_s10_500k_kernel=linear_scale=_auto__box=10.png) |
| linear | 1 | 0.1 | ![kernel=linear,scale=1,box=0.1](nonparam_IdSet_simul_s10_500k_kernel=linear_scale=1_box=0.1.png) |
| linear | 1 | 1 | ![kernel=linear,scale=1,box=1](nonparam_IdSet_simul_s10_500k_kernel=linear_scale=1_box=1.png) |
| linear | 1 | 10 | ![kernel=linear,scale=1,box=10](nonparam_IdSet_simul_s10_500k_kernel=linear_scale=1_box=10.png) |

---

#### Iteration: 1000k

**Full Parameter Grid:**
![Full Grid s=10, 1000k](nonparam_FullGrid_s10_1000k.png)

**How to Read These Figures:**

1. **Full Parameter Grid Figure (above)**: This scatter plot shows all 10,000 evaluated parameter combinations in the (μ, σ) space:
   - **Black dots**: Parameter combinations that do NOT belong to the identified set (rejected by the moment/test conditions)
   - **Cyan dots**: Parameter combinations that DO belong to the identified set (consistent with observed data)
   - **Black marker at (μ≈3, σ≈1)**: The true/reference parameter value used in the simulation
   - This figure helps assess **coverage** - whether the 10k parameter grid adequately explores the space and whether there are sparse regions

2. **SVM IdSet Figures (below)**: Each figure shows the SVM-predicted identified set for a specific hyperparameter configuration:
   - **Cyan shaded region**: The continuous region where the SVM predicts parameter combinations belong to the identified set
   - **Gray dots**: The dense evaluation grid (PGx) showing where we evaluated the SVM
   - Compare across different SVM settings to see which produce more "sensible" (convex, connected) identified sets

**SVM Hyperparameter Comparison:**

| Kernel | Scale | Box Constraint | Figure |
|--------|-------|---------------|--------|
| gaussian | auto | 0.1 | ![kernel=gaussian,scale=auto,box=0.1](nonparam_IdSet_simul_s10_1000k_kernel=gaussian_scale=_auto__box=0.1.png) |
| gaussian | auto | 1 | ![kernel=gaussian,scale=auto,box=1](nonparam_IdSet_simul_s10_1000k_kernel=gaussian_scale=_auto__box=1.png) |
| gaussian | auto | 10 | ![kernel=gaussian,scale=auto,box=10](nonparam_IdSet_simul_s10_1000k_kernel=gaussian_scale=_auto__box=10.png) |
| gaussian | 1 | 0.1 | ![kernel=gaussian,scale=1,box=0.1](nonparam_IdSet_simul_s10_1000k_kernel=gaussian_scale=1_box=0.1.png) |
| gaussian | 1 | 1 | ![kernel=gaussian,scale=1,box=1](nonparam_IdSet_simul_s10_1000k_kernel=gaussian_scale=1_box=1.png) |
| gaussian | 1 | 10 | ![kernel=gaussian,scale=1,box=10](nonparam_IdSet_simul_s10_1000k_kernel=gaussian_scale=1_box=10.png) |
| linear | auto | 0.1 | ![kernel=linear,scale=auto,box=0.1](nonparam_IdSet_simul_s10_1000k_kernel=linear_scale=_auto__box=0.1.png) |
| linear | auto | 1 | ![kernel=linear,scale=auto,box=1](nonparam_IdSet_simul_s10_1000k_kernel=linear_scale=_auto__box=1.png) |
| linear | auto | 10 | ![kernel=linear,scale=auto,box=10](nonparam_IdSet_simul_s10_1000k_kernel=linear_scale=_auto__box=10.png) |
| linear | 1 | 0.1 | ![kernel=linear,scale=1,box=0.1](nonparam_IdSet_simul_s10_1000k_kernel=linear_scale=1_box=0.1.png) |
| linear | 1 | 1 | ![kernel=linear,scale=1,box=1](nonparam_IdSet_simul_s10_1000k_kernel=linear_scale=1_box=1.png) |
| linear | 1 | 10 | ![kernel=linear,scale=1,box=10](nonparam_IdSet_simul_s10_1000k_kernel=linear_scale=1_box=10.png) |

---

#### Iteration: 2000k

**Full Parameter Grid:**
![Full Grid s=10, 2000k](nonparam_FullGrid_s10_2000k.png)

**How to Read These Figures:**

1. **Full Parameter Grid Figure (above)**: This scatter plot shows all 10,000 evaluated parameter combinations in the (μ, σ) space:
   - **Black dots**: Parameter combinations that do NOT belong to the identified set (rejected by the moment/test conditions)
   - **Cyan dots**: Parameter combinations that DO belong to the identified set (consistent with observed data)
   - **Black marker at (μ≈3, σ≈1)**: The true/reference parameter value used in the simulation
   - This figure helps assess **coverage** - whether the 10k parameter grid adequately explores the space and whether there are sparse regions

2. **SVM IdSet Figures (below)**: Each figure shows the SVM-predicted identified set for a specific hyperparameter configuration:
   - **Cyan shaded region**: The continuous region where the SVM predicts parameter combinations belong to the identified set
   - **Gray dots**: The dense evaluation grid (PGx) showing where we evaluated the SVM
   - Compare across different SVM settings to see which produce more "sensible" (convex, connected) identified sets

**SVM Hyperparameter Comparison:**

| Kernel | Scale | Box Constraint | Figure |
|--------|-------|---------------|--------|
| gaussian | auto | 0.1 | ![kernel=gaussian,scale=auto,box=0.1](nonparam_IdSet_simul_s10_2000k_kernel=gaussian_scale=_auto__box=0.1.png) |
| gaussian | auto | 1 | ![kernel=gaussian,scale=auto,box=1](nonparam_IdSet_simul_s10_2000k_kernel=gaussian_scale=_auto__box=1.png) |
| gaussian | auto | 10 | ![kernel=gaussian,scale=auto,box=10](nonparam_IdSet_simul_s10_2000k_kernel=gaussian_scale=_auto__box=10.png) |
| gaussian | 1 | 0.1 | ![kernel=gaussian,scale=1,box=0.1](nonparam_IdSet_simul_s10_2000k_kernel=gaussian_scale=1_box=0.1.png) |
| gaussian | 1 | 1 | ![kernel=gaussian,scale=1,box=1](nonparam_IdSet_simul_s10_2000k_kernel=gaussian_scale=1_box=1.png) |
| gaussian | 1 | 10 | ![kernel=gaussian,scale=1,box=10](nonparam_IdSet_simul_s10_2000k_kernel=gaussian_scale=1_box=10.png) |
| linear | auto | 0.1 | ![kernel=linear,scale=auto,box=0.1](nonparam_IdSet_simul_s10_2000k_kernel=linear_scale=_auto__box=0.1.png) |
| linear | auto | 1 | ![kernel=linear,scale=auto,box=1](nonparam_IdSet_simul_s10_2000k_kernel=linear_scale=_auto__box=1.png) |
| linear | auto | 10 | ![kernel=linear,scale=auto,box=10](nonparam_IdSet_simul_s10_2000k_kernel=linear_scale=_auto__box=10.png) |
| linear | 1 | 0.1 | ![kernel=linear,scale=1,box=0.1](nonparam_IdSet_simul_s10_2000k_kernel=linear_scale=1_box=0.1.png) |
| linear | 1 | 1 | ![kernel=linear,scale=1,box=1](nonparam_IdSet_simul_s10_2000k_kernel=linear_scale=1_box=1.png) |
| linear | 1 | 10 | ![kernel=linear,scale=1,box=10](nonparam_IdSet_simul_s10_2000k_kernel=linear_scale=1_box=10.png) |

---

#### Iteration: 4000k

**Full Parameter Grid:**
![Full Grid s=10, 4000k](nonparam_FullGrid_s10_4000k.png)

**How to Read These Figures:**

1. **Full Parameter Grid Figure (above)**: This scatter plot shows all 10,000 evaluated parameter combinations in the (μ, σ) space:
   - **Black dots**: Parameter combinations that do NOT belong to the identified set (rejected by the moment/test conditions)
   - **Cyan dots**: Parameter combinations that DO belong to the identified set (consistent with observed data)
   - **Black marker at (μ≈3, σ≈1)**: The true/reference parameter value used in the simulation
   - This figure helps assess **coverage** - whether the 10k parameter grid adequately explores the space and whether there are sparse regions

2. **SVM IdSet Figures (below)**: Each figure shows the SVM-predicted identified set for a specific hyperparameter configuration:
   - **Cyan shaded region**: The continuous region where the SVM predicts parameter combinations belong to the identified set
   - **Gray dots**: The dense evaluation grid (PGx) showing where we evaluated the SVM
   - Compare across different SVM settings to see which produce more "sensible" (convex, connected) identified sets

**SVM Hyperparameter Comparison:**

| Kernel | Scale | Box Constraint | Figure |
|--------|-------|---------------|--------|
| gaussian | auto | 0.1 | ![kernel=gaussian,scale=auto,box=0.1](nonparam_IdSet_simul_s10_4000k_kernel=gaussian_scale=_auto__box=0.1.png) |
| gaussian | auto | 1 | ![kernel=gaussian,scale=auto,box=1](nonparam_IdSet_simul_s10_4000k_kernel=gaussian_scale=_auto__box=1.png) |
| gaussian | auto | 10 | ![kernel=gaussian,scale=auto,box=10](nonparam_IdSet_simul_s10_4000k_kernel=gaussian_scale=_auto__box=10.png) |
| gaussian | 1 | 0.1 | ![kernel=gaussian,scale=1,box=0.1](nonparam_IdSet_simul_s10_4000k_kernel=gaussian_scale=1_box=0.1.png) |
| gaussian | 1 | 1 | ![kernel=gaussian,scale=1,box=1](nonparam_IdSet_simul_s10_4000k_kernel=gaussian_scale=1_box=1.png) |
| gaussian | 1 | 10 | ![kernel=gaussian,scale=1,box=10](nonparam_IdSet_simul_s10_4000k_kernel=gaussian_scale=1_box=10.png) |
| linear | auto | 0.1 | ![kernel=linear,scale=auto,box=0.1](nonparam_IdSet_simul_s10_4000k_kernel=linear_scale=_auto__box=0.1.png) |
| linear | auto | 1 | ![kernel=linear,scale=auto,box=1](nonparam_IdSet_simul_s10_4000k_kernel=linear_scale=_auto__box=1.png) |
| linear | auto | 10 | ![kernel=linear,scale=auto,box=10](nonparam_IdSet_simul_s10_4000k_kernel=linear_scale=_auto__box=10.png) |
| linear | 1 | 0.1 | ![kernel=linear,scale=1,box=0.1](nonparam_IdSet_simul_s10_4000k_kernel=linear_scale=1_box=0.1.png) |
| linear | 1 | 1 | ![kernel=linear,scale=1,box=1](nonparam_IdSet_simul_s10_4000k_kernel=linear_scale=1_box=1.png) |
| linear | 1 | 10 | ![kernel=linear,scale=1,box=10](nonparam_IdSet_simul_s10_4000k_kernel=linear_scale=1_box=10.png) |

---

### s = 20

#### Iteration: 500k

**Full Parameter Grid:**
![Full Grid s=20, 500k](nonparam_FullGrid_s20_500k.png)

**How to Read These Figures:**

1. **Full Parameter Grid Figure (above)**: This scatter plot shows all 10,000 evaluated parameter combinations in the (μ, σ) space:
   - **Black dots**: Parameter combinations that do NOT belong to the identified set (rejected by the moment/test conditions)
   - **Cyan dots**: Parameter combinations that DO belong to the identified set (consistent with observed data)
   - **Black marker at (μ≈3, σ≈1)**: The true/reference parameter value used in the simulation
   - This figure helps assess **coverage** - whether the 10k parameter grid adequately explores the space and whether there are sparse regions

2. **SVM IdSet Figures (below)**: Each figure shows the SVM-predicted identified set for a specific hyperparameter configuration:
   - **Cyan shaded region**: The continuous region where the SVM predicts parameter combinations belong to the identified set
   - **Gray dots**: The dense evaluation grid (PGx) showing where we evaluated the SVM
   - Compare across different SVM settings to see which produce more "sensible" (convex, connected) identified sets

**SVM Hyperparameter Comparison:**

| Kernel | Scale | Box Constraint | Figure |
|--------|-------|---------------|--------|
| gaussian | auto | 0.1 | ![kernel=gaussian,scale=auto,box=0.1](nonparam_IdSet_simul_s20_500k_kernel=gaussian_scale=_auto__box=0.1.png) |
| gaussian | auto | 1 | ![kernel=gaussian,scale=auto,box=1](nonparam_IdSet_simul_s20_500k_kernel=gaussian_scale=_auto__box=1.png) |
| gaussian | auto | 10 | ![kernel=gaussian,scale=auto,box=10](nonparam_IdSet_simul_s20_500k_kernel=gaussian_scale=_auto__box=10.png) |
| gaussian | 1 | 0.1 | ![kernel=gaussian,scale=1,box=0.1](nonparam_IdSet_simul_s20_500k_kernel=gaussian_scale=1_box=0.1.png) |
| gaussian | 1 | 1 | ![kernel=gaussian,scale=1,box=1](nonparam_IdSet_simul_s20_500k_kernel=gaussian_scale=1_box=1.png) |
| gaussian | 1 | 10 | ![kernel=gaussian,scale=1,box=10](nonparam_IdSet_simul_s20_500k_kernel=gaussian_scale=1_box=10.png) |
| linear | auto | 0.1 | ![kernel=linear,scale=auto,box=0.1](nonparam_IdSet_simul_s20_500k_kernel=linear_scale=_auto__box=0.1.png) |
| linear | auto | 1 | ![kernel=linear,scale=auto,box=1](nonparam_IdSet_simul_s20_500k_kernel=linear_scale=_auto__box=1.png) |
| linear | auto | 10 | ![kernel=linear,scale=auto,box=10](nonparam_IdSet_simul_s20_500k_kernel=linear_scale=_auto__box=10.png) |
| linear | 1 | 0.1 | ![kernel=linear,scale=1,box=0.1](nonparam_IdSet_simul_s20_500k_kernel=linear_scale=1_box=0.1.png) |
| linear | 1 | 1 | ![kernel=linear,scale=1,box=1](nonparam_IdSet_simul_s20_500k_kernel=linear_scale=1_box=1.png) |
| linear | 1 | 10 | ![kernel=linear,scale=1,box=10](nonparam_IdSet_simul_s20_500k_kernel=linear_scale=1_box=10.png) |

---

#### Iteration: 1000k

**Full Parameter Grid:**
![Full Grid s=20, 1000k](nonparam_FullGrid_s20_1000k.png)

**How to Read These Figures:**

1. **Full Parameter Grid Figure (above)**: This scatter plot shows all 10,000 evaluated parameter combinations in the (μ, σ) space:
   - **Black dots**: Parameter combinations that do NOT belong to the identified set (rejected by the moment/test conditions)
   - **Cyan dots**: Parameter combinations that DO belong to the identified set (consistent with observed data)
   - **Black marker at (μ≈3, σ≈1)**: The true/reference parameter value used in the simulation
   - This figure helps assess **coverage** - whether the 10k parameter grid adequately explores the space and whether there are sparse regions

2. **SVM IdSet Figures (below)**: Each figure shows the SVM-predicted identified set for a specific hyperparameter configuration:
   - **Cyan shaded region**: The continuous region where the SVM predicts parameter combinations belong to the identified set
   - **Gray dots**: The dense evaluation grid (PGx) showing where we evaluated the SVM
   - Compare across different SVM settings to see which produce more "sensible" (convex, connected) identified sets

**SVM Hyperparameter Comparison:**

| Kernel | Scale | Box Constraint | Figure |
|--------|-------|---------------|--------|
| gaussian | auto | 0.1 | ![kernel=gaussian,scale=auto,box=0.1](nonparam_IdSet_simul_s20_1000k_kernel=gaussian_scale=_auto__box=0.1.png) |
| gaussian | auto | 1 | ![kernel=gaussian,scale=auto,box=1](nonparam_IdSet_simul_s20_1000k_kernel=gaussian_scale=_auto__box=1.png) |
| gaussian | auto | 10 | ![kernel=gaussian,scale=auto,box=10](nonparam_IdSet_simul_s20_1000k_kernel=gaussian_scale=_auto__box=10.png) |
| gaussian | 1 | 0.1 | ![kernel=gaussian,scale=1,box=0.1](nonparam_IdSet_simul_s20_1000k_kernel=gaussian_scale=1_box=0.1.png) |
| gaussian | 1 | 1 | ![kernel=gaussian,scale=1,box=1](nonparam_IdSet_simul_s20_1000k_kernel=gaussian_scale=1_box=1.png) |
| gaussian | 1 | 10 | ![kernel=gaussian,scale=1,box=10](nonparam_IdSet_simul_s20_1000k_kernel=gaussian_scale=1_box=10.png) |
| linear | auto | 0.1 | ![kernel=linear,scale=auto,box=0.1](nonparam_IdSet_simul_s20_1000k_kernel=linear_scale=_auto__box=0.1.png) |
| linear | auto | 1 | ![kernel=linear,scale=auto,box=1](nonparam_IdSet_simul_s20_1000k_kernel=linear_scale=_auto__box=1.png) |
| linear | auto | 10 | ![kernel=linear,scale=auto,box=10](nonparam_IdSet_simul_s20_1000k_kernel=linear_scale=_auto__box=10.png) |
| linear | 1 | 0.1 | ![kernel=linear,scale=1,box=0.1](nonparam_IdSet_simul_s20_1000k_kernel=linear_scale=1_box=0.1.png) |
| linear | 1 | 1 | ![kernel=linear,scale=1,box=1](nonparam_IdSet_simul_s20_1000k_kernel=linear_scale=1_box=1.png) |
| linear | 1 | 10 | ![kernel=linear,scale=1,box=10](nonparam_IdSet_simul_s20_1000k_kernel=linear_scale=1_box=10.png) |

---

#### Iteration: 2000k

**Full Parameter Grid:**
![Full Grid s=20, 2000k](nonparam_FullGrid_s20_2000k.png)

**How to Read These Figures:**

1. **Full Parameter Grid Figure (above)**: This scatter plot shows all 10,000 evaluated parameter combinations in the (μ, σ) space:
   - **Black dots**: Parameter combinations that do NOT belong to the identified set (rejected by the moment/test conditions)
   - **Cyan dots**: Parameter combinations that DO belong to the identified set (consistent with observed data)
   - **Black marker at (μ≈3, σ≈1)**: The true/reference parameter value used in the simulation
   - This figure helps assess **coverage** - whether the 10k parameter grid adequately explores the space and whether there are sparse regions

2. **SVM IdSet Figures (below)**: Each figure shows the SVM-predicted identified set for a specific hyperparameter configuration:
   - **Cyan shaded region**: The continuous region where the SVM predicts parameter combinations belong to the identified set
   - **Gray dots**: The dense evaluation grid (PGx) showing where we evaluated the SVM
   - Compare across different SVM settings to see which produce more "sensible" (convex, connected) identified sets

**SVM Hyperparameter Comparison:**

| Kernel | Scale | Box Constraint | Figure |
|--------|-------|---------------|--------|
| gaussian | auto | 0.1 | ![kernel=gaussian,scale=auto,box=0.1](nonparam_IdSet_simul_s20_2000k_kernel=gaussian_scale=_auto__box=0.1.png) |
| gaussian | auto | 1 | ![kernel=gaussian,scale=auto,box=1](nonparam_IdSet_simul_s20_2000k_kernel=gaussian_scale=_auto__box=1.png) |
| gaussian | auto | 10 | ![kernel=gaussian,scale=auto,box=10](nonparam_IdSet_simul_s20_2000k_kernel=gaussian_scale=_auto__box=10.png) |
| gaussian | 1 | 0.1 | ![kernel=gaussian,scale=1,box=0.1](nonparam_IdSet_simul_s20_2000k_kernel=gaussian_scale=1_box=0.1.png) |
| gaussian | 1 | 1 | ![kernel=gaussian,scale=1,box=1](nonparam_IdSet_simul_s20_2000k_kernel=gaussian_scale=1_box=1.png) |
| gaussian | 1 | 10 | ![kernel=gaussian,scale=1,box=10](nonparam_IdSet_simul_s20_2000k_kernel=gaussian_scale=1_box=10.png) |
| linear | auto | 0.1 | ![kernel=linear,scale=auto,box=0.1](nonparam_IdSet_simul_s20_2000k_kernel=linear_scale=_auto__box=0.1.png) |
| linear | auto | 1 | ![kernel=linear,scale=auto,box=1](nonparam_IdSet_simul_s20_2000k_kernel=linear_scale=_auto__box=1.png) |
| linear | auto | 10 | ![kernel=linear,scale=auto,box=10](nonparam_IdSet_simul_s20_2000k_kernel=linear_scale=_auto__box=10.png) |
| linear | 1 | 0.1 | ![kernel=linear,scale=1,box=0.1](nonparam_IdSet_simul_s20_2000k_kernel=linear_scale=1_box=0.1.png) |
| linear | 1 | 1 | ![kernel=linear,scale=1,box=1](nonparam_IdSet_simul_s20_2000k_kernel=linear_scale=1_box=1.png) |
| linear | 1 | 10 | ![kernel=linear,scale=1,box=10](nonparam_IdSet_simul_s20_2000k_kernel=linear_scale=1_box=10.png) |

---

#### Iteration: 4000k

**Full Parameter Grid:**
![Full Grid s=20, 4000k](nonparam_FullGrid_s20_4000k.png)

**How to Read These Figures:**

1. **Full Parameter Grid Figure (above)**: This scatter plot shows all 10,000 evaluated parameter combinations in the (μ, σ) space:
   - **Black dots**: Parameter combinations that do NOT belong to the identified set (rejected by the moment/test conditions)
   - **Cyan dots**: Parameter combinations that DO belong to the identified set (consistent with observed data)
   - **Black marker at (μ≈3, σ≈1)**: The true/reference parameter value used in the simulation
   - This figure helps assess **coverage** - whether the 10k parameter grid adequately explores the space and whether there are sparse regions

2. **SVM IdSet Figures (below)**: Each figure shows the SVM-predicted identified set for a specific hyperparameter configuration:
   - **Cyan shaded region**: The continuous region where the SVM predicts parameter combinations belong to the identified set
   - **Gray dots**: The dense evaluation grid (PGx) showing where we evaluated the SVM
   - Compare across different SVM settings to see which produce more "sensible" (convex, connected) identified sets

**SVM Hyperparameter Comparison:**

| Kernel | Scale | Box Constraint | Figure |
|--------|-------|---------------|--------|
| gaussian | auto | 0.1 | ![kernel=gaussian,scale=auto,box=0.1](nonparam_IdSet_simul_s20_4000k_kernel=gaussian_scale=_auto__box=0.1.png) |
| gaussian | auto | 1 | ![kernel=gaussian,scale=auto,box=1](nonparam_IdSet_simul_s20_4000k_kernel=gaussian_scale=_auto__box=1.png) |
| gaussian | auto | 10 | ![kernel=gaussian,scale=auto,box=10](nonparam_IdSet_simul_s20_4000k_kernel=gaussian_scale=_auto__box=10.png) |
| gaussian | 1 | 0.1 | ![kernel=gaussian,scale=1,box=0.1](nonparam_IdSet_simul_s20_4000k_kernel=gaussian_scale=1_box=0.1.png) |
| gaussian | 1 | 1 | ![kernel=gaussian,scale=1,box=1](nonparam_IdSet_simul_s20_4000k_kernel=gaussian_scale=1_box=1.png) |
| gaussian | 1 | 10 | ![kernel=gaussian,scale=1,box=10](nonparam_IdSet_simul_s20_4000k_kernel=gaussian_scale=1_box=10.png) |
| linear | auto | 0.1 | ![kernel=linear,scale=auto,box=0.1](nonparam_IdSet_simul_s20_4000k_kernel=linear_scale=_auto__box=0.1.png) |
| linear | auto | 1 | ![kernel=linear,scale=auto,box=1](nonparam_IdSet_simul_s20_4000k_kernel=linear_scale=_auto__box=1.png) |
| linear | auto | 10 | ![kernel=linear,scale=auto,box=10](nonparam_IdSet_simul_s20_4000k_kernel=linear_scale=_auto__box=10.png) |
| linear | 1 | 0.1 | ![kernel=linear,scale=1,box=0.1](nonparam_IdSet_simul_s20_4000k_kernel=linear_scale=1_box=0.1.png) |
| linear | 1 | 1 | ![kernel=linear,scale=1,box=1](nonparam_IdSet_simul_s20_4000k_kernel=linear_scale=1_box=1.png) |
| linear | 1 | 10 | ![kernel=linear,scale=1,box=10](nonparam_IdSet_simul_s20_4000k_kernel=linear_scale=1_box=10.png) |

---

## Technical Notes

- **Code location**: `II_MAIN_nonparam_simul.m`
- **Runtime information**:
  - **s=5**: Elapsed time = 02:13:12 (7,992.3 seconds), completed on 2025-10-29
  - **s=10**: Elapsed time = 03:55:03 (14,103.6 seconds), completed on 2025-10-30
  - **s=20**: Elapsed time = 03:44:40 (13,480.3 seconds), completed on 2025-10-30
  - Full runtime log available in `runtime_log.txt`
- **Figure naming convention**: 
  - Full grid: `nonparam_FullGrid_s{S}_{ITER}k.png`
  - SVM results: `nonparam_IdSet_simul_s{S}_{ITER}k_kernel={K}_scale={SCALE}_box={BOX}.png`
- **Evaluation grid**: PGx uses 200,000 Halton sequence points over data-driven bounds (1-99 percentile range with 10% padding)
- **Parameter grid size**: 10,000 evaluated parameter combinations

