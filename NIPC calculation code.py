import numpy as np
import pandas as pd
import numpy.polynomial.legendre as leg
from itertools import combinations_with_replacement
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
INPUT_FILE_PARAMS = "SSP245_Parameters.xlsx" 
INPUT_FILE_RESULTS = "SSP245_Simulation_Results.xlsx"
OUTPUT_FILE = "NIPC_Analysis_Output.xlsx"

MAX_DEGREE = 3          

# ==================== Core Functions ====================

def generate_multi_indices(n_vars, max_degree):
    
    indices = [np.zeros(n_vars, dtype=int)]
    for d in range(1, max_degree + 1):
        for combo in combinations_with_replacement(range(n_vars), d):
            idx = np.zeros(n_vars, dtype=int)
            for v in combo:
                idx[v] += 1
            indices.append(idx)
    return np.array(indices)

def solve_nipc_qr(X, Y, max_degree, verbose=False):
    
    n_samples, n_vars = X.shape
    
    if verbose:
        print(f"    Input: n_samples={n_samples}, n_vars={n_vars}, max_degree={max_degree}")
    
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    range_x = x_max - x_min
    range_x[range_x == 0] = 1  
    X_norm = 2 * (X - x_min) / range_x - 1
    
    indices = generate_multi_indices(n_vars, max_degree)
    n_terms = len(indices)
    
    if verbose:
        print(f"    Generated {n_terms} polynomial terms for PC expansion")
    
    if n_samples < n_terms:
        print(f"     SKIP: Need at least {n_terms} samples, but only have {n_samples}")
        return None, None, None
    
    if verbose:
        print(f"    Sample check: OK (n_samples={n_samples} ≥ n_terms={n_terms})")
    
    
    Psi = np.ones((n_samples, n_terms))
    
    for j in range(1, n_terms):
       
         for v in range(n_vars):
            k = indices[j, v]
            if k > 0:
                
                legendre_val = leg.legval(X_norm[:, v], [0]*k + [1])
                Psi[:, j] *= legendre_val
    
    if verbose:
        print(f"    Built design matrix Psi of shape {Psi.shape}")
    
    Q, R = np.linalg.qr(Psi)
    
    if verbose:
        print(f"    Performed QR decomposition")
        print(f"    Q shape: {Q.shape}, R shape: {R.shape}")
       
        ortho_check = np.max(np.abs(Q.T @ Q - np.eye(Q.shape[1])))
        print(f"    Orthonormality check: max|Q^T Q - I| = {ortho_check:.2e}")
    
    
    coeffs = np.linalg.solve(R, Q.T @ Y)
    
    if verbose:
        print(f"    Solved least squares problem for {len(coeffs)} coefficients")
    
    Y_pred = Psi @ coeffs
    ss_res = np.sum((Y - Y_pred)**2)
    ss_tot = np.sum((Y - np.mean(Y))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    if verbose:
        print(f"    R² = {r2:.6f}")
    
    mean_val = coeffs[0]
    variance_val = np.sum(coeffs[1:]**2)
    
    if verbose:
        print(f"    Mean (c0) = {mean_val:.6f}")
        print(f"    Variance (Σ ci² for i≥1) = {variance_val:.6f}")
        print(f"    Total explained variance (c0² + variance) = {coeffs[0]**2 + variance_val:.6f}")
    
    return mean_val, variance_val, r2

# ==================== Main Process ====================

def main():
    
    print("=" * 80)
    print("NIPC Analysis using QR Decomposition (Orthonormal Basis)")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Input parameters file: {INPUT_FILE_PARAMS}")
    print(f"  Input results file: {INPUT_FILE_RESULTS}")
    print(f"  Output file: {OUTPUT_FILE}")
    print(f"  Maximum polynomial degree: {MAX_DEGREE}")
    print("=" * 80)
    
    # ======== Load Data ========
    try:
        df_p = pd.read_excel(INPUT_FILE_PARAMS)
        df_r = pd.read_excel(INPUT_FILE_RESULTS)
        print(f"  Successfully loaded input files")
        print(f"  Parameters file: {df_p.shape[0]} rows × {df_p.shape[1]} columns")
        print(f"  Results file: {df_r.shape[0]} rows × {df_r.shape[1]} columns")
    except Exception as e:
        print(f"  Error loading files: {e}")
        return

    
    param_cols = df_p.columns[2:].tolist()
    
    if not param_cols:
        print(" Error: No parameter columns found (need columns after City and Year)")
        return
    
    print(f"  Found {len(param_cols)} parameter columns:")
    print(f"  {param_cols}")
    
    
    indices = generate_multi_indices(len(param_cols), MAX_DEGREE)
    n_terms = len(indices)
    print(f"  Basis configuration: {n_terms} terms (degree {MAX_DEGREE})")
    print("=" * 80)
    
    cities = sorted(df_p['City'].unique())
    years = sorted(df_p['Year'].unique())
    
    print(f"Processing {len(cities)} cities × {len(years)} years = {len(cities)*len(years)} combinations\n")
    
    results = []
    processed_count = 0
    skipped_count = 0
    
   
    for city in cities:
        print(f"City: {city}")
        for year in years:
           
            mask_p = (df_p['City'] == city) & (df_p['Year'] == year)
            mask_r = (df_r['City'] == city) & (df_r['Year'] == year)
            
            X = df_p[mask_p][param_cols].values
            Y = df_r[mask_r]['Yield'].values
            
           
            if len(X) == 0 or len(Y) == 0:
                print(f"  Year {year}:   No data available")
                skipped_count += 1
                continue
            
           
            if len(X) != len(Y):
                print(f"  Year {year}:   X and Y size mismatch ({len(X)} vs {len(Y)})")
                skipped_count += 1
                continue
            
           
            mean_y, var_y, r2 = solve_nipc_qr(X, Y, MAX_DEGREE, verbose=False)
            
            
            if mean_y is None:
                skipped_count += 1
                continue
            
           
            if mean_y != 0:
                cv = np.sqrt(max(0, var_y)) / abs(mean_y)
            else:
                cv = np.inf
            
            print(f"  Year {year}:   Mean={mean_y:.4f}, Var={var_y:.6f}, CV={cv:.4f}, R²={r2:.6f}")
            
            results.append({
                'City': city,
                'Year': year,
                'Mean_Yield': mean_y,
                'Variance': var_y,
                'Stability_CV': cv,
                'R2': r2,
                'Sample_Count': len(X)
            })
            processed_count += 1
    
   
    print("\n" + "=" * 80)
    
    if results:
        output_df = pd.DataFrame(results)
        output_df.to_excel(OUTPUT_FILE, index=False)
        
        print(f"  Analysis complete!")
        print(f"  Processed: {processed_count} city-year combinations")
        print(f"  Skipped: {skipped_count} city-year combinations")
        print(f"  Results saved to: {OUTPUT_FILE}")
        print(f"\nOutput file contains columns:")
        print(f"  - City: City name")
        print(f"  - Year: Year")
        print(f"  - Mean_Yield: Mean value (constant term c0)")
        print(f"  - Variance: Variance contribution from higher-order terms (Σ ci²)")
        print(f"  - Stability_CV: Coefficient of variation (√Variance / Mean_Yield)")
        print(f"  - R2: R² value (goodness of fit)")
        print(f"  - Sample_Count: Number of samples used")
    else:
        print(f"  Error: No results generated! Check input data.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
