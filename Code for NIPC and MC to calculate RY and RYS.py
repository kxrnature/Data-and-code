import pandas as pd
import numpy as np

file_path1 = r"C:\Users\Desktop\NIPC-simulation_results_SSP_all_cities585.xlsx"
file_path2 = r"C:\Users\Desktop\MC-simulation_results_SSP_all_cities585.xlsx"
file_path3 = r"C:\Users\Desktop\NIPC_K_585.xlsx"
file_path4 = r"C:\Users\Desktop\mean_MC585.xlsx"


df1 = pd.read_excel(file_path1)
df2 = pd.read_excel(file_path2)
df3 = pd.read_excel(file_path3)
df4 = pd.read_excel(file_path4)

def calculate_variance(data):
    return np.var(data, ddof=0)

def calculate_u_and_lamda(x, y, x_k, y_k):
    var_x = calculate_variance(x)
    var_y = calculate_variance(y)
    cov_xy = np.cov(x, y, ddof=0)[0, 1]
    denominator = var_x + var_y - 2 * cov_xy
    if denominator == 0:
        return np.nan, np.nan  
    lamda = (var_y - cov_xy) / denominator
    lamda2 = (var_x - cov_xy) / denominator
    u = lamda * x_k + lamda2 * y_k
    return u, lamda


results = []


for city in df1['City'].unique():
    for year in df1['Year'].unique():
        
        df1_city_year = df1[(df1['City'] == city) & (df1['Year'] == year)]
        df2_city_year = df2[(df2['City'] == city) & (df2['Year'] == year)]
        df3_city_year = df3[(df3['City'] == city) & (df3['Year'] == year)]
        df4_city_year = df4[(df4['City'] == city) & (df4['Year'] == year)]

        x = df1_city_year.iloc[:, 2].values
        y = df2_city_year.iloc[:, 2].values
        x_k = df3_city_year.iloc[:, 2].values
        y_k = df4_city_year.iloc[:, 2].values

        u, lamda = calculate_u_and_lamda(x, y, x_k, y_k)

        results.append((city, year, u, lamda))

results_df = pd.DataFrame(results, columns=['City', 'Year', 'Final Output', 'Lamda'])

output_path = r"C:\Users\Desktop\finial_585.xlsx"
results_df.to_excel(output_path, index=False)
#Please note, lambda is the w in the method, and u is the U in the method.
