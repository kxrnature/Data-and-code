import numpy as np
import pandas as pd
import numpy.polynomial.legendre as leg


file_path1 = r"C:\Users\Desktop\SSP245.xlsx"
file_path2 = r"C:\Users\Desktop\NIPC-simulation_results_SSP_all_cities245.xlsx"


df_x = pd.read_excel(file_path1)
df_M = pd.read_excel(file_path2)


cities_x = df_x.iloc[:, -1].unique()
cities_M = df_M.iloc[:, -1].unique()


common_cities = np.intersect1d(cities_x, cities_M)


k_values = []


for city in common_cities:
   
    df_x_city = df_x[df_x.iloc[:, -1] == city]
    df_M_city = df_M[df_M.iloc[:, -1] == city]

    X = df_x_city.iloc[:, :-1].values

    X_scaled = np.zeros_like(X)
    for i in range(X.shape[1]):
        X_min = X[:, i].min()
        X_max = X[:, i].max()
        X_scaled[:, i] = 2 * (X[:, i] - X_min) / (X_max - X_min) - 1

    for year in range(2025, 2101):
        df_year = df_M_city[df_M_city['Year'] == year]

        if not df_year.empty:
            M = df_year.iloc[:, -2].values 

           
            P = np.zeros((len(M), X.shape[1]))
            for i in range(X.shape[1]):
                P[:, i] = leg.legval(X_scaled[:, i], [0] * i + [1])

           
            k, _, _, _ = np.linalg.lstsq(P, M, rcond=None)

           
            k_values.append([city, year] + list(k))
        else:
            print(f"nothing")


k_df = pd.DataFrame(k_values, columns=['City'] + ['Year'] + [f'k{i+1}' for i in range(X.shape[1])])

k_df.to_excel(r"C:\Users\Desktop\NIPC_K_245.xlsx", index=False)
