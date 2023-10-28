# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd

# +
# Load Sentinel 2 data

#######   On Mac  ########



path_to_data = "/home/amnnrz/GoogleDrive - msaminnorouzi/PhD/Projects/DSFAS/Data/Sentinel2_data/dry_2/"
path_to_data = ("/Users/aminnorouzi/Library/CloudStorage/",
                "GoogleDrive-msaminnorouzi@gmail.com/My Drive/",
                "PhD/Projects/DSFAS/Data/Sentinel2_data/dry_irgted/")

firstimg_2020 = pd.read_csv(path_to_data + "firstImg_2020.csv")
secondimg_2020 = pd.read_csv(path_to_data + "secondImg_2020.csv")
thirdimg_2020 = pd.read_csv(path_to_data + "thirdImg_2020.csv")
firstimg_2021 = pd.read_csv(path_to_data + "firstImg_2021.csv")
secondimg_2021 = pd.read_csv(path_to_data + "secondImg_2021.csv")
thirdimg_2021 = pd.read_csv(path_to_data + "thirdImg_2021.csv")
firstimg_2022 = pd.read_csv(path_to_data + "firstImg_2022.csv")
secondimg_2022 = pd.read_csv(path_to_data + "secondImg_2022.csv")
thirdimg_2022 = pd.read_csv(path_to_data + "thirdImg_2022.csv")

# -

firstimg_2020.DepthSampl.value_counts()


# +
firstimg_2020.rename(
    columns={'NDVI': 'NDVI_first', 'tvi': 'tvi_first', 'savi': 'savi_first',
             'MSI': 'MSI_first', 'GNDVI': 'GNDVI_first', 'GRVI': 'GRVI_first',
             'LSWI': 'LSWI_first', 'MSAVI2': 'MSAVI2_first', 'WDVI': 'WDVI_first',
             'BI': 'BI_first', 'BI2': 'BI2_first', 'RI': 'RI_first', 'CI':'CI_first',
             'v':'v_first', 
             'B1': 'B1_first', 'B2': 'B2_first', 'B3': 'B3_first', 'B4': 'B4_first',
             'B8': 'B8_first', 'B11': 'B11_first', 'B12': 'B12_first'}, inplace=True)
firstimg_2020.columns

firstimg_2021.rename(
    columns={'NDVI': 'NDVI_first', 'tvi': 'tvi_first', 'savi': 'savi_first',
             'MSI': 'MSI_first', 'GNDVI': 'GNDVI_first', 'GRVI': 'GRVI_first',
             'LSWI': 'LSWI_first', 'MSAVI2': 'MSAVI2_first', 'WDVI': 'WDVI_first',
             'BI': 'BI_first', 'BI2': 'BI2_first', 'RI': 'RI_first', 'CI': 'CI_first',
             'v': 'v_first',
             'B1': 'B1_first', 'B2': 'B2_first', 'B3': 'B3_first', 'B4': 'B4_first',
             'B8': 'B8_first', 'B11': 'B11_first', 'B12': 'B12_first'}, inplace=True)

firstimg_2022.rename(
    columns={'NDVI': 'NDVI_first', 'tvi': 'tvi_first', 'savi': 'savi_first',
             'MSI': 'MSI_first', 'GNDVI': 'GNDVI_first', 'GRVI': 'GRVI_first',
             'LSWI': 'LSWI_first', 'MSAVI2': 'MSAVI2_first', 'WDVI': 'WDVI_first',
             'BI': 'BI_first', 'BI2': 'BI2_first', 'RI': 'RI_first', 'CI': 'CI_first',
             'v': 'v_first',
             'B1': 'B1_first', 'B2': 'B2_first', 'B3': 'B3_first', 'B4': 'B4_first',
             'B8': 'B8_first', 'B11': 'B11_first', 'B12': 'B12_first'}, inplace=True)

secondimg_2020.rename(
    columns={'NDVI': 'NDVI_second', 'tvi': 'tvi_second', 'savi': 'savi_second',
             'MSI': 'MSI_second', 'GNDVI': 'GNDVI_second', 'GRVI': 'GRVI_second',
             'LSWI': 'LSWI_second', 'MSAVI2': 'MSAVI2_second', 'WDVI': 'WDVI_second',
             'BI': 'BI_second', 'BI2': 'BI2_second', 'RI': 'RI_second', 'CI': 'CI_second',
             'v': 'v_second',
             'B1': 'B1_second', 'B2': 'B2_second', 'B3': 'B3_second', 'B4': 'B4_second',
             'B8': 'B8_second', 'B11': 'B11_second', 'B12': 'B12_second'}, inplace=True)
secondimg_2020.columns

secondimg_2021.rename(
    columns={'NDVI': 'NDVI_second', 'tvi': 'tvi_second', 'savi': 'savi_second',
             'MSI': 'MSI_second', 'GNDVI': 'GNDVI_second', 'GRVI': 'GRVI_second',
             'LSWI': 'LSWI_second', 'MSAVI2': 'MSAVI2_second', 'WDVI': 'WDVI_second',
             'BI': 'BI_second', 'BI2': 'BI2_second', 'RI': 'RI_second', 'CI': 'CI_second',
             'v': 'v_second',
             'B1': 'B1_second', 'B2': 'B2_second', 'B3': 'B3_second', 'B4': 'B4_second',
             'B8': 'B8_second', 'B11': 'B11_second', 'B12': 'B12_second'}, inplace=True)

secondimg_2022.rename(
    columns={'NDVI': 'NDVI_second', 'tvi': 'tvi_second', 'savi': 'savi_second',
             'MSI': 'MSI_second', 'GNDVI': 'GNDVI_second', 'GRVI': 'GRVI_second',
             'LSWI': 'LSWI_second', 'MSAVI2': 'MSAVI2_second', 'WDVI': 'WDVI_second',
             'BI': 'BI_second', 'BI2': 'BI2_second', 'RI': 'RI_second', 'CI': 'CI_second',
             'v': 'v_second',
             'B1': 'B1_second', 'B2': 'B2_second', 'B3': 'B3_second', 'B4': 'B4_second',
             'B8': 'B8_second', 'B11': 'B11_second', 'B12': 'B12_second'}, inplace=True)

secondimg_2022.columns

thirdimg_2020.rename(
    columns={'NDVI': 'NDVI_third', 'tvi': 'tvi_third', 'savi': 'savi_third',
             'MSI': 'MSI_third', 'GNDVI': 'GNDVI_third', 'GRVI': 'GRVI_third',
             'LSWI': 'LSWI_third', 'MSAVI2': 'MSAVI2_third', 'WDVI': 'WDVI_third',
             'BI': 'BI_third', 'BI2': 'BI2_third', 'RI': 'RI_third', 'CI': 'CI_third',
             'v': 'v_third',
             'B1': 'B1_third', 'B2': 'B2_third', 'B3': 'B3_third', 'B4': 'B4_third',
             'B8': 'B8_third', 'B11': 'B11_third', 'B12': 'B12_third'}, inplace=True)
thirdimg_2020.columns

thirdimg_2021.rename(
    columns={'NDVI': 'NDVI_third', 'tvi': 'tvi_third', 'savi': 'savi_third',
             'MSI': 'MSI_third', 'GNDVI': 'GNDVI_third', 'GRVI': 'GRVI_third',
             'LSWI': 'LSWI_third', 'MSAVI2': 'MSAVI2_third', 'WDVI': 'WDVI_third',
             'BI': 'BI_third', 'BI2': 'BI2_third', 'RI': 'RI_third', 'CI': 'CI_third',
             'v': 'v_third',
             'B1': 'B1_third', 'B2': 'B2_third', 'B3': 'B3_third', 'B4': 'B4_third',
             'B8': 'B8_third', 'B11': 'B11_third', 'B12': 'B12_third'}, inplace=True)

thirdimg_2022.rename(
    columns={'NDVI': 'NDVI_third', 'tvi': 'tvi_third', 'savi': 'savi_third',
             'MSI': 'MSI_third', 'GNDVI': 'GNDVI_third', 'GRVI': 'GRVI_third',
             'LSWI': 'LSWI_third', 'MSAVI2': 'MSAVI2_third', 'WDVI': 'WDVI_third',
             'BI': 'BI_third', 'BI2': 'BI2_third', 'RI': 'RI_third', 'CI': 'CI_third',
             'v': 'v_third',
             'B1': 'B1_third', 'B2': 'B2_third', 'B3': 'B3_third', 'B4': 'B4_third',
             'B8': 'B8_third', 'B11': 'B11_third', 'B12': 'B12_third'}, inplace=True)

thirdimg_2022.columns


# +
firstimg_2020 = firstimg_2020.loc[firstimg_2020['YearSample'] == 2020].copy()
secondimg_2020 = secondimg_2020.loc[secondimg_2020['YearSample'] == 2020].copy()
thirdimg_2020 = thirdimg_2020.loc[thirdimg_2020['YearSample'] == 2020].copy()
firstimg_2020.dropna(subset="B1_first", inplace=True)
secondimg_2020.dropna(subset="B1_second", inplace=True)
thirdimg_2020.dropna(subset="B1_third", inplace=True)

firstimg_2021 = firstimg_2021.loc[firstimg_2021['YearSample'] == 2021].copy()
secondimg_2021 = secondimg_2021.loc[secondimg_2021['YearSample'] == 2021].copy()
thirdimg_2021 = thirdimg_2021.loc[thirdimg_2021['YearSample'] == 2021].copy()
firstimg_2021.dropna(subset="B1_first", inplace=True)
secondimg_2021.dropna(subset="B1_second", inplace=True)
thirdimg_2021.dropna(subset="B1_third", inplace=True)

firstimg_2022 = firstimg_2022.loc[firstimg_2022['YearSample'] == 2022].copy()
secondimg_2022 = secondimg_2022.loc[secondimg_2022['YearSample'] == 2022].copy()
thirdimg_2022 = thirdimg_2022.loc[thirdimg_2022['YearSample'] == 2022].copy()
firstimg_2022.dropna(subset="B1_first", inplace=True)
secondimg_2022.dropna(subset="B1_second", inplace=True)
thirdimg_2022.dropna(subset="B1_third", inplace=True)


# +
df_first = pd.concat([firstimg_2020, firstimg_2021, firstimg_2022])
df_second = pd.concat([secondimg_2020, secondimg_2021, secondimg_2022])
df_third = pd.concat([thirdimg_2020, thirdimg_2021, thirdimg_2022])

df_first.reset_index(inplace=True)
df_second.reset_index(inplace=True)
df_third.reset_index(inplace=True)
# -

df_second.iloc[:, 11:]

df_first.shape, df_second.shape, df_third.shape

print(df_first.shape, df_second.shape, df_third.shape)
n_row = min(df_first.shape[0], df_second.shape[0], df_third.shape[0])
df = pd.concat([df_first, df_second.iloc[:, 11:], df_third.iloc[:, 11:]], axis=1)
df = df.iloc[:n_row, :]
df

# Save final df
df.to_csv(path_to_data + "Carbon&satellite_data_dry_joined_v1.csv")
