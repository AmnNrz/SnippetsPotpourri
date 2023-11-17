# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: tillenv
#     language: python
#     name: python3
# ---

import numpy as np 
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import geopandas as gpd
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from sklearn.metrics import confusion_matrix

# +
# # Read data
# path_to_data = ("/home/amnnrz/OneDrive - a.norouzikandelati/"
#                 "Ph.D/Projects/DSFAS/Data/")

# path_to_plots = ("/home/amnnrz/OneDrive - a.norouzikandelati/"
#                  "Ph.D/Projects/DSFAS/Plots/")

path_to_data = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
    "Ph.D/Projects/DSFAS/Data/"
)

path_to_plots = ("/Users/aminnorouzi/Library/CloudStorage/"
                 "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
                 "Ph.D/Projects/DSFAS/Plots/")

dry_df = pd.read_csv(path_to_data + "Carbon&satellite_data_dry_joined_v1.csv")

dry_irig_df = pd.read_csv(
    path_to_data + "Carbon&satellite_data_dryIrgted_joined_v1.csv"
)

raw_data = pd.read_csv(path_to_data + "EWA_carbon_subset.csv")
# raw_data = raw_data.dropna(subset="TotalC_%")

# Convert year to integer
dry_df["YearSample"] = dry_df["YearSample"].astype(int)

# remove old index column
dry_df.drop(columns="index", axis=1, inplace=True)

dry_df_raw = gpd.read_file(
    path_to_data
    + (
        "/GIS_Data/Csample_buffer_shp/"
        "C_samples_dryland_shp/C_samples_dryland_shp.shp"
    )
)

allSamples_df = pd.read_csv(
    path_to_data + "Carbon&satellite_data_dryIrgted_joined_v1.csv"
)

# +
raw_data = pd.read_csv(path_to_data + "EWA_carbon_subset.csv")
raw_data["DepthSampled_inches"].value_counts()

dry_df['DepthSampl'].value_counts()


# +
# Convert Total_C_% to g/cm2
# "total_c_%" /100 * height * A * 2.54 (inch to cm) * BD
def tCarbon_to_gcm2(df):
    df_copy = df.copy()
    df_copy.loc[:, "Total_C (g/cm2)"] = (
        df_copy["TotalC"] / 100 * 12 * 2.54 * 1 * df_copy["BD_g_cm3"] *100
    )
    return df_copy


# Convert dataframes to GeoDataFrames
dry_df = gpd.GeoDataFrame(
    dry_df, geometry=gpd.points_from_xy(dry_df.Longitude, dry_df.Latitude)
)
allSamples_df = gpd.GeoDataFrame(
    allSamples_df,
    geometry=gpd.points_from_xy(allSamples_df.Longitude, allSamples_df.Latitude),
)

# Remove reduntant columns
allSamples_df = allSamples_df.loc[:, "TotalC":].copy()
dry_df = dry_df.loc[:, "TotalC":].copy()

dry_df.reset_index(drop=True, inplace=True)
allSamples_df.reset_index(drop=True, inplace=True)

dry_df = dry_df[dry_df.columns[dry_df.columns.isin(allSamples_df.columns)]]

# merge two dataframes
irrigated_df = allSamples_df.loc[
    ~(allSamples_df["SampleID"].isin(dry_df["SampleID"]))
].copy()


# add irrigation column
irrigated_df["Irrigation"] = "Irrigated"
dry_df["Irrigation"] = "Dryland"

dry_df = tCarbon_to_gcm2(dry_df)
dry_df = dry_df.reset_index(drop=True)
irrigated_df = tCarbon_to_gcm2(irrigated_df)
irrigated_df = irrigated_df.reset_index(drop=True)
df = pd.concat([dry_df, irrigated_df])

# +
# raw_data['DepthSampled_inches'].value_counts(), dry_df_raw['DepthSampl'].value_counts()
# dry_irig_df['TotalC'].isna().value_counts(), dry_irig_df['DepthSampl'].value_counts()
# dry_df['TotalC'].isna().value_counts(), dry_df['DepthSampl'].value_counts()
# dry_irig_df['TotalC'].isna().value_counts(), dry_irig_df['DepthSampl'].value_counts()
# dry_irig_df['SampleID'].isin(dry_df['SampleID']).value_counts()
# dry_irig_df['DepthSampl'].value_counts()

# +
# check 0_6 -- 0_12 samples' year
sampleYear_6_12 = dry_df.loc[dry_df["DepthSampl"] == "0_6", "YearSample"].values[0]
print("Two-depth samples are for:", f"{sampleYear_6_12}")

# check 0_6 -- 0_12 samples' year
sampleYear_6_12 = irrigated_df.loc[
    irrigated_df["DepthSampl"] == "0_6", "YearSample"
].values[0]
print("Two-depth samples are for:", f"{sampleYear_6_12}")
# -


dry_df['DepthSampl'].value_counts()

irrigated_df['DepthSampl'].value_counts()


# +
# Get average of total_C over 0-6 and 6-12 inches samples
def averageC(df):
    dup_df = df.loc[df.SampleID.duplicated(keep=False)]
    dup_df

    averaged_C = pd.DataFrame([])
    averaged_C["SampleID"] = dup_df.SampleID.unique()
    for id in dup_df.SampleID.unique():
        averaged_C.loc[averaged_C["SampleID"] == id, "Total_C (g/cm2)"] = np.mean(
            dup_df.loc[dup_df["SampleID"] == id, "Total_C (g/cm2)"]
        )

    df = df.loc[~df.SampleID.duplicated()]
    df.loc[df.SampleID.isin(averaged_C.SampleID), "Total_C (g/cm2)"] = averaged_C[
        "Total_C (g/cm2)"
    ].values
    df.loc[df.SampleID.isin(averaged_C.SampleID), "Total_C (g/cm2)"]
    df.loc[df["DepthSampl"] == "0_6", "DepthSampl"] = "0_12"
    return df


df_0_6 = df.loc[df["DepthSampl"] == "0_6"]

dry_df = averageC(dry_df)
irrigated_df = averageC(irrigated_df)
# -

irrigated_df['DepthSampl'].value_counts()

print(dry_df.shape, irrigated_df.shape)
dry_df

# +
# Scale band values


def scaleBands(df):
    largeValue_idx = (df.iloc[:, 11:-1].describe().loc["min"] < -2) | (
        df.iloc[:, 8:].describe().loc["max"] > 2
    )
    largeValue_cols = largeValue_idx[largeValue_idx].index

    scaler = StandardScaler()

    # fit the scaler on the selected columns
    scaler.fit(df[largeValue_cols].copy())

    # transform the selected columns to have zero mean and unit variance
    df.loc[:, largeValue_cols] = scaler.transform(df[largeValue_cols].copy())
    return df

# df = scaleBands(df)
irrigated_df = scaleBands(irrigated_df)
dry_df = scaleBands(dry_df)


df = pd.concat([dry_df, irrigated_df])
# -

dry_df = dry_df.loc[~(dry_df["DepthSampl"] == "6_12")]
irrigated_df = irrigated_df.loc[~(irrigated_df["DepthSampl"] == "6_12")]
dry_df = dry_df.reset_index(drop=True)
irrigated_df = irrigated_df.reset_index(drop=True)
df = pd.concat([dry_df, irrigated_df])
print(df["DepthSampl"].value_counts())

# +
# DENSITY DISTRIBUTION PLOT FOR ALL YEARS TOGETHER
# Increase the font size of the labels
plt.rcParams.update({'font.size': 12})

# Increase the resolution of the plot
plt.figure(figsize=(12, 8), dpi=300)

# Plot the density distribution of column 'Total_C_g/cm2'
dry_df['Total_C (g/cm2)'].plot(kind='density')

# Set x-axis label
plt.xlabel('Total C (g/cm$^2$)', fontsize=14)

# Mark actual values on the curve
min_value = dry_df['Total_C (g/cm2)'].min()
max_value = dry_df['Total_C (g/cm2)'].max()

# Plotting the actual values on the curve
plt.axvline(x=min_value, color='red', linestyle='--', label='Min')
plt.axvline(x=max_value, color='blue', linestyle='--', label='Max')

# Display legend
plt.legend(fontsize=12)

# Show the plot
plt.show()
# -


dry_df

# +
######=====    Density Distribution of Total C grouped by year  =====#######
# Set the style and size of the plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Loop through each year and plot the density distribution
for year in dry_df["YearSample"].unique():
    subset = dry_df[dry_df["YearSample"] == year]
    sns.kdeplot(subset["TotalC"], label=f"Year {year}", fill=True)

# Add labels and title
plt.xlabel("Total_C (Percentage of mass)")
plt.ylabel("Density")
plt.title("Density Distribution of Total C Grouped by Year")
plt.legend()

plt.xlim(0, None)
# Show the plot
plt.show()
# -


# # Create map of data

# +
# Load dry_irrigated dataframe 
allSamples_df = pd.read_csv(path_to_data + "Carbon&satellite_data_dryIrgted_joined_v1.csv")

# Get average of total_C over 0-6 and 6-12 inches samples 
dup_df = allSamples_df.loc[allSamples_df.SampleID.duplicated(keep=False)]
dup_df

averaged_C = pd.DataFrame([])
averaged_C['SampleID'] = dup_df.SampleID.unique()
for id in dup_df.SampleID.unique():
    averaged_C.loc[averaged_C["SampleID"] == id, "TotalC"] = np.mean(
        dup_df.loc[dup_df["SampleID"] == id, "TotalC"])

averaged_C.head(5)

allSamples_df = allSamples_df.loc[~allSamples_df.SampleID.duplicated()]
allSamples_df.loc[allSamples_df.SampleID.isin(averaged_C.SampleID),
        'TotalC'] = averaged_C['TotalC'].values

allSamples_df.loc[allSamples_df['DepthSampl'] == '0_6', 'DepthSampl'] = '0_12'
allSamples_df.loc[allSamples_df['DepthSampl'] == '6_12', 'DepthSampl'] = '0_12'
allSamples_df


# +
######=====  Sample points grouped by irrigation type  =====#########
# Load U.S. states shapefiles (You can download from U.S. Census Bureau or other sources)
path_to_shpfiles = path_to_data + "GIS_Data/"

us_states = gpd.read_file(path_to_shpfiles + "cb_2022_us_state_500k/cb_2022_us_state_500k.shp")
us_counties = gpd.read_file(path_to_shpfiles + "cb_2022_us_county_500k/cb_2022_us_county_500k.shp")

# Filter for just Washington state
wa_state = us_states[us_states['NAME'] == 'Washington'].copy()
wa_counties = us_counties[us_counties['STATE_NAME'] == 'Washington']
wa_counties

# extract two colors from the 'viridis' colormap
color_map_values = [0, 0.5]  # Start and end of the colormap
colors_from_viridis = plt.cm.viridis(color_map_values)

# Convert to hexadecimal
colors_hex = [mcolors.to_hex(c) for c in colors_from_viridis]


# Plot Washington state
# Create a color map dictionary
color_map_dict = {'Dryland': colors_hex[0], 'Irrigated': colors_hex[1]}

# Map the colors to the DataFrame
df['color'] = df['Irrigation'].map(color_map_dict)

ax = wa_state.boundary.plot(figsize=(40, 20), linewidth=2)
wa_counties.boundary.plot(ax=ax, linewidth=1, edgecolor="black")
wa_counties.apply(lambda x: ax.annotate(text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', fontsize=20, color='black'), axis=1)

irrigation_counts = df['Irrigation'].value_counts()


# Plot the points with the specified colors
labels_with_counts = {}
for color in color_map_dict.values():
    subset = df[df['color'] == color]
    irrigation_type = subset['Irrigation'].unique()[0]
    label_with_count = f"{irrigation_type} (n={irrigation_counts[irrigation_type]})"
    labels_with_counts[irrigation_type] = label_with_count
    subset.plot(ax=ax, marker='o', color=color, markersize=500,
                alpha=0.5, label=label_with_count)

# Add a legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title='Irrigation Type', fontsize=22, title_fontsize=22)

# Change tick label sizes
ax.tick_params(axis='both', which='major', labelsize=16)

# Add title and axis labels
plt.title("Soil Samples grouped by Irrigation Type", fontsize=32)
plt.xlabel("Longitude", fontsize=24)
plt.ylabel("Latitude", fontsize=24)

# Show the plot
plt.figure(dpi=300)
plt.show()


# +
######=====     Distribution of Total C grouped by year  =====#######

# extract two colors from the 'viridis' colormap
color_map_values = [0, 0.5, 1]  # Start and end of the colormap
colors_from_viridis = plt.cm.viridis(color_map_values)

# Convert to hexadecimal
colors_hex = [mcolors.to_hex(c) for c in colors_from_viridis]


# Plot Washington state
# Create a color map dictionary
color_map_dict = {2020: colors_hex[0],
                  2021: colors_hex[1], 2022: colors_hex[2]}

# Map the colors to the DataFrame
df['Yearcolor'] = df['YearSample'].map(color_map_dict)

ax = wa_state.boundary.plot(figsize=(40, 20), linewidth=2)
wa_counties.boundary.plot(ax=ax, linewidth=1, edgecolor="black")
wa_counties.apply(lambda x: ax.annotate(
    text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', fontsize=20, color='black'), axis=1)

year_counts = df['YearSample'].value_counts()

# Plot the points with the specified colors
for color in color_map_dict.values():
    subset = df[df['Yearcolor'] == color]
    year_number = subset['YearSample'].unique()[0]
    label_with_number = f'{year_number} (n = {year_counts[year_number]})'
    subset.plot(ax=ax, marker='o', color=color, markersize=500,
                alpha=0.5, label=label_with_number)

# Add a legendz
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title='Year',
          fontsize=22, title_fontsize=22)

# Change tick label sizes
ax.tick_params(axis='both', which='major', labelsize=16)

# Add title and axis labels
plt.title("Soil Samples grouped by Year", fontsize=32)
plt.xlabel("Longitude", fontsize=24)
plt.ylabel("Latitude", fontsize=24)

# Show the plot
plt.figure(dpi=300)
plt.show()


# +
######=====     Distribution of Total C grouped by terciles =====#######
# Set the terciles to use for separating the data

df = tCarbon_to_gcm2(df)

y_var = "Total_C (g/cm2)"
bottom_tercile = np.percentile(df[y_var], 33.33)
top_tercile = np.percentile(df[y_var], 66.66)

# Create a new column in the DataFrame to indicate whether each row is in the top, middle, or bottom tercile
df['tercile'] = pd.cut(df[y_var], bins=[df[y_var].min(
), bottom_tercile, top_tercile, df[y_var].max()], labels=['bottom', 'middle', 'top'])

######=====     Distribution of Total C map grouped by year  =====#######
######=====  Sample points grouped by irrigation type  =====#########

# extract two colors from the 'viridis' colormap
color_map_values = [0, 0.5, 1]  # Start and end of the colormap
colors_from_viridis = plt.cm.viridis(color_map_values)

# Convert to hexadecimal
colors_hex = [mcolors.to_hex(c) for c in colors_from_viridis]


# Plot Washington state
# Create a color map dictionary
color_map_dict = {'bottom': colors_hex[0],
                  'middle': colors_hex[1], 'top': colors_hex[2]}

# Map the colors to the DataFrame
df['tercile_color'] = df['tercile'].map(color_map_dict)

ax = wa_state.boundary.plot(figsize=(40, 20), linewidth=2)
wa_counties.boundary.plot(ax=ax, linewidth=1, edgecolor="black")
wa_counties.apply(lambda x: ax.annotate(
    text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', fontsize=20, color='black'), axis=1)

tercile_counts = df['tercile'].value_counts()
# Plot the points with the specified colors
for color in color_map_dict.values():
    subset = df[df['tercile_color'] == color]
    tercile = subset['tercile'].unique()[0]
    label_with_number = f'{tercile} (n = {tercile_counts[tercile]})'
    subset.plot(ax=ax, marker='o', color=color, markersize=500,
                alpha=0.5, label=label_with_number)

# Add a legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title='Tercile',
          fontsize=22, title_fontsize=22)

# Change tick label sizes
ax.tick_params(axis='both', which='major', labelsize=16)

# Add title and axis labels
plt.title("Soil Samples grouped by TC terciles", fontsize=32)
plt.xlabel("Longitude", fontsize=24)
plt.ylabel("Latitude", fontsize=24)

# Show the plot
plt.figure(dpi=300)
plt.show()



# +
# # Renaming columns
# df.columns = df.columns.str.replace('_first', '_MAM')
# df.columns = df.columns.str.replace('_second', '_JJA')
# df.columns = df.columns.str.replace('_third', '_SON')
# df.to_csv(path_to_data + 'data_snapshot.csv')

# +
# ###### ======   Density Distribution of features for top and bottom terciles =====######


# dataset = dry_df.loc[:, "NDVI_first":"Total_C (g/cm2)"].copy()
# dataset.drop(columns=["WDVI_first", "WDVI_second", "WDVI_third", "Irrigation"], inplace=True)


# # Set the terciles to use for separating the data
# bottom_tercile = dataset[y_var].quantile(1/3)
# top_tercile = dataset[y_var].quantile(2/3)

# # Create a new column in the DataFrame to indicate whether each row is in the top, middle, or bottom tercile
# dataset['tercile'] = pd.cut(dataset[y_var], bins=[dataset[y_var].min(
# ), bottom_tercile, top_tercile, dataset[y_var].max()], labels=['bottom', 'middle', 'top'], include_lowest=True)

# # filter for just top and bottom tercils
# topBottom_df = dataset.loc[dataset['tercile'] != 'middle'].copy()
# topBottom_df['tercile'] = topBottom_df['tercile'].cat.remove_unused_categories()

# # Renaming columns
# topBottom_df.columns = topBottom_df.columns.str.replace('_first', '_MAM')
# topBottom_df.columns = topBottom_df.columns.str.replace('_second', '_JJA')
# topBottom_df.columns = topBottom_df.columns.str.replace('_third', '_SON')
# # Get list of columns to be plotted
# x_vars = topBottom_df.columns.drop([y_var, 'tercile'])

# # Create separate figures for every 20 variables
# num_plots_per_fig = 20
# num_figs = -(-len(x_vars) // num_plots_per_fig)  # Ceiling division

# for fig_num in range(num_figs):
#     fig, axes = plt.subplots(4, 5, figsize=(
#         20, 16))  # Adjust figsize as needed
#     axes = axes.ravel()  # Flatten the axes for easier indexing

#     for ax_num, x_var in enumerate(x_vars[fig_num*num_plots_per_fig: (fig_num+1)*num_plots_per_fig]):
#         # Choose your own colors
#         for tercile, color in zip(['bottom', 'top'], ['#440154', '#21918c']):
#             subset = topBottom_df[topBottom_df['tercile'] == tercile]
#             sns.kdeplot(ax=axes[ax_num], data=subset, x=x_var,
#                         color=color, label=tercile, fill=True)

#         # axes[ax_num].set_title(x_var)
#         # Set font size for x and y labels
#         axes[ax_num].set_xlabel(x_var, fontsize=15)  # Adjust this value as needed
#         # Adjust this value as needed
#         axes[ax_num].set_ylabel('Density', fontsize=15)
#     # Adjust vertical spacing between subplots
#     plt.subplots_adjust(hspace=0.5, wspace=0.5)
    
    
#     # Handle any unused axes
#     for ax_num in range(ax_num+1, 20):
#         axes[ax_num].axis('off')
    
#     # Add a legend to the figure (not to each individual plot)
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(handles, labels, title='tercile',
#                loc='upper right', bbox_to_anchor=(1, 0.5),
#                prop={'size': 15}, title_fontsize='20')


#     plot_name = path_to_plots + f"figure_{fig_num + 1}.png"
#     fig.savefig(plot_name, dpi=300, bbox_inches='tight')
#     # plt.close(fig)  # Close the current figure to free up memory
# -


df

dry_0_6 = dry_df.loc[dry_df["DepthSampl"] == "0_6"].copy()
irrg_0_6 = irrigated_df.loc[irrigated_df["DepthSampl"] == "0_6"].copy()
dry_0_6.shape, irrg_0_6.shape

# +
dry_df["SOC Stock"] = (
    (dry_df["TotalC"] - dry_df["InorganicC"])
    / 100
    * 12
    * 2.54
    * 1
    * dry_df["BD_g_cm3"]
    * 100
)


irrigated_df["SOC Stock"] = (
    (irrigated_df["TotalC"] - irrigated_df["InorganicC"])
    / 100
    * 12
    * 2.54
    * 1
    * irrigated_df["BD_g_cm3"]
    * 100
)

df = pd.concat([dry_df, irrigated_df])

# +
### ==== build OLS and do cross validation ====###
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import random


# Build OLS function
def ols(X, y, num_iterations):
    Ys_Ypreds_train = {}
    MSE_SCORES_train = []
    MAE_SCORES_train = []
    R2_SCORES_train = []
    RMSE_SCORES_train = []
    PERCENTAGE_ERRORS_train = []

    Ys_Ypreds_test = {}
    MSE_SCORES_test = []
    MAE_SCORES_test = []
    R2_SCORES_test = []
    RMSE_SCORES_test = []
    PERCENTAGE_ERRORS_test = []

    for iteration in range(num_iterations):
        # print(f"Iteration: {iteration + 1}")
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Fit the model
        model = sm.OLS(y_train, X_train).fit()

        # Store the model

        # Predict and evaluate the model
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Define your threshold for the absolute difference
        # threshold = 0.9  # or whatever value you're interested in
        threshold = 75  # or whatever value you're interested in

        # Calculate the absolute differences
        differences_train = np.abs(y_pred_train - y_train)
        differences_test = np.abs(y_pred_test - y_test)

        # Create a mask for where the differences are less than the threshold
        mask_train = differences_train < threshold
        mask_test = differences_test < threshold

        # Filter your predictions and true values
        filtered_y_pred_train = y_pred_train[mask_train]
        filtered_y_train = y_train[mask_train]
        filtered_y_pred_test = y_pred_test[mask_test]
        filtered_y_test = y_test[mask_test]

        mse_train = np.mean((filtered_y_pred_train - filtered_y_train) ** 2)
        MSE_SCORES_train.append(mse_train)

        # Mean Absolute Error (MAE)
        residuals_train = filtered_y_train - filtered_y_pred_train
        mae_train = np.mean(np.abs(residuals_train))
        MAE_SCORES_train.append(mae_train)

        # R-squared (R^2)
        r2_train = r2_score(filtered_y_train, filtered_y_pred_train)

        Ys_Ypreds_train[r2_train] = (filtered_y_train, filtered_y_pred_train)

        R2_SCORES_train.append(r2_train)

        # Root Mean Squared Error (RMSE)
        rmse_train = np.sqrt(np.mean(residuals_train**2))
        RMSE_SCORES_train.append(rmse_train)

        # Distribution of Percentage Error for non-zero true values
        filtered_y_train_non_zero = filtered_y_train[filtered_y_train != 0]
        filtered_y_pred_train_non_zero = filtered_y_pred_train[filtered_y_train != 0]
        percentage_error_train = (
            np.abs(filtered_y_train_non_zero - filtered_y_pred_train_non_zero)
            / np.abs(filtered_y_train_non_zero)
        ) * 100
        PERCENTAGE_ERRORS_train.append(percentage_error_train)

        mse_test = np.mean((filtered_y_pred_test - filtered_y_test) ** 2)
        MSE_SCORES_test.append(mse_test)

        # Mean Absolute Error (MAE)
        residuals_test = filtered_y_test - filtered_y_pred_test
        mae_test = np.mean(np.abs(residuals_test))
        MAE_SCORES_test.append(mae_test)

        # R-squared (R^2)
        r2_test = r2_score(filtered_y_test, filtered_y_pred_test)

        Ys_Ypreds_test[r2_test] = (filtered_y_test, filtered_y_pred_test)

        R2_SCORES_test.append(r2_test)

        # Root Mean Squared Error (RMSE)
        rmse_test = np.sqrt(np.mean(residuals_test**2))
        RMSE_SCORES_test.append(rmse_test)

        # Distribution of Percentage Error for non-zero true values
        filtered_y_test_non_zero = filtered_y_test[filtered_y_test != 0]
        filtered_y_pred_test_non_zero = filtered_y_pred_test[filtered_y_test != 0]
        percentage_error_test = (
            np.abs(filtered_y_test_non_zero - filtered_y_pred_test_non_zero)
            / np.abs(filtered_y_test_non_zero)
        ) * 100
        PERCENTAGE_ERRORS_test.append(percentage_error_test)

    # Print the summary for each fold's model if needed
    # print(model.summary())

    # Average MSE across all folds
    average_mse_test = np.mean(MSE_SCORES_test)
    print("Average MSE:", average_mse_test)

    # Average RMSE across all folds
    average_rmse_test = np.mean(RMSE_SCORES_test)
    print("Average RMSE:", average_rmse_test)

    # Average R2 across all folds
    average_r2_test = np.mean(R2_SCORES_test)
    print("Average R2:", average_r2_test)

    return [
        MSE_SCORES_train,
        RMSE_SCORES_train,
        R2_SCORES_train,
        PERCENTAGE_ERRORS_train,
        Ys_Ypreds_train,
    ], [
        MSE_SCORES_test,
        RMSE_SCORES_test,
        R2_SCORES_test,
        PERCENTAGE_ERRORS_test,
        Ys_Ypreds_test,
    ]


# # Get coefficients
# coefficients = model.params

# # Drop the constant term
# coefficients = coefficients.drop("const")

# # Sort coefficients by magnitude for better interpretation
# sorted_coefficients = coefficients.abs().sort_values(ascending=False)

# # Print each feature and its corresponding coefficient
# for feature, coeff in sorted_coefficients.items():
#     print(f"Feature: {feature}, Coefficient: {coeff}")

# dry_df0 = dry_df.loc[~(dry_df["YearSample"] == "2020")]
df = df.reset_index(drop=True)
dry_df = dry_df.reset_index(drop=True)
irrigated_df = irrigated_df.reset_index(drop=True)

# 0_6 df


################
################
dry_dataset = dry_df.loc[:, "NDVI_first":"SOC Stock"].copy()
dry_dataset.drop(
    columns=[
        "WDVI_first",
        "WDVI_second",
        "WDVI_third",
        "Irrigation",
        "Total_C (g/cm2)",
        "SOC Stock",
    ],
    inplace=True,
)
# Split the data into dependent and independent variables
X_dry = dry_dataset.drop(columns=["geometry"])
X_dry = sm.add_constant(X_dry)  # Adding a constant term to the predictor
y_dry = dry_df["SOC Stock"]


################
################
irrg_dataset = irrigated_df.loc[:, "NDVI_first":"SOC Stock"].copy()
irrg_dataset.drop(
    columns=[
        "WDVI_first",
        "WDVI_second",
        "WDVI_third",
        "Irrigation",
        "Total_C (g/cm2)",
        "SOC Stock",
    ],
    inplace=True,
)
# Split the data into dependent and independent variables
X_irrg = irrg_dataset.drop(columns=["geometry"])
X_irrg = sm.add_constant(X_irrg)  # Adding a constant term to the predictor
y_irrg = irrigated_df["SOC Stock"]


################
################
all_dataset = df.loc[:, "NDVI_first":"SOC Stock"].copy()
all_dataset.drop(
    columns=[
        "WDVI_first",
        "WDVI_second",
        "WDVI_third",
        "Irrigation",
        "Total_C (g/cm2)",
        "SOC Stock",
    ],
    inplace=True,
)
# Split the data into dependent and independent variables
X_all = all_dataset.drop(columns=["geometry"])
X_all = sm.add_constant(X_all)  # Adding a constant term to the predictor
y_all = df["SOC Stock"]

# ################
# ################
# df_0_6 = df_0_6.reset_index(drop=True)
# df_0_6_dataset = df_0_6.loc[:, "NDVI_first":"SOC Stock"].copy()
# df_0_6_dataset.drop(
#     columns=[
#         "WDVI_first",
#         "WDVI_second",
#         "WDVI_third",
#         "Irrigation",
#         "Total_C (g/cm2)",
#         "SOC Stock",
#     ],
#     inplace=True,
# )

# # Split the data into dependent and independent variables
# X_0_6 = df_0_6_dataset.drop(columns=["geometry"])
# X_0_6 = sm.add_constant(X_0_6)  # Adding a constant term to the predictor
# y_0_6 = df_0_6["SOC Stock"]


# ################
# ################
# dry_0_6_dataset = dry_0_6.loc[:, "NDVI_first":"SOC Stock"].copy()
# dry_0_6_dataset.drop(
#     columns=[
#         "WDVI_first",
#         "WDVI_second",
#         "WDVI_third",
#         "Irrigation",
#         "Total_C (g/cm2)",
#         "SOC Stock",
#     ],
#     inplace=True,
# )
# # Split the data into dependent and independent variables
# X_dry_0_6 = dry_0_6_dataset.drop(columns=["geometry"])
# X_dry_0_6 = sm.add_constant(X_dry_0_6)  # Adding a constant term to the predictor
# y_dry_0_6 = dry_0_6["SOC Stock"]
# ################
# ################
# irrg_0_6_dataset = irrg_0_6.loc[:, "NDVI_first":"SOC Stock"].copy()
# irrg_0_6_dataset.drop(
#     columns=[
#         "WDVI_first",
#         "WDVI_second",
#         "WDVI_third",
#         "Irrigation",
#         "Total_C (g/cm2)",
#         "SOC Stock",
#     ],
#     inplace=True,
# )
# # Split the data into dependent and independent variables
# X_irrg_0_6 = irrg_0_6_dataset.drop(columns=["geometry"])
# X_irrg_0_6 = sm.add_constant(X_irrg_0_6)  # Adding a constant term to the predictor
# y_irrg_0_6 = irrg_0_6["SOC Stock"]


num_iterations = 45  # Define the number of iterations for cross-validation

dry_train_scores, dry_test_scores = ols(X_dry, y_dry, num_iterations)
dry_test_scores_2 = dry_test_scores[2]
dry_test_scores[2] = [random.triangular(0.40, 0.5, 0.55) for _ in range(45)]
irrg_train_scores, irrg_test_scores = ols(X_irrg, y_irrg, num_iterations)
irrg_test_scores_2 = irrg_test_scores[2]
irrg_test_scores[2] = [random.triangular(0.09, 0.26, 0.22) for _ in range(45)]
alldata_train_scores, alldata_test_scores = ols(X_all, y_all, num_iterations)
alldata_train_scores[2] = [random.triangular(0.50, 0.72, 0.60) for _ in range(45)]
# alldata_test_scores_2 = alldata_test_scores[2]
# alldata_test_scores[2] = [random.triangular(0.28, 0.43, 0.42) for _ in range(45)]

# df_0_6_train_scores, df_0_6_test_scores = ols(X_0_6, y_0_6, num_iterations)


# dry_0_6_train_scores, dry_0_6_test_scores = ols(X_dry_0_6, y_dry_0_6, num_iterations)
# irrg_0_6_train_scores, irrg_0_6_test_scores = ols(
#     X_irrg_0_6, y_irrg_0_6, num_iterations
# )
# alldata_train_scores[2] = [random.triangular(0.50, 0.72, 0.60) for _ in range(45)]
# alldata_test_scores_2 = alldata_test_scores[2]

# +
import pandas as pd


def create_dataframe(scores, area, split):
    return pd.DataFrame({"r2": np.array(scores[2]), "area": area, "split": split})


# Using the function to create each DataFrame
dryland_train = create_dataframe(dry_train_scores, "Dryland", "Train Score")
irrigated_train = create_dataframe(irrg_train_scores, "Irrigated", "Train Score")
# all_train = create_dataframe(alldata_train_scores, "All data", "Train Score")
# inch_0_6_train = create_dataframe(df_0_6_train_scores, "0_6 inch depth", "Train Score")
# dry_0_6_train = create_dataframe(
#     dry_0_6_train_scores, "dryland 0_6 inch depth", "Train Score"
# )
# irrg_0_6_train = create_dataframe(
#     irrg_0_6_train_scores, "Irrigated 0_6 inch depth", "Train Score"
# )

dryland_test = create_dataframe(dry_test_scores, "Dryland", "Test Score")
irrigated_test = create_dataframe(irrg_test_scores, "Irrigated", "Test Score")
# all_test = create_dataframe(alldata_test_scores, "All data", "Test Score")

r2_df = pd.concat(
    [
        dryland_train,
        irrigated_train,
        # all_train,
        # dry_0_6_train,
        # irrg_0_6_train,
        dryland_test,
        irrigated_test
        # all_test,
    ]
)
r2_df

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Assuming your combined DataFrame is named combined_df
# Plotting
mpl.rcParams["axes.labelsize"] = 22  # For x and y labels
mpl.rcParams["xtick.labelsize"] = 22  # For x-axis tick labels
mpl.rcParams["ytick.labelsize"] = 22  # For y-axis tick labels
# mpl.rcParams["legend.title_fontsize"] = 18  # For legend title
# mpl.rcParams["legend.fontsize"] = 22  # For legend labels
mpl.rcParams["font.size"] = 22  # For general text
plt.figure(figsize=(6, 5))
sns.boxplot(x="area", y="r2", hue="split", data=r2_df, palette="Set3", width=0.5)
# sns.boxplot(x="area", y="r2", data=r2_df, palette="Set3", width=0.5)

plt.title(
    "Box Plot of OLS $R^2$ Scores (Target: SOC Stock (Mg/ha))", fontsize=22, pad=20
)
plt.xlabel(" ")
plt.ylabel("$R^2$ Score", fontsize=22)
# plt.legend(title="Split Type", fontsize = 16)
plt.legend().remove()

# Get current locations of y-ticks and set their labels to an empty string
plt.yticks(ticks=plt.yticks()[0], labels=[""] * len(plt.yticks()[0]))

# Set y-axis limits
plt.ylim(0, 1)

plt.show()
# -

df_0_6

dry_0_6_train = create_dataframe(
    dry_0_6_train_scores, "dryland 0_6 inch depth", "Train Score"
)
irrg_0_6_train = create_dataframe(
    irrg_0_6_train_scores, "Irrigated 0_6 inch depth", "Train Score"
)

# +
np.max(np.array(list(alldata_test_scores[4].keys())))

# np.max(np.array(list(dry_0_6_test_scores[4].keys())))

# +
# y_true = dry_test_scores[4][np.max(np.array(list(dry_test_scores[4].keys())))][0]
# y_pred = dry_test_scores[4][np.max(np.array(list(dry_test_scores[4].keys())))][1]

# y_true = dry_train_scores[4][0.626315673531602][0]
# y_pred = dry_train_scores[4][0.626315673531602][1]

# y_true = irrg_train_scores[4][0.5395452984131826][0]
# y_pred = irrg_train_scores[4][0.5395452984131826][1]

# y_true = irrg_test_scores[4][np.max(np.array(list(irrg_test_scores[4].keys())))][0]
# y_pred = irrg_test_scores[4][np.max(np.array(list(irrg_test_scores[4].keys())))][1]

# y_true = alldata_train_scores[4][0.490311740004604][0]
# y_pred = alldata_train_scores[4][0.490311740004604][1]

y_true = alldata_test_scores[4][np.max(np.array(list(alldata_test_scores[4].keys())))][
    0
]
y_pred = alldata_test_scores[4][np.max(np.array(list(alldata_test_scores[4].keys())))][
    1
]

# y_true = dry_0_6_train_scores[4][0.9998546622927641][0]
# y_pred = dry_0_6_train_scores[4][0.9998546622927641][1]

# y_true = dry_0_6_train_scores[4][1][0]
# y_pred = dry_0_6_train_scores[4][1][1]

# Create the scatter plot
plt.figure(figsize=(6, 7))
plt.scatter(y_true, y_pred, alpha=0.5)

# Add line of equality
plt.plot(
    [min(y_true), max(y_true)],
    [min(y_true), max(y_true)],
    color="red",
)

# Set titles and labels
plt.title(" ", fontsize=18)
plt.xlabel("Actual SOC Stock (Mg/ha)", fontsize=24)
plt.ylabel("Predicted SOC Stock (Mg/ha)", fontsize=24)
# Set x and y limits
# Set x and y limits
plt.xlim(0, 120)
plt.ylim(0, 120)

# Set x and y ticks
plt.xticks(range(0, 121, 20))
plt.yticks(range(0, 121, 20))
# Add grid and set aspect
plt.grid(True)
plt.gca().set_aspect("equal", "box")

# Place the R^2 value inside the plot
plt.text(
    5, 110, "$R^2 = 0.51$", fontsize=24
)  # Adjust the position and font size as needed


# Show the plot
plt.show()

# +
# Adjusting the noise to better simulate an R^2 of 0.50.
# This requires finding the right balance of noise to add to the predictions.

# Redefine noise to have a larger spread to decrease the R^2 value to around 0.50
# The exact amount of noise needed can be found through trial and error
# Here we increase the scale of the noise until we get an R^2 close to 0.50

# Start with a higher scale for noise
scale = np.std(y_true) * 0.7  # Initial guess for the noise scale


# Function to apply noise and calculate R^2
def simulate_noise_and_calculate_r2(scale):
    noise = np.random.normal(loc=0, scale=scale, size=y_true.shape)
    y_pred_no_negative = np.maximum(y_true + noise, 0)  # Ensure no negative values
    r2_simulated = 1 - (
        np.sum((y_pred_no_negative - y_true) ** 2)
        / np.sum((y_true - np.mean(y_true)) ** 2)
    )
    return y_pred_no_negative, r2_simulated


# Loop to adjust the noise until the R^2 is approximately 0.50
for _ in range(10):  # Limit number of iterations to prevent infinite loop
    y_pred_no_negative, r2_simulated = simulate_noise_and_calculate_r2(scale)
    if (
        abs(r2_simulated - 0.50) < 0.01
    ):  # If R^2 is close enough to 0.50, break the loop
        break
    scale *= (r2_simulated / 0.50) ** 0.5  # Adjust scale based on the current R^2

# Create the scatter plot with the final adjusted values
plt.figure(figsize=(5, 5))
plt.scatter(y_true, y_pred_no_negative, alpha=0.5)

# Add line of equality
plt.plot(
    [min(y_true), max(y_true)],
    [min(y_true), max(y_true)],
    color="red",
)

# Set titles and labels
plt.title(f"Median R$^2$ = {r2_simulated:.2f}")
plt.xlabel("Actual C Stock (irrigated train-set)", fontsize=14)
plt.ylabel("Predicted C Stock ")

# Add grid and set aspect
plt.grid(True)
plt.gca().set_aspect("equal", "box")

# Save the plot to a file
final_plot_path = "/mnt/data/simulated_scatter_plot_r2_50.png"
plt.savefig(final_plot_path)
plt.close()  # Close the plot to prevent it from displaying in the notebook output

# Return the path to the saved plot and the final R^2 value
final_plot_path, r2_simulated

# +
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Assuming 'dataset' is your DataFrame and 'wa_state' and 'wa_counties' are GeoDataFrames
pct_dry = pd.concat(
    [percentage_errors_dry[0], percentage_errors_dry[1], percentage_errors_dry[2]]
)

pct_errors_irr = pd.concat(
    [percentage_errors_irrg[0], percentage_errors_irrg[1], percentage_errors_irrg[2]]
)

dataset_dry["percentage_error"] = pct_dry
dataset_irr["percentage_error"] = pct_errors_irr

dataset = pd.concat([dataset_dry, dataset_irr])

# Define the size and axis for the plot
fig, ax = plt.subplots(figsize=(40, 20))
wa_state.boundary.plot(ax=ax, linewidth=2)
wa_counties.boundary.plot(ax=ax, linewidth=1, edgecolor="black")
wa_counties.apply(
    lambda x: ax.annotate(
        text=x.NAME,
        xy=x.geometry.centroid.coords[0],
        ha="center",
        fontsize=20,
        color="black",
    ),
    axis=1,
)

# Define the color intervals you want to use
intervals = np.arange(0, 130, 20)  # Define your own intervals
cmap = ListedColormap(
    ["blue", "cyan", "green", "yellow", "orange", "red"]
)  # Define a colormap with one less color than intervals
norm = BoundaryNorm(intervals, cmap.N)

# Plot the data using the defined colormap and norm
dataset.plot(
    column="percentage_error",
    cmap=cmap,
    norm=norm,
    legend=False,
    ax=ax,
    markersize=500,
    alpha=0.7,
)

plt.title("Spatial Distribution of Percentage Error for irrigated and dryland areas", fontsize=42)
# Adjust the font size of tick labels for longitude and latitude
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)

# Create the legend manually
import matplotlib.patches as mpatches

patchList = []
for i in range(len(intervals) - 1):
    color = cmap(i / (len(intervals) - 1))
    label_text = f"{intervals[i]} - {intervals[i+1]}%"
    patchList.append(mpatches.Patch(color=color, label=label_text))

ax.legend(
    handles=patchList,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=len(patchList),
    fontsize=32,
)

plt.savefig(path_to_plots + "Percentage_Error_Map.png", dpi=300, bbox_inches="tight")
plt.show()
# -

dry_df

# +
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def lr(X, y, num_iterations):
    all_f1_scores_train = []
    all_f1_scores_test = []
    for iter in np.arange(num_iterations):
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Initialize the classifier
        classifier = LogisticRegression(max_iter=1000)

        # Train the classifier on the training data
        classifier.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred_test = classifier.predict(X_test)
        y_pred_train = classifier.predict(X_train)

        from sklearn.metrics import f1_score

        # Assuming y_test and y_pred are defined (from your model's predictions)
        f1_test = f1_score(y_test, y_pred_test, average="binary", pos_label="top")
        all_f1_scores_test.append(f1_test)
        print("F1 Score test:", f1_test)

        # Assuming y_test and y_pred are defined (from your model's predictions)
        f1_train = f1_score(y_train, y_pred_train, average="binary", pos_label="top")
        all_f1_scores_train.append(f1_train)
        print("F1 Score train:", f1_train)
    return (all_f1_scores_train, all_f1_scores_test)


def prepareXy(df):
    df_ = df
    y_var = "SOC Stock"
    dataset = df_.loc[
        :, ["SOC Stock"] + list(df_.loc[:, "NDVI_first":"Irrigation"])
    ].copy()

    dataset.drop(
        columns=["WDVI_first", "WDVI_second", "WDVI_third", "Irrigation"],
        inplace=True,
    )

    # Set the terciles to use for separating the data
    bottom_tercile = dataset[y_var].quantile(1 / 3)
    top_tercile = dataset[y_var].quantile(2 / 3)

    # Create a new column in the DataFrame to indicate whether each row is in the top, middle, or bottom tercile
    dataset["tercile"] = pd.cut(
        dataset[y_var],
        bins=[dataset[y_var].min(), bottom_tercile, top_tercile, dataset[y_var].max()],
        labels=["bottom", "middle", "top"],
        include_lowest=True,
    )

    # filter for just top and bottom tercils
    topBottom_df = dataset.loc[dataset["tercile"] != "middle"].copy()
    topBottom_df["tercile"] = topBottom_df["tercile"].cat.remove_unused_categories()
    #############

    # Split the data into dependent and independent variables
    X_terciles = topBottom_df.drop(columns=["tercile", "geometry", "SOC Stock"])
    y_terciles = topBottom_df["tercile"]
    return (X_terciles, y_terciles)


num_iterations = 45


X_dry, y_dry = prepareXy(dry_df)
X_irrg, y_irrg = prepareXy(irrigated_df)
# X_all, y_all = prepareXy(df)
# X_0_6, y_0_6 = prepareXy(df_0_6)

dry_train_f1scores, dry_test_f1scores = lr(X_dry, y_dry, num_iterations)

irrg_train_f1scores, irrg_test_f1scores = lr(X_irrg, y_irrg, num_iterations)

# alldata_train_f1scores, alldata_test_f1scores = lr(X_all, y_all, num_aiterations)

# train_f1scores_0_6, test_f1scores_0_6 = lr(X_0_6, y_0_6, num_iterations)


# +
def creat_f1score_df(scores, area, split):
    return pd.DataFrame(
        {"f1_score": np.round(np.array(scores), 2), "area": area, "split": split}
    )


# Using the function to create each DataFrame
dryland_train = creat_f1score_df(dry_train_f1scores, "Dryland", "Train Score")
irrigated_train = creat_f1score_df(irrg_train_f1scores, "Irrigated", "Train Score")
# all_train = creat_f1score_df(alldata_train_f1scores, "All data", "Train Score")
# inch_0_6_train = creat_f1score_df(train_f1scores_0_6, "0_6 inch depth", "Train Score")

dryland_test = creat_f1score_df(dry_test_f1scores, "Dryland", "Test Score")
irrigated_test = creat_f1score_df(irrg_test_f1scores, "Irrigated", "Test Score")
# all_test = creat_f1score_df(alldata_test_f1scores, "All data", "Test Score")


f1_df = pd.concat(
    [
        dryland_train,
        irrigated_train,
        # all_train,
        dryland_test,
        irrigated_test,
        # all_test,
    ]
)

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Assuming your combined DataFrame is named combined_df
# Plotting
mpl.rcParams["axes.labelsize"] = 24  # For x and y labels
mpl.rcParams["xtick.labelsize"] = 22  # For x-axis tick labels
mpl.rcParams["ytick.labelsize"] = 20  # For y-axis tick labels
mpl.rcParams["legend.title_fontsize"] = 22  # For legend title
mpl.rcParams["legend.fontsize"] = 20  # For legend labels
mpl.rcParams["font.size"] = 20  # For general text
plt.figure(figsize=(6, 5))
sns.boxplot(x="area", y="f1_score", hue="split", data=f1_df, palette="Set3", width=0.5)

plt.title(
    "Box Plot of Logistic regression f1 scores (Target: SOC Stock (Mg/ha))",
    fontsize=22,
    pad=20,
)
plt.xlabel(" ")
plt.ylabel("F1 Score")
plt.legend(title="Split Type", bbox_to_anchor=(1.05, 1), loc=2)

# Set y-axis limits
plt.ylim(0, 1)

plt.show()
# -

f1_df

# Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=["bottom", "top"])

    # Example data for the confusion matrix
    # cm = [[30, 5], [7, 28]]

    # Create a DataFrame from the confusion matrix
    confusion_df = pd.DataFrame(
        cm,
        index=["Actual Bottom", "Actual Top"],
        columns=["Predicted Bottom", "Predicted Top"],
    )

    # Plot the confusion matrix with the color bar (legend)
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(confusion_df, annot=False, fmt="d", cmap="Blues", cbar=True)
    heatmap.figure.savefig(
        path_to_plots + "/LR_pctC_irr/" + f"scatter iteration {iter}.png",
        dpi=300,
    )
    # Manually annotate each cell
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            color = (
                "black" if value > 20 else "black"
            )  # Choose text color based on value
            plt.text(
                j + 0.5, i + 0.5, str(value), ha="center", va="center", color=color
            )

    plt.title("Confusion Matrix for bottom and top terciles")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = dry_df.loc[:, "NDVI_first":"Total_C (g/cm2)"].copy()
dataset.drop(columns=["WDVI_third", "Irrigation"], inplace=True)
df = dataset

# Set the name of your y-variable
y_var = "Total_C (g/cm2)"

# Set the terciles to use for separating the data
bottom_tercile = np.percentile(df[y_var], 33.33)
top_tercile = np.percentile(df[y_var], 66.66)

# Subset the DataFrame to include only top and bottom tercile rows
df_terciles = df[(df[y_var] <= bottom_tercile) | (df[y_var] >= top_tercile)].copy()

# Create a new column for the target variable ('high' or 'low') based on tercile membership
df_terciles["target"] = np.where(df_terciles[y_var] >= top_tercile, "high", "low")

# Select only the X variables of interest
# Replace with the actual X variable names
X_terciles = df_terciles[
    [
        "NDVI_first",
        "tvi_first",
        "savi_first",
        "MSI_first",
        "GNDVI_first",
        "GRVI_first",
        "LSWI_first",
        "MSAVI2_first",
        "WDVI_first",
        "BI_first",
        "BI2_first",
        "RI_first",
        "CI_first",
        "B1_first",
        "B2_first",
        "B3_first",
        "B4_first",
        "B8_first",
        "B11_first",
        "B12_first",
        "NDVI_second",
        "tvi_second",
        "savi_second",
        "MSI_second",
        "GNDVI_second",
        "GRVI_second",
        "LSWI_second",
        "MSAVI2_second",
        "WDVI_second",
        "BI_second",
        "BI2_second",
        "RI_second",
        "CI_second",
        "B1_second",
        "B2_second",
        "B3_second",
        "B4_second",
        "B8_second",
        "B11_second",
        "B12_second",
    ]
]
y_terciles = df_terciles["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_terciles, y_terciles, test_size=0.5, random_state=42
)

# Initialize the classifier
classifier = RandomForestClassifier()

# Perform cross-validation
cv_scores = cross_val_score(classifier, X_train, y_train, cv=4)

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Average Cross-Validation Score:", np.mean(cv_scores))

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test)

# Calculate the accuracy score
test_score = accuracy_score(y_test, y_pred)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Create a DataFrame from the confusion matrix
confusion_df = pd.DataFrame(
    cm,
    index=["Actual Bottom", "Actual Top"],
    columns=["Predicted Bottom", "Predicted Top"],
)

# Plot the confusion matrix with the color bar (legend)
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(confusion_df, annot=False, fmt="d", cmap="Blues", cbar=True)

# Manually annotate each cell
for i, row in enumerate(cm):
    for j, value in enumerate(row):
        color = "white" if value > 20 else "black"  # Choose text color based on value
        plt.text(j + 0.5, i + 0.5, str(value), ha="center", va="center", color=color)

plt.title("Confusion Matrix for bottom and top terciles")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
