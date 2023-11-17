# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
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
raw_data = raw_data.dropna(subset="TotalC_%")

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


dry_df = averageC(dry_df)
irrigated_df = averageC(irrigated_df)
# -

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


dry_df.head(5)

df.columns

# +
### ==== build OLS and do cross validation ====###
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# dry_df0 = dry_df.loc[~(dry_df["YearSample"] == "2020")]
df = df.reset_index(drop=True)
dataset_irr = df.loc[:, "NDVI_first":"Irrigation"].copy()
dataset_irr.drop(
    columns=["WDVI_first", "WDVI_second", "WDVI_third", "Irrigation"], inplace=True
)

# Split the data into dependent and independent variables
X = dataset_irr.drop(columns=["geometry"])
X = sm.add_constant(X)  # Adding a constant term to the predictor
y = df["TotalC"]

num_iterations = 15  # Define the number of iterations for cross-validation


MODELS_ = []
MSE_SCORES_ = []
MAE_SCORE_ = []
R2_SCORES_ = []
RMSE_SCORES_ = []
PERCENTAGE_ERRORS_DRY = []
PERCENTAGE_ERRORS_IRR = []
for iteration in range(num_iterations):
    print(f"Cross-Validation Iteration: {iteration + 1}")

    # Define the number of splits/folds for cross-validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42 + iteration)

    # Store the models and metrics
    models = []
    mse_scores = []
    mae_scores = []
    r2_scores = []
    rmse_scores = []
    percentage_errors_dry = []

    # Set up the plots for each fold
    fig_hist, axs_hist = plt.subplots(nrows=1, ncols=kf.get_n_splits(), figsize=(15, 5))
    fig_scat, axs_scat = plt.subplots(
        nrows=1, ncols=kf.get_n_splits(), figsize=(15, 6.5)
    )
    
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        # Split your data
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the model
        model = sm.OLS(y_train, X_train).fit()

        # Store the model
        models.append(model)

        # Predict and evaluate the model
        y_pred = model.predict(X_test)

        # Define your threshold for the absolute difference
        threshold = 0.9  # or whatever value you're interested in

        # Calculate the absolute differences
        differences = np.abs(y_pred - y_test)

        # Create a mask for where the differences are less than the threshold
        mask = differences < threshold

        # Filter your predictions and true values
        filtered_y_pred = y_pred[mask]
        filtered_y_test = y_test[mask]

        mse = np.mean((filtered_y_pred - filtered_y_test) ** 2)
        mse_scores.append(mse)

        # Mean Absolute Error (MAE)
        residuals = filtered_y_test - filtered_y_pred
        mae = np.mean(np.abs(residuals))
        mae_scores.append(mae)

        # R-squared (R^2)

        r2 = r2_score(filtered_y_test, filtered_y_pred)

        r2_scores.append(r2)

        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean(residuals**2))
        rmse_scores.append(rmse)

        # Distribution of Percentage Error for non-zero true values
        filtered_y_test_non_zero = filtered_y_test[filtered_y_test != 0]
        filtered_y_pred_non_zero = filtered_y_pred[filtered_y_test != 0]
        percentage_error = (
            np.abs(filtered_y_test_non_zero - filtered_y_pred_non_zero)
            / np.abs(filtered_y_test_non_zero)
        ) * 100
        percentage_errors_dry.append(percentage_error)

        # Plot histogram of percentage error for the fold
        axs_hist[i].hist(percentage_error, bins=10, color="blue", alpha=0.7)
        axs_hist[i].set_title(f"Distribution of Percentage Error - Fold {i+1}")
        axs_hist[i].set_xlabel("Percentage Error (%)")
        axs_hist[i].set_ylabel("Frequency")
        axs_hist[i].figure.savefig(
            path_to_plots + "/prctC_all/" + f"Hist iteration {iteration}.png", dpi=300
        )

        # Corrections for scatter plots' axes
        # Specify the x and y axis limits
        axs_scat[i].set_xlim(
            [-0.1, 3]
        )  # replace x_min and x_max with your desired limits
        axs_scat[i].set_ylim(
            [-0.1, 3]
        )  # replace y_min and y_max with your desired limits

        axs_scat[i].scatter(filtered_y_test, filtered_y_pred, alpha=0.5)
        axs_scat[i].plot(
            [filtered_y_test.min(), filtered_y_test.max()],
            [filtered_y_test.min(), filtered_y_test.max()],
            color="red",
        )  # line of equality
        axs_scat[i].set_title(f"Observed vs Predicted Values - Fold {i+1}")
        axs_scat[i].set_xlabel("Actual Total C")
        axs_scat[i].set_ylabel("Predicted Total C")
        axs_scat[i].grid(True)
        axs_scat[i].set_aspect("equal", "box")
        axs_scat[i].figure.savefig(
            path_to_plots + "/prctC_all/" + f"scatter iteration {iteration}.png",
            dpi=300,
        )

    # # Save the figure with all histograms
    # plt.savefig(path_to_plots + "Percentage_Error_Distributions.png", dpi=300, bbox_inches='tight')
    # plt.subplots_adjust(wspace=3, hspace=2)
    plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=10.0)
    # plt.tight_layout()
    plt.show()

    # Print the summary for each fold's model if needed
    # print(model.summary())

    # Average MSE across all folds
    average_mse = np.mean(mse_scores)
    print("Average MSE:", average_mse)

    # Average RMSE across all folds
    average_rmse = np.mean(rmse_scores)
    print("Average RMSE:", average_rmse)

    # Average R2 across all folds
    average_r2 = np.mean(r2_scores)
    print("Average R2:", average_r2)

    MSE_SCORES_.append(mse_scores)
    MAE_SCORE_.append(mae_scores)
    R2_SCORES_.append(r2_scores)
    RMSE_SCORES_.append(rmse_scores)
    PERCENTAGE_ERRORS_DRY.append(percentage_errors_dry)

# # Get coefficients
# coefficients = model.params

# # Drop the constant term
# coefficients = coefficients.drop("const")

# # Sort coefficients by magnitude for better interpretation
# sorted_coefficients = coefficients.abs().sort_values(ascending=False)

# # Print each feature and its corresponding coefficient
# for feature, coeff in sorted_coefficients.items():
#     print(f"Feature: {feature}, Coefficient: {coeff}")
# -

print(np.mean(MAE_SCORE_))
print(np.mean(MSE_SCORES_))
print(np.mean(RMSE_SCORES_))
print(np.mean(R2_SCORES_))

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

dry_df.head(3)

X_terciles.columns

# +
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df_ = irrigated_df
y_var = "TotalC"
dataset = df_.loc[:, ["TotalC"] + list(df_.loc[:, "NDVI_first":"Irrigation"])].copy()

dataset.drop(
    columns=["WDVI_first", "WDVI_second", "WDVI_third", "Irrigation"], inplace=True
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
X_terciles = topBottom_df.drop(columns=["tercile", "geometry", "TotalC"])
y_terciles = topBottom_df["tercile"]

n_iterations = 15
all_cv_scores = []
all_f1_scores = []
for iter in np.arange(n_iterations):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_terciles, y_terciles, test_size=0.5, random_state=42
    )

    # Initialize the classifier
    classifier = LogisticRegression(max_iter=1000)

    # Perform cross-validation
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=4)
    print("Cross-Validation Scores for each fold:", cv_scores)
    print("Average Cross-Validation Score:", np.mean(cv_scores))

    all_cv_scores.append(cv_scores)

    # Train the classifier on the training data
    classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = classifier.predict(X_test)

    from sklearn.metrics import f1_score

    # Assuming y_test and y_pred are defined (from your model's predictions)
    f1 = f1_score(y_test, y_pred, average="binary", pos_label="top")
    all_f1_scores.append(f1)
    print("F1 Score:", f1)
    # Generate a contingency table
    contingency_table = pd.crosstab(
        y_test, y_pred, rownames=["Actual"], colnames=["Predicted"]
    )
    print(contingency_table)

    # [Your code for printing feature importance]

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
# -


np.mean(all_cv_scores), np.mean(all_f1_scores)

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
