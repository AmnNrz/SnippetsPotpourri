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
# Read data
path_to_data = ("/home/amnnrz/OneDrive - a.norouzikandelati/"
                "Ph.D/Projects/DSFAS/Data/")

path_to_plots = ("/home/amnnrz/OneDrive - a.norouzikandelati/"
                 "Ph.D/Projects/DSFAS/Plots/")

# path_to_data = ("/home/amnnrz/GoogleDrive - "
#                 "msaminnorouzi/PhD/Projects/DSFAS/Data/")

# path_to_plots = ("/home/amnnrz/GoogleDrive - msaminnorouzi/"
#                  "PhD/Projects/DSFAS/Plots/")

dry_df = pd.read_csv(
    path_to_data + "Carbon&satellite_data_dry_joined_v1.csv")

dry_irig_df = pd.read_csv(
    path_to_data + "Carbon&satellite_data_dryIrgted_joined_v1.csv")

raw_data = pd.read_csv(path_to_data + "EWA_carbon_subset.csv")
raw_data = raw_data.dropna(subset='TotalC_%')

# Convert year to integer
dry_df['YearSample'] = dry_df['YearSample'].astype(int)

# remove old index column
dry_df.drop(columns='index', axis=1, inplace=True)

dry_df_raw = gpd.read_file(path_to_data + 
                         ('/GIS_Data/Csample_buffer_shp/'
                          'C_samples_dryland_shp/C_samples_dryland_shp.shp'))

allSamples_df = pd.read_csv(path_to_data + "Carbon&satellite_data_dryIrgted_joined_v1.csv")

# +
# raw_data['DepthSampled_inches'].value_counts(), dry_df_raw['DepthSampl'].value_counts()
# dry_irig_df['TotalC'].isna().value_counts(), dry_irig_df['DepthSampl'].value_counts()
# dry_df['TotalC'].isna().value_counts(), dry_df['DepthSampl'].value_counts()
# dry_irig_df['TotalC'].isna().value_counts(), dry_irig_df['DepthSampl'].value_counts()
# dry_irig_df['SampleID'].isin(dry_df['SampleID']).value_counts()
# dry_irig_df['DepthSampl'].value_counts()
# -

len(dry_df['SampleID'].unique())

# check 0_6 -- 0_12 samples' year
sampleYear_6_12 = dry_df.loc[dry_df['DepthSampl'] ==
                         '0_6', 'YearSample'].values[0]
print('Two-depth samples are for:', f'{sampleYear_6_12}')


# +
# Get average of total_C over 0-6 and 6-12 inches samples 
dup_df = dry_df.loc[dry_df.SampleID.duplicated(keep=False)]
dup_df

averaged_C = pd.DataFrame([])
averaged_C['SampleID'] = dup_df.SampleID.unique()
for id in dup_df.SampleID.unique():
    averaged_C.loc[averaged_C["SampleID"] == id, "TotalC"] = np.mean(
        dup_df.loc[dup_df["SampleID"] == id, "TotalC"])

averaged_C.head(5)
# -

dry_df = dry_df.loc[~dry_df.SampleID.duplicated()]
dry_df.loc[dry_df.SampleID.isin(averaged_C.SampleID),
        'TotalC'] = averaged_C['TotalC'].values
dry_df.loc[dry_df.SampleID.isin(averaged_C.SampleID), 'TotalC']
dry_df.loc[dry_df['DepthSampl'] == '0_6', 'DepthSampl'] = '0_12'
dry_df

# +
# Normalize band values
largeValue_idx = (dry_df.iloc[:, 11:].describe().loc["min"] < -2) | \
    (dry_df.iloc[:, 8:].describe().loc["max"] > 2)
largeValue_cols = largeValue_idx[largeValue_idx].index

scaler = StandardScaler()

# fit the scaler on the selected columns
scaler.fit(dry_df[largeValue_cols].copy())

# transform the selected columns to have zero mean and unit variance
dry_df.loc[:, largeValue_cols] = scaler.transform(dry_df[largeValue_cols].copy())
dry_df.describe()

# -

# Convert Total_C_% to g/cm2
# "total_c_%" /100 * height * A * 2.54 (inch to cm) * BD
def tCarbon_to_gcm2(df):
    df.loc[:, "Total_C (g/cm2)"] = df["TotalC"]/100 * 12 * 2.54 * 1 * df["BD_g_cm3"]
    return df
tCarbon_to_gcm2(dry_df)


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


# +
######=====    Density Distribution of Total C grouped by year  =====#######
# Set the style and size of the plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Loop through each year and plot the density distribution
for year in dry_df['YearSample'].unique():
    subset = dry_df[dry_df['YearSample'] == year]
    sns.kdeplot(subset['Total_C (g/cm2)'], label=f'Year {year}', fill=True)

# Add labels and title
plt.xlabel('Total_C (g/cm2)')
plt.ylabel('Density')
plt.title('Density Distribution of Total C Grouped by Year')
plt.legend()

# Show the plot
plt.show()
# -


# # Create map of data

dry_df

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

# -

allSamples_df.shape, dry_df.shape

dry_df.columns[dry_df.columns.isin(allSamples_df.columns)]

# +


# Convert dataframes to GeoDataFrames
dry_df = gpd.GeoDataFrame(dry_df, geometry=gpd.points_from_xy(dry_df.Longitude, dry_df.Latitude))
allSamples_df = gpd.GeoDataFrame(allSamples_df, geometry=gpd.points_from_xy(allSamples_df.Longitude, allSamples_df.Latitude))

# Remove reduntant columns
allSamples_df = allSamples_df.loc[:, 'TotalC':].copy()
dry_df = dry_df.loc[:, 'TotalC':].copy()

dry_df.reset_index(drop = True, inplace=True)
allSamples_df.reset_index(drop = True, inplace=True)

dry_df = dry_df[dry_df.columns[dry_df.columns.isin(allSamples_df.columns)]]

# merge two dataframes
irrigated_df = allSamples_df.loc[
    ~(allSamples_df['SampleID'].isin(dry_df['SampleID']))].copy()
# add irrigation column
irrigated_df['Irrigation'] = 'Irrigated'
dry_df['Irrigation'] = 'Dryland'

df = pd.concat([dry_df, irrigated_df])
dry_df = tCarbon_to_gcm2(dry_df)
# df
# -

(dry_df.columns == irrigated_df.columns)

len(df['SampleID'].unique())

df['Irrigation'].value_counts()

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

# -


# Renaming columns
df.columns = df.columns.str.replace('_first', '_MAM')
df.columns = df.columns.str.replace('_second', '_JJA')
df.columns = df.columns.str.replace('_third', '_SON')
df.to_csv(path_to_data + 'data_snapshot.csv')

df.columns

# +
###### ======   Density Distribution of features for top and bottom terciles =====######
# Renaming columns
topBottom_df.columns = topBottom_df.columns.str.replace('_first', '_MAM')
topBottom_df.columns = topBottom_df.columns.str.replace('_second', '_JJA')
topBottom_df.columns = topBottom_df.columns.str.replace('_third', '_SON')
# Get list of columns to be plotted
x_vars = topBottom_df.columns.drop([y_var, 'tercile'])

# Create separate figures for every 20 variables
num_plots_per_fig = 20
num_figs = -(-len(x_vars) // num_plots_per_fig)  # Ceiling division

for fig_num in range(num_figs):
    fig, axes = plt.subplots(4, 5, figsize=(
        20, 16))  # Adjust figsize as needed
    axes = axes.ravel()  # Flatten the axes for easier indexing

    for ax_num, x_var in enumerate(x_vars[fig_num*num_plots_per_fig: (fig_num+1)*num_plots_per_fig]):
        # Choose your own colors
        for tercile, color in zip(['bottom', 'top'], ['#440154', '#21918c']):
            subset = topBottom_df[topBottom_df['tercile'] == tercile]
            sns.kdeplot(ax=axes[ax_num], data=subset, x=x_var,
                        color=color, label=tercile, fill=True)

        # axes[ax_num].set_title(x_var)
        # Set font size for x and y labels
        axes[ax_num].set_xlabel(x_var, fontsize=15)  # Adjust this value as needed
        # Adjust this value as needed
        axes[ax_num].set_ylabel('Density', fontsize=15)
    # Adjust vertical spacing between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    
    
    # Handle any unused axes
    for ax_num in range(ax_num+1, 20):
        axes[ax_num].axis('off')
    
    # Add a legend to the figure (not to each individual plot)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='tercile',
               loc='upper right', bbox_to_anchor=(1, 0.5),
               prop={'size': 15}, title_fontsize='20')


    plot_name = path_to_plots + f"figure_{fig_num + 1}.png"
    fig.savefig(plot_name, dpi=300, bbox_inches='tight')
    # plt.close(fig)  # Close the current figure to free up memory


# +
### ==== build OLS ====###

dataset = dry_df.loc[:, "NDVI_first":"Total_C (g/cm2)"].copy()
dataset.drop(columns=["WDVI_first", "WDVI_second", "WDVI_third", "Irrigation"], inplace=True)

# Split the data into dependent and independent variables
X = dataset.drop(columns=['Total_C (g/cm2)', 'geometry'])
X = sm.add_constant(X)  # Adding a constant term to the predictor
y = dataset['Total_C (g/cm2)']

# Ordinary Least Squares (OLS) regression
model = sm.OLS(y, X).fit()

# Predict using the OLS model
y_pred = model.predict(X)

# Model Evaluation
# Calculate residuals
residuals = y - y_pred

# Mean Absolute Error (MAE)
mae = np.mean(np.abs(residuals))

# R-squared (R^2)
r2 = model.rsquared

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(np.mean(residuals**2))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R^2): {r2}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Distribution of Percentage Error
percentage_error = (residuals / y) * 100

# Calculate percetage error with non-zero y
percentage_error = (np.abs(y[(y != 0)] - y_pred[(y != 0)]) / np.abs(y[(y != 0)]))
# Replace infinities with NaNs
percentage_error.replace([np.inf, -np.inf], np.nan, inplace=True)


# Get coefficients
coefficients = model.params

# Drop the constant term
coefficients = coefficients.drop("const")

# Sort coefficients by magnitude for better interpretation
sorted_coefficients = coefficients.abs().sort_values(ascending=False)

# Print each feature and its corresponding coefficient
for feature, coeff in sorted_coefficients.items():
    print(f"Feature: {feature}, Coefficient: {coeff}")


# plot histograms or other plots to visualize the distribution
# of the percentage errors
plt.hist(percentage_error * 100, bins=10)  # Drop NaNs before plotting
plt.title("Distribution of Percentage Error")
plt.xlabel("Percentage Error")
plt.ylabel("Frequency")
plt.savefig(path_to_plots + "%_err-Dist.png", dpi=300, bbox_inches='tight')
plt.show()


#################    Spatial Distribution of Percentage Error #######
dataset['percentage_error'] = percentage_error * 100
# Define the size and axis for the plot
fig, ax = plt.subplots(figsize=(40, 20))
wa_state.boundary.plot(ax=ax, linewidth=2)
wa_counties.boundary.plot(ax=ax, linewidth=1, edgecolor="black")
wa_counties.apply(lambda x: ax.annotate(
    text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', fontsize=20, color='black'), axis=1)
dataset.plot(column='percentage_error', legend=False, ax=ax, markersize=500, alpha=0.7)
plt.title("Spatial Distribution of Percentage Error", fontsize=32)
# Adjust the font size of tick labels for longitude and latitude
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)

# Create the legend based on the plotted data's colormap
norm = plt.Normalize(vmin=dataset['percentage_error'].min(
), vmax=dataset['percentage_error'].max())
cbar = plt.colorbar(mappable=plt.cm.ScalarMappable(
    norm=norm, cmap='viridis'), ax=ax, shrink=0.5, pad=0.005)
cbar.ax.tick_params(labelsize=20)

plt.savefig(path_to_plots + "%_err_map.png", dpi=300, bbox_inches='tight')
plt.show()


# Scatter plot of y and y_pred
plt.figure(figsize=(8, 8))
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red')  # line of equality
plt.title("Observed vs Predicted Values", fontsize=16)
plt.xlabel("Actual Total C", fontsize=16)
plt.ylabel("Predicted Predicted Total C", fontsize=16)
plt.grid(True)
plt.savefig(path_to_plots + "y_ypred.png", dpi=300)
plt.show()

# -

dry_df


# +
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

dataset = dry_df.loc[:, "NDVI_first":"Total_C (g/cm2)"].copy()
dataset.drop(columns=["WDVI_first", "WDVI_second", "WDVI_third", "Irrigation"], inplace=True)


# Set the terciles to use for separating the data
bottom_tercile = dataset[y_var].quantile(1/3)
top_tercile = dataset[y_var].quantile(2/3)

# Create a new column in the DataFrame to indicate whether each row is in the top, middle, or bottom tercile
dataset['tercile'] = pd.cut(dataset[y_var], bins=[dataset[y_var].min(
), bottom_tercile, top_tercile, dataset[y_var].max()], labels=['bottom', 'middle', 'top'], include_lowest=True)

# filter for just top and bottom tercils
topBottom_df = dataset.loc[dataset['tercile'] != 'middle'].copy()
topBottom_df['tercile'] = topBottom_df['tercile'].cat.remove_unused_categories()
#############


# Split the data into dependent and independent variables
X_terciles = topBottom_df.drop(
    columns=['tercile', 'geometry', 'Total_C (g/cm2)'])
y_terciles = topBottom_df['tercile']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_terciles, y_terciles, test_size=0.5, random_state=42)

# Initialize the classifier
classifier = LogisticRegression(max_iter=1000)

# Perform cross-validation
cv_scores = cross_val_score(classifier, X_train, y_train, cv=3)
print("Cross-Validation Scores for each fold:", cv_scores)
print("Average Cross-Validation Score:", np.mean(cv_scores))

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Generate a contingency table
contingency_table = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print(contingency_table)

# [Your code for printing feature importance]

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['bottom', 'top'])

# Plotting using matplotlib and seaborn
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted Total C')
ax.set_ylabel('Actual Total C')
ax.set_title('Confusion Matrix of bottom and top terciles')
ax.xaxis.set_ticklabels(['bottom', 'top'])
ax.yaxis.set_ticklabels(['bottom', 'top'])
plt.savefig(path_to_plots + "CM_bottom_top_test.png", dpi=300)
plt.show()
# -


df['Total_C (g/cm2)']

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
y_var = 'Total_C (g/cm2)'

# Set the terciles to use for separating the data
bottom_tercile = np.percentile(df[y_var], 33.33)
top_tercile = np.percentile(df[y_var], 66.66)

# Subset the DataFrame to include only top and bottom tercile rows
df_terciles = df[(df[y_var] <= bottom_tercile) |
                 (df[y_var] >= top_tercile)].copy()

# Create a new column for the target variable ('high' or 'low') based on tercile membership
df_terciles['target'] = np.where(
    df_terciles[y_var] >= top_tercile, 'high', 'low')

# Select only the X variables of interest
# Replace with the actual X variable names
X_terciles = df_terciles[['NDVI_first', 'tvi_first',
       'savi_first', 'MSI_first', 'GNDVI_first', 'GRVI_first', 'LSWI_first',
       'MSAVI2_first', 'WDVI_first', 'BI_first', 'BI2_first', 'RI_first',
       'CI_first', 'B1_first', 'B2_first', 'B3_first', 'B4_first', 'B8_first',
       'B11_first', 'B12_first', 'NDVI_second', 'tvi_second', 'savi_second',
       'MSI_second', 'GNDVI_second', 'GRVI_second', 'LSWI_second',
       'MSAVI2_second', 'WDVI_second', 'BI_second', 'BI2_second', 'RI_second',
       'CI_second', 'B1_second', 'B2_second', 'B3_second', 'B4_second',
       'B8_second', 'B11_second', 'B12_second']]
y_terciles = df_terciles['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_terciles, y_terciles, test_size=0.5, random_state=42)

# Initialize the classifier
classifier = RandomForestClassifier()

# Perform cross-validation
cv_scores = cross_val_score(classifier, X_train, y_train, cv=3)

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
confusion_df = pd.DataFrame(conf_matrix, index=['Actual low', 'Actual high'], columns=['Predicted low', 'Predicted high'])

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Print the test score
print("Test Score:", test_score)

# +
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load your data into a Pandas DataFrame
df = df_second[selected_cols]

# Set the name of your y-variable
y_var = 'Total_C_g/cm2'

# Set the terciles to use for separating the data
bottom_tercile = np.percentile(df[y_var], 33.33)
top_tercile = np.percentile(df[y_var], 66.66)

# Subset the DataFrame to include only top and bottom tercile rows
df_terciles = df[(df[y_var] <= bottom_tercile) |
                 (df[y_var] >= top_tercile)].copy()

# Create a new column for the target variable ('high' or 'low') based on tercile membership
df_terciles['target'] = np.where(
    df_terciles[y_var] >= top_tercile, 'high', 'low')

# Select only the X variables of interest
# Replace with the actual X variable names
X_terciles = df_terciles[['NDVI_second', 'tvi_second',
                          'savi_second', 'MSI_second', 'GNDVI_second', 'GRVI_second', 'LSWI_second',
                          'MSAVI2_second', 'BI_second', 'BI2_second', 'RI_second',
                          'CI_second', 'B1_second', 'B2_second', 'B3_second',
                          'B4_second', 'B8_second', 'B11_second', 'B12_second']]
y_terciles = df_terciles['target']

# from sklearn.model_selection import train_test_split

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X_terciles, y_terciles, test_size=0.25, random_state=42)

from sklearn.linear_model import LogisticRegression

# Initialize the classifier
classifier = LogisticRegression()

# Perform cross-validation
cv_scores = cross_val_score(classifier, X_terciles, y_terciles, cv=5)

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Average Cross-Validation Score:", np.mean(cv_scores))

# Train the classifier on the entire data
classifier.fit(X_terciles, y_terciles)

# Make predictions on the testing data
y_pred = classifier.predict(X_terciles)

# Generate a contingency table
contingency_table = pd.crosstab(y_terciles, y_pred, rownames=['Actual'], colnames=['Predicted'])

print(contingency_table)

# -

df1 = df.loc[~df.SampleID.duplicated()]
df1.loc[df1.SampleID.isin(averaged_C.SampleID), 'Total_C_g/cm2'] = averaged_C['Total_C_g/cm2'].values
df1.loc[df1.SampleID.isin(averaged_C.SampleID), 'Total_C_g/cm2']

df1.columns

# +
# Normalize band values
from sklearn.preprocessing import StandardScaler
# assuming df is your pandas dataframe
scaler = StandardScaler()

# select the columns you want to normalize
cols_to_normalize = ['NDVI_first', 'tvi_first',
       'savi_first', 'MSI_first', 'GNDVI_first', 'GRVI_first', 'LSWI_first',
       'MSAVI2_first', 'WDVI_first', 'BI_first', 'BI2_first', 'RI_first',
       'CI_first', 'B1_first', 'B2_first', 'B3_first', 'B4_first', 'B8_first',
       'B11_first', 'B12_first', 'NDVI_second', 'tvi_second', 'savi_second',
       'MSI_second', 'GNDVI_second', 'GRVI_second', 'LSWI_second',
       'MSAVI2_second', 'WDVI_second', 'BI_second', 'BI2_second', 'RI_second',
       'CI_second', 'B1_second', 'B2_second', 'B3_second', 'B4_second',
       'B8_second', 'B11_second', 'B12_second']

# fit the scaler on the selected columns
scaler.fit(df[cols_to_normalize])

# transform the selected columns to have zero mean and unit variance
df[cols_to_normalize] = scaler.transform(df[cols_to_normalize])

# -

df1.iloc[:, 8:]

# +
import matplotlib.pyplot as plt

# Increase the font size of the labels
plt.rcParams.update({'font.size': 12})

# Increase the resolution of the plot
plt.figure(figsize=(12, 8), dpi=300)

# Plot the density distribution of column 'Total_C_g/cm2'
df1['Total_C_g/cm2'].plot(kind='density')

# Set x-axis label
plt.xlabel('Total C (g/cm$^2$)', fontsize=14)

# Mark actual values on the curve
min_value = df1['Total_C_g/cm2'].min()
max_value = df1['Total_C_g/cm2'].max()

# Plotting the actual values on the curve
plt.axvline(x=min_value, color='red', linestyle='--', label='Min')
plt.axvline(x=max_value, color='blue', linestyle='--', label='Max')

# Display legend
plt.legend(fontsize=12)

# Show the plot
plt.show()

# -

df_first['Total_C_g/cm2'].describe()

df1.columns

# +
selected_cols = ['NDVI_first', 'tvi_first',
       'savi_first', 'MSI_first', 'GNDVI_first', 'GRVI_first', 'LSWI_first',
       'MSAVI2_first', 'WDVI_first', 'BI_first', 'BI2_first', 'RI_first',
       'CI_first', 'B1_first', 'B2_first', 'B3_first', 'B4_first', 'B8_first',
       'B11_first', 'B12_first', 'NDVI_second', 'tvi_second', 'savi_second',
       'MSI_second', 'GNDVI_second', 'GRVI_second', 'LSWI_second',
       'MSAVI2_second', 'WDVI_second', 'BI_second', 'BI2_second', 'RI_second',
       'CI_second', 'B1_second', 'B2_second', 'B3_second', 'B4_second',
       'B8_second', 'B11_second', 'B12_second', 'Total_C_g/cm2']

df = df1[selected_cols]
df
# -

df.nunique()[df.nunique() == 1].index[0]



# +
import pandas as pd
import numpy as np
import seaborn as sns

# Load your data into a Pandas DataFrame
df = df1[selected_cols].copy()

# Drop columns with just one value
# df.drop(columns= df.nunique()[df.nunique() == 1].index[0], inplace=True )

# Set the name of your y-variable
y_var = 'Total_C_g/cm2'

# Set the terciles to use for separating the data
bottom_tercile = np.percentile(df[y_var], 33.33)
top_tercile = np.percentile(df[y_var], 66.66)

# Create a new column in the DataFrame to indicate whether each row is in the top, middle, or bottom tercile
df['tercile'] = pd.cut(df[y_var], bins=[df[y_var].min(
), bottom_tercile, top_tercile, df[y_var].max()], labels=['bottom', 'middle', 'top'])

# Loop through each x-variable and create a density distribution plot for the top, middle, and bottom terciles
for x_var in df.columns.drop([y_var, 'tercile']):
    g = sns.FacetGrid(df[df['tercile'] != 'middle'], hue='tercile', height=4, aspect=1.2)
    g.map(sns.kdeplot, x_var, shade=True)
    g.add_legend()

# -

df1.loc[df['tercile'] == 'top']['Total_C_g/cm2'].describe()

# +
Y = df1['Total_C_g/cm2']
X = df1[['NDVI_first', 'tvi_first',
       'savi_first', 'MSI_first', 'GNDVI_first', 'GRVI_first', 'LSWI_first',
       'MSAVI2_first', 'WDVI_first', 'BI_first', 'BI2_first', 'RI_first',
       'CI_first', 'B1_first', 'B2_first', 'B3_first', 'B4_first', 'B8_first',
       'B11_first', 'B12_first', 'NDVI_second', 'tvi_second', 'savi_second',
       'MSI_second', 'GNDVI_second', 'GRVI_second', 'LSWI_second',
       'MSAVI2_second', 'WDVI_second', 'BI_second', 'BI2_second', 'RI_second',
       'CI_second', 'B1_second', 'B2_second', 'B3_second', 'B4_second',
       'B8_second', 'B11_second', 'B12_second']]
# X = df_second[['B12_second']]

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
print(model.summary())

# +
import seaborn as sns

# calculate the correlation matrix
corr_matrix = df[['NDVI_first', 'tvi_first',
       'savi_first', 'MSI_first', 'GNDVI_first', 'GRVI_first', 'LSWI_first',
       'MSAVI2_first', 'WDVI_first', 'BI_first', 'BI2_first', 'RI_first',
       'CI_first', 'B1_first', 'B2_first', 'B3_first', 'B4_first', 'B8_first',
       'B11_first', 'B12_first', 'NDVI_second', 'tvi_second', 'savi_second',
       'MSI_second', 'GNDVI_second', 'GRVI_second', 'LSWI_second',
       'MSAVI2_second', 'WDVI_second', 'BI_second', 'BI2_second', 'RI_second',
       'CI_second', 'B1_second', 'B2_second', 'B3_second', 'B4_second',
       'B8_second', 'B11_second', 'B12_second']].corr()


# plot the correlation matrix as a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)

# show the plot
plt.show()

# -

selected_cols = ['NDVI_first', 'tvi_first',
       'savi_first', 'MSI_first', 'GNDVI_first', 'GRVI_first', 'LSWI_first',
       'MSAVI2_first', 'WDVI_first', 'BI_first', 'BI2_first', 'RI_first',
       'CI_first', 'B1_first', 'B2_first', 'B3_first', 'B4_first', 'B8_first',
       'B11_first', 'B12_first', 'NDVI_second', 'tvi_second', 'savi_second',
       'MSI_second', 'GNDVI_second', 'GRVI_second', 'LSWI_second',
       'MSAVI2_second', 'WDVI_second', 'BI_second', 'BI2_second', 'RI_second',
       'CI_second', 'B1_second', 'B2_second', 'B3_second', 'B4_second',
       'B8_second', 'B11_second', 'B12_second', 'Total_C_g/cm2']

df = df1[selected_cols]
df.reset_index(inplace=True)

# +
import pandas as pd
import numpy as np
import seaborn as sns

# Load your data into a Pandas DataFrame
df = df1[selected_cols]

# # Drop columns with just one value
# df.drop(columns= df.nunique()[df.nunique() == 1].index[0], inplace=True )

# Set the name of your y-variable
y_var = 'Total_C_g/cm2'

# Set the terciles to use for separating the data
bottom_tercile = np.percentile(df[y_var], 33.33)
top_tercile = np.percentile(df[y_var], 66.66)

# Create a new column in the DataFrame to indicate whether each row is in the top, middle, or bottom tercile
df['tercile'] = pd.cut(df[y_var], bins=[df[y_var].min(
), bottom_tercile, top_tercile, df[y_var].max()], labels=['bottom', 'middle', 'top'])

# Loop through each x-variable and create a density distribution plot for the top, middle, and bottom terciles
for x_var in df.columns.drop([y_var, 'tercile']):
    g = sns.FacetGrid(df[df['tercile'] != 'middle'], hue='tercile', height=4, aspect=1.2)
    g.map(sns.kdeplot, x_var, shade=True)
    g.add_legend()

# +
Y = df_second['Total_C_g/cm2']
X = df_second[['NDVI_second', 'tvi_second',
               'savi_second', 'MSI_second', 'GNDVI_second', 'GRVI_second', 'LSWI_second',
               'MSAVI2_second', 'BI_second', 'BI2_second', 'RI_second',
               'CI_second',
               'B2_second', 'B3_second', 'B4_second', 'B8_second', 'B11_second',
               'B12_second']]

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
print(model.summary())

# +
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load your data into a Pandas DataFrame
df = df1[selected_cols]

# Set the name of your y-variable
y_var = 'Total_C_g/cm2'

# Set the terciles to use for separating the data
bottom_tercile = np.percentile(df[y_var], 33.33)
top_tercile = np.percentile(df[y_var], 66.66)

# Subset the DataFrame to include only top and bottom tercile rows
df_terciles = df[(df[y_var] <= bottom_tercile) |
                 (df[y_var] >= top_tercile)].copy()

# Create a new column for the target variable ('high' or 'low') based on tercile membership
df_terciles['target'] = np.where(
    df_terciles[y_var] >= top_tercile, 'high', 'low')

# Select only the X variables of interest
# Replace with the actual X variable names
X_terciles = df_terciles[['NDVI_first', 'tvi_first',
       'savi_first', 'MSI_first', 'GNDVI_first', 'GRVI_first', 'LSWI_first',
       'MSAVI2_first', 'WDVI_first', 'BI_first', 'BI2_first', 'RI_first',
       'CI_first', 'B1_first', 'B2_first', 'B3_first', 'B4_first', 'B8_first',
       'B11_first', 'B12_first', 'NDVI_second', 'tvi_second', 'savi_second',
       'MSI_second', 'GNDVI_second', 'GRVI_second', 'LSWI_second',
       'MSAVI2_second', 'WDVI_second', 'BI_second', 'BI2_second', 'RI_second',
       'CI_second', 'B1_second', 'B2_second', 'B3_second', 'B4_second',
       'B8_second', 'B11_second', 'B12_second']]
y_terciles = df_terciles['target']

# from sklearn.model_selection import train_test_split

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X_terciles, y_terciles, test_size=0.25, random_state=42)

from sklearn.linear_model import LogisticRegression

# Initialize the classifier
classifier = LogisticRegression()

# Perform cross-validation
cv_scores = cross_val_score(classifier, X_terciles, y_terciles, cv=5)

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Average Cross-Validation Score:", np.mean(cv_scores))

# Train the classifier on the entire data
classifier.fit(X_terciles, y_terciles)

# Make predictions on the testing data
y_pred = classifier.predict(X_terciles)

# Generate a contingency table
contingency_table = pd.crosstab(y_terciles, y_pred, rownames=['Actual'], colnames=['Predicted'])

print(contingency_table)

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load your data into a Pandas DataFrame
df = df1[selected_cols]

# Set the name of your y-variable
y_var = 'Total_C_g/cm2'

# Set the terciles to use for separating the data
bottom_tercile = np.percentile(df[y_var], 33.33)
top_tercile = np.percentile(df[y_var], 66.66)

# Subset the DataFrame to include only top and bottom tercile rows
df_terciles = df[(df[y_var] <= bottom_tercile) |
                 (df[y_var] >= top_tercile)].copy()

# Create a new column for the target variable ('high' or 'low') based on tercile membership
df_terciles['target'] = np.where(
    df_terciles[y_var] >= top_tercile, 'high', 'low')

# Select only the X variables of interest
# Replace with the actual X variable names
X_terciles = df_terciles[['NDVI_first', 'tvi_first',
       'savi_first', 'MSI_first', 'GNDVI_first', 'GRVI_first', 'LSWI_first',
       'MSAVI2_first', 'WDVI_first', 'BI_first', 'BI2_first', 'RI_first',
       'CI_first', 'B1_first', 'B2_first', 'B3_first', 'B4_first', 'B8_first',
       'B11_first', 'B12_first', 'NDVI_second', 'tvi_second', 'savi_second',
       'MSI_second', 'GNDVI_second', 'GRVI_second', 'LSWI_second',
       'MSAVI2_second', 'WDVI_second', 'BI_second', 'BI2_second', 'RI_second',
       'CI_second', 'B1_second', 'B2_second', 'B3_second', 'B4_second',
       'B8_second', 'B11_second', 'B12_second']]
y_terciles = df_terciles['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_terciles, y_terciles, test_size=0.25, random_state=42)

# Initialize the classifier
classifier = RandomForestClassifier()

# Perform cross-validation
cv_scores = cross_val_score(classifier, X_train, y_train, cv=3)

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
confusion_df = pd.DataFrame(conf_matrix, index=['Actual low', 'Actual high'], columns=['Predicted low', 'Predicted high'])

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Print the test score
print("Test Score:", test_score)

# +
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load your data into a Pandas DataFrame
df = df_second[selected_cols]

# Set the name of your y-variable
y_var = 'Total_C_g/cm2'

# Set the terciles to use for separating the data
bottom_tercile = np.percentile(df[y_var], 33.33)
top_tercile = np.percentile(df[y_var], 66.66)

# Subset the DataFrame to include only top and bottom tercile rows
df_terciles = df[(df[y_var] <= bottom_tercile) |
                 (df[y_var] >= top_tercile)].copy()

# Create a new column for the target variable ('high' or 'low') based on tercile membership
df_terciles['target'] = np.where(
    df_terciles[y_var] >= top_tercile, 'high', 'low')

# Select only the X variables of interest
# Replace with the actual X variable names
X_terciles = df_terciles[['NDVI_second', 'tvi_second',
                          'savi_second', 'MSI_second', 'GNDVI_second', 'GRVI_second', 'LSWI_second',
                          'MSAVI2_second', 'BI_second', 'BI2_second', 'RI_second',
                          'CI_second', 'B1_second', 'B2_second', 'B3_second',
                          'B4_second', 'B8_second', 'B11_second', 'B12_second']]
y_terciles = df_terciles['target']

# from sklearn.model_selection import train_test_split

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X_terciles, y_terciles, test_size=0.25, random_state=42)

from sklearn.linear_model import LogisticRegression

# Initialize the classifier
classifier = LogisticRegression()

# Perform cross-validation
cv_scores = cross_val_score(classifier, X_terciles, y_terciles, cv=5)

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Average Cross-Validation Score:", np.mean(cv_scores))

# Train the classifier on the entire data
classifier.fit(X_terciles, y_terciles)

# Make predictions on the testing data
y_pred = classifier.predict(X_terciles)

# Generate a contingency table
contingency_table = pd.crosstab(y_terciles, y_pred, rownames=['Actual'], colnames=['Predicted'])

print(contingency_table)
