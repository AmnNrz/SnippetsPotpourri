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
#     display_name: gis_env
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Read data
path_to_data = "/home/amnnrz/GoogleDrive - msaminnorouzi/PhD/Projects/DSFAS/Data/"
df = pd.read_csv(path_to_data + "Carbon&satellite_data_joined_v1.csv")
df

df.DepthSampl.value_counts()

dff = df
depth_0_12_idx = dff.DepthSampl == "0_12"
print(depth_0_12_idx.value_counts())
dff.loc[depth_0_12_idx, "rrrr"] = dff.loc[depth_0_12_idx,
                                          "TotalC"]/100 * 12 * 1 * 2.54 * df.loc[depth_0_12_idx, "BD_g_cm3"]
dff.rrrr.isna().value_counts()

# +


# df_first = pd.read_csv(
#     "/Users/aminnorouzi/Library/CloudStorage/GoogleDrive-msaminnorouzi@gmail.com/My Drive/PhD/Projects/DSFAS/df_first.csv", index_col = 0)
# df_second = pd.read_csv(
#     "/Users/aminnorouzi/Library/CloudStorage/GoogleDrive-msaminnorouzi@gmail.com/My Drive/PhD/Projects/DSFAS/df_second.csv", index_col= 0)
# df_first = df_first.loc[df_first["DepthSampl"] == "0_12"].copy()
# df_second = df_second.loc[df_second["DepthSampl"] == "0_12"].copy()

# "total_c_%" /100 * height * A * 2.54 (inch to cm) * BD
df["Total_C_g/cm2"] = df["TotalC"]/100 * 12 * 1 * 2.54 * df["BD_g_cm3"]
df
# -

df.DepthSampl.value_counts()

dup_df

# +
dup_df = df.loc[df.SampleID.duplicated(keep=False)]
dup_df

averaged_C = pd.DataFrame([])
averaged_C['SampleID'] = dup_df.SampleID.unique()
for id in dup_df.SampleID.unique():
    averaged_C.loc[averaged_C["SampleID"] == id, "Total_C_g/cm2"] = np.mean(dup_df.loc[dup_df["SampleID"] == id, "Total_C_g/cm2"])
# -

averaged_C['Total_C_g/cm2']

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

