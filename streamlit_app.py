import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import io


column_names = [
    "CRIM",  # per capita crime rate by town
    "ZN",  # proportion of residential land zoned for lots over 25,000 sq.ft.
    "INDUS",  # proportion of non-retail business acres per town
    "CHAS",
    # Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    "NOX",  # nitric oxides concentration (parts per 10 million)
    "RM",  # average number of rooms per dwelling
    "AGE",  # proportion of owner-occupied units built prior to 1940
    "DIS",  # weighted distances to five Boston employment centres
    "RAD",  # index of accessibility to radial highways
    "TAX",  # full-value property-tax rate per $10,000
    "PTRATIO",  # pupil-teacher ratio by town
    "B",  # 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    "LSTAT",  # % lower status of the population
    "MEDV"  # Median value of owner-occupied homes in $1000's
]


lines = []
with open('data/boston.txt', 'r') as f:
    all_lines = f.read().strip().split('\n')

data_lines = all_lines[22:]
merged_lines = []
for i in range(0, len(data_lines), 2):
    line = data_lines[i].strip() + " " + data_lines[i + 1].strip()
    merged_lines.append(line)

data = pd.read_csv(io.StringIO('\n'.join(merged_lines)),
                   sep=r'\s+',
                   header=None,
                   names=column_names)


st.title("Boston Housing Analysis")

st.write("""
This web application presents the Boston Housing dataset analysis performed in the Jupyter Notebook.
You can see the descriptive statistics, plots, and test a hypothesis, and also interact with the data via a REST API.
""")

base_url = "http://127.0.0.1:8000"


st.subheader("First 5 rows of the dataset")
st.dataframe(data.head())

st.subheader("Dataset shape")
st.write(data.shape)

st.subheader("Data info")
buffer = io.StringIO()
data.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)

st.subheader("Descriptive Statistics")
st.write(data.describe())

st.subheader("Check for missing values")
st.write(data.isnull().sum())
st.markdown("### no NaNs and nulls so no need to fill them")

st.markdown("### Basic statistics for selected columns")
st.write("**MEDV**")
st.write("Mean of MEDV:", data['MEDV'].mean())
st.write("Median of MEDV:", data['MEDV'].median())
st.write("Std of MEDV:", data['MEDV'].std())

st.write("**CRIM**")
st.write("Mean of CRIM:", data['CRIM'].mean())
st.write("Median of CRIM:", data['CRIM'].median())
st.write("Std of CRIM:", data['CRIM'].std())

st.write("**RM**")
st.write("Mean of RM:", data['RM'].mean())
st.write("Median of RM:", data['RM'].median())
st.write("Std of RM:", data['RM'].std())

st.markdown("### Visualizations")

fig = px.histogram(data, x="MEDV", nbins=20, title='Distribution of MEDV')
st.plotly_chart(fig)

st.markdown("### It shows how often each value of MEDV occurs in the dataset")
st.markdown("### As we can see, most houses have a MEDV of around 20-25 (in $1000)")

fig = px.scatter(data, x='RM', y='MEDV', title='MEDV vs RM')
st.plotly_chart(fig)

st.markdown("### The scatter plot shows a positive correlation between the number of rooms (RM) and the median value of homes (MEDV)")
st.markdown("### As the number of rooms increases, the median value of homes also increases")

fig = px.line(data, x=data.index, y='CRIM', title='CRIM values by index')
st.plotly_chart(fig)

st.markdown("### The line plot shows the distribution of CRIM values in the dataset")
st.markdown("### As we can see, the crime rate is mostly below 10, with a few outliers")

st.markdown("### Detailed Overview")


fig = px.box(data, x='CHAS', y='MEDV', title='Comparison of MEDV based on CHAS')
st.plotly_chart(fig)

st.markdown("### The box plot shows the distribution of MEDV based on the Charles River variable (CHAS)")
st.markdown("### Houses near the river (CHAS=1) have a higher median value compared to houses not near the river (CHAS=0)")

chas_0_medv = data.loc[data['CHAS'] == 0, 'MEDV']
chas_1_medv = data.loc[data['CHAS'] == 1, 'MEDV']
st.write("Statistics for MEDV where CHAS=0:")
st.write(chas_0_medv.describe())
st.write("Statistics for MEDV where CHAS=1:")
st.write(chas_1_medv.describe())


corr_matrix = data.corr()
fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                color_continuous_scale='RdBu_r', origin='lower',
                title='Correlation Matrix Heatmap')
st.plotly_chart(fig)

st.markdown("### The correlation matrix heatmap shows the correlation between different features in the dataset")
st.markdown("### The correlation coefficient ranges from -1 to 1 (which shows that we have some strong correlations)")


st.markdown("### Hypothesis")
st.write("""
**Hypothesis:**  
Homes located near the river (CHAS=1) with a higher than average number of rooms (RM > mean(RM)) 
and a lower than average LSTAT (lower status of population) (LSTAT < mean(LSTAT)) have a significantly higher median home value (MEDV) 
compared to the rest of the homes.
""")

mean_RM = data['RM'].mean()
mean_LSTAT = data['LSTAT'].mean()

filtered_data = data[(data['CHAS'] == 1) & (data['RM'] > mean_RM) & (data['LSTAT'] < mean_LSTAT)]
other_data = data[~((data['CHAS'] == 1) & (data['RM'] > mean_RM) & (data['LSTAT'] < mean_LSTAT))]

st.write("Filtered group stats:")
st.write(filtered_data['MEDV'].describe())
st.write("Other group stats:")
st.write(other_data['MEDV'].describe())

combined = pd.concat([
    pd.DataFrame({'Group': 'Filtered group', 'MEDV': filtered_data['MEDV']}),
    pd.DataFrame({'Group': 'Other', 'MEDV': other_data['MEDV']})
], ignore_index=True)
fig = px.box(combined, x='Group', y='MEDV', title='Comparison of MEDV for Hypothesis Testing')
st.plotly_chart(fig)


st.markdown("""
**Conclusion:**  
The filtered group shows a mean MEDV of approximately 36.44 and a median of 33.15, both of which are substantially higher 
than the mean (22.08) and median (20.9) MEDV of all other homes. Thus, the hypothesis is supported by the data.
""")

st.markdown("### Data Transformation")
data['MEDV_per_RM'] = data['MEDV'] / data['RM']
data['CRIM_distance_ratio'] = data['CRIM'] / (data['DIS'] + 0.1)
st.write(data[['CRIM', 'LSTAT', 'DIS', 'MEDV_per_RM', 'CRIM_distance_ratio']].head(10))

st.header("Filter Houses by MEDV")
min_medv = st.slider("Min MEDV", 0.0, 50.0, 0.0)
max_medv = st.slider("Max MEDV", 0.0, 50.0, 50.0)
if st.button("Get Houses"):
    response = requests.get(
        f"{base_url}/houses/?min_medv={min_medv}&max_medv={max_medv}")
    if response.status_code == 200:
        houses = pd.DataFrame(response.json())
        st.write(houses.head())
        st.write("Data shape:", houses.shape)

        fig = px.histogram(houses, x="MEDV", nbins=20,
                           title="Distribution of MEDV (filtered)")
        st.plotly_chart(fig)
    else:
        st.error("Failed to fetch data from API.")

st.header("Add a new house")
with st.form("add_house_form"):
    CRIM = st.number_input("CRIM", 0.0)
    ZN = st.number_input("ZN", 0.0)
    INDUS = st.number_input("INDUS", 0.0)
    CHAS = st.selectbox("CHAS", [0, 1])
    NOX = st.number_input("NOX", 0.0)
    RM = st.number_input("RM", 1.0)
    AGE = st.number_input("AGE", 0.0)
    DIS = st.number_input("DIS", 0.0)
    RAD = st.number_input("RAD", 1, step=1)
    TAX = st.number_input("TAX", 0.0)
    PTRATIO = st.number_input("PTRATIO", 0.0)
    B = st.number_input("B", 0.0)
    LSTAT = st.number_input("LSTAT", 0.0)
    MEDV = st.number_input("MEDV", 0.0)
    submitted = st.form_submit_button("Add House")
    if submitted:
        new_house = {
            "CRIM": CRIM,
            "ZN": ZN,
            "INDUS": INDUS,
            "CHAS": CHAS,
            "NOX": NOX,
            "RM": RM,
            "AGE": AGE,
            "DIS": DIS,
            "RAD": RAD,
            "TAX": TAX,
            "PTRATIO": PTRATIO,
            "B": B,
            "LSTAT": LSTAT,
            "MEDV": MEDV
        }
        response = requests.post(f"{base_url}/houses/", json=new_house)
        if response.status_code == 200:
            st.success("House added successfully!")
        else:
            st.error("Failed to add house.")
