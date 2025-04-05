import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

# Initialize Spark Session
spark = SparkSession.builder.appName("EnergyConsumptionOptimization").getOrCreate()

# Load and display the data
df_pd = pd.read_csv("D:\Energy-Consumption-Optimization-\Dataset.csv")  # Replace with your actual CSV file path
df_spark = spark.createDataFrame(df_pd)

st.set_page_config(layout="wide")
st.title("🔋 Energy Consumption Optimization")
# st.markdown("This dashboard uses **Big Data Analytics** to forecast energy consumption, detect anomalies, and recommend optimal usage strategies for residential, commercial, and industrial sectors.")

# Sidebar for filtering building types
building_type = st.sidebar.selectbox("🏢 Select Building Type", df_pd['Building Type'].unique())
df_filtered = df_pd[df_pd['Building Type'] == building_type]

# Layout grid
col1, col2 = st.columns(2)

# 1. Box Plot for Energy Consumption by Building Type
with col1:
    st.subheader("📦 Energy Consumption Distribution")
    fig_box = px.box(df_pd, x='Building Type', y='Energy Consumption', color='Building Type',
                     title="Energy Consumption by Building Type")
    st.plotly_chart(fig_box, use_container_width=True)

# 2. Scatter Plot: Square Footage vs Energy Consumption
with col2:
    st.subheader("📊 Energy vs Square Footage")
    fig_scatter = px.scatter(df_filtered, x='Square Footage', y='Energy Consumption', color='Building Type',
                             title="Energy Consumption vs Square Footage")
    st.plotly_chart(fig_scatter, use_container_width=True)

# 3. Heatmap of Correlation Matrix
st.subheader("🔗 Feature Correlation Heatmap")
corr_matrix = df_filtered[['Energy Consumption', 'Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature']].corr()
fig_heatmap = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='Viridis'))
fig_heatmap.update_layout(title="Correlation Matrix", xaxis_nticks=36)
st.plotly_chart(fig_heatmap, use_container_width=True)

# 4. Bar Chart: Average Energy Consumption (Weekday vs Weekend)
st.subheader("📅 Average Energy by Day Type")
avg_energy = df_filtered.groupby('Day of Week')['Energy Consumption'].mean().reset_index()
fig_bar = px.bar(avg_energy, x='Day of Week', y='Energy Consumption', color='Day of Week',
                 title="Average Energy Consumption: Weekday vs Weekend")
st.plotly_chart(fig_bar, use_container_width=True)

# 5. Anomaly Detection using Isolation Forest
st.subheader("⚠️ Anomaly Detection")
iso_model = IsolationForest(contamination=0.1)
df_filtered['Anomaly'] = iso_model.fit_predict(df_filtered[['Energy Consumption']])
fig_anomaly = px.scatter(df_filtered, x='Average Temperature', y='Energy Consumption',
                         color=df_filtered['Anomaly'].map({1: 'Normal', -1: 'Anomaly'}),
                         title="Energy Consumption vs Temperature (Anomaly Detection)")
st.plotly_chart(fig_anomaly, use_container_width=True)

# 6. Line Chart of Temperature vs Energy Consumption (using synthetic dates)
st.subheader("🌡️ Temperature vs Energy Over Time")
df_filtered['Date'] = pd.date_range(start='2023-01-01', periods=len(df_filtered), freq='D')
fig_line = px.line(df_filtered.sort_values('Date'), x='Date', y=['Average Temperature', 'Energy Consumption'],
                   title="Temperature and Energy Consumption Trend")
st.plotly_chart(fig_line, use_container_width=True)

# 7. Energy Consumption Forecast using Prophet (using synthetic 'Date')
df_time = df_filtered.copy()
df_time['ds'] = df_filtered['Date']  # Using synthetic date
df_time['y'] = df_filtered['Energy Consumption']
model = Prophet()
model.fit(df_time[['ds', 'y']])
future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)

st.subheader(f"📈 Energy Consumption Forecast - {building_type}")
fig_forecast = px.line(forecast, x='ds', y='yhat', labels={'ds': 'Date', 'yhat': 'Forecasted Energy (kWh)'},
                       title=f"Next 7-Day Forecast for {building_type} Buildings")
st.plotly_chart(fig_forecast, use_container_width=True)

# 8. Feature-Based Prediction Using Random Forest Regressor
st.subheader("🔍 Power Usage Prediction")
features = df_filtered[['Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature']]
target = df_filtered['Energy Consumption']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(X_train, y_train)
predictions = rfr.predict(X_test)


comparison_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predictions
})

comparison_df.reset_index(drop=True, inplace=True)

fig = px.line(comparison_df, labels={'value': 'Energy Usage', 'index': 'Time'},
              title='Actual vs Predicted Power Usage Over Time')

fig.update_traces(mode='lines+markers')
st.plotly_chart(fig, use_container_width=True)


# 9. Insights and Environmental Recommendations based on Graph Analysis
st.subheader("🌍 Insights & Eco-Friendly Recommendations")

# Collecting insights
insights = []

# Box Plot Insight (general trend across types)
high_consumption_mean = df_pd['Energy Consumption'].mean()
if df_pd['Energy Consumption'].max() > 1.5 * high_consumption_mean:
    insights.append("Some sectors show significantly higher energy consumption—indicates optimization scope through efficient systems.")

# Scatter Plot Insight
if df_filtered['Square Footage'].corr(df_filtered['Energy Consumption']) > 0.6:
    insights.append("Energy consumption increases with square footage—larger buildings should prioritize insulation and energy audits.")

# Heatmap Insight
if abs(corr_matrix.loc['Energy Consumption', 'Average Temperature']) > 0.5:
    insights.append("Strong correlation between temperature and energy use—suggests using smart thermostats and climate controls.")

# Weekday vs Weekend Insight
weekday_energy = avg_energy.loc[avg_energy['Day of Week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']), 'Energy Consumption'].mean()
weekend_energy = avg_energy.loc[avg_energy['Day of Week'].isin(['Saturday', 'Sunday']), 'Energy Consumption'].mean()
if weekday_energy > weekend_energy:
    insights.append("Weekdays show higher energy use—consider scheduling some operations on weekends to reduce weekday peak loads.")

# Anomaly Detection Insight
anomaly_rate = df_filtered['Anomaly'].value_counts(normalize=True).get(-1, 0)
if anomaly_rate > 0.1:
    insights.append("Notable number of anomalies detected—recommends real-time monitoring and predictive maintenance.")

# Forecasting Insight
forecast_trend = forecast['yhat'].diff().mean()
if forecast_trend > 0:
    insights.append("Forecast predicts rising energy consumption—need to plan for renewable integration and consumption control.")

# Prediction Insight
r2_score = rfr.score(X_test, y_test)
if r2_score > 0.7:
    insights.append("Energy consumption is predictable from building features—encouraging use of AI models to recommend efficiency upgrades.")

# Displaying Insights
for insight in insights:
    st.markdown(f"◆   {insight}")



# Energy Recommendations
st.subheader("💡 Energy Efficiency Recommendations")
recommendations = {
    "Residential": [
        "Use smart thermostats to automate climate control.",
        "Upgrade to energy-efficient LED lighting.",
        "Unplug idle electronics to reduce phantom load.",
        "Utilize solar panels for daytime usage.",
        "Encourage off-peak appliance use.",
        "Conduct regular energy audits.",
        "Insulate walls and ceilings to retain temperature."
    ],
    "Commercial": [
        "Implement automated lighting controls.",
        "Optimize HVAC systems with predictive maintenance.",
        "Use motion sensors for low-traffic areas.",
        "Shift operations to off-peak hours when possible.",
        "Train employees on energy best practices.",
        "Monitor building energy use in real time.",
        "Use energy-efficient office equipment."
    ],
    "Industrial": [
        "Install smart meters and track usage patterns.",
        "Schedule high-energy operations during off-peak hours.",
        "Upgrade outdated machinery to efficient models.",
        "Improve insulation in large facilities.",
        "Use predictive analytics to detect inefficiencies.",
        "Recover and reuse waste heat.",
        "Integrate renewable energy sources."
    ]
}
for rec in recommendations[building_type]:
    st.markdown(f"- {rec}")







