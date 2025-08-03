import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="GPT-4o Infrastructure Calculator",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🔧 GPT-4o Infrastructure Calculator")
st.markdown("""
Calculate the GPU, power, and cost requirements to serve GPT-4o to billions of users.
Adjust the parameters in the sidebar to see how assumptions affect the infrastructure requirements.
""")

# Sidebar for parameters
st.sidebar.header("📊 Model Parameters")
st.sidebar.markdown("Adjust the assumptions below to see real-time calculations:")

# Model parameters
gpt4o_throughput = st.sidebar.number_input(
    "GPT-4o Throughput (tokens/second per H100)",
    min_value=1,
    max_value=1000,
    value=109,
    help="Assumed throughput of GPT-4o per H100 GPU"
)

total_users = st.sidebar.number_input(
    "Total Users (billions)",
    min_value=0.1,
    max_value=20.0,
    value=8.0,
    step=0.1,
    help="Total number of users to serve"
) * 1e9  # Convert to actual number

queries_per_user_per_day = st.sidebar.number_input(
    "Queries per User per Day",
    min_value=1,
    max_value=100,
    value=10,
    help="Average number of queries each user sends per day"
)

tokens_per_query = st.sidebar.number_input(
    "Tokens Generated per Query",
    min_value=1,
    max_value=5000,
    value=750,
    help="Average number of tokens generated per query"
)

# Hardware parameters
st.sidebar.header("⚡ Hardware Parameters")

gpu_power_consumption = st.sidebar.number_input(
    "H100 Power Consumption (Watts)",
    min_value=100,
    max_value=1000,
    value=700,
    help="Power consumption per H100 GPU in watts"
)

# Cost parameters
st.sidebar.header("💰 Cost Parameters")

electricity_cost_per_kwh = st.sidebar.number_input(
    "Electricity Cost ($/kWh)",
    min_value=0.01,
    max_value=1.0,
    value=0.15,
    step=0.01,
    help="Cost of electricity per kilowatt-hour"
)

# Additional parameters for sensitivity analysis
st.sidebar.header("🔄 Load Distribution")

peak_load_multiplier = st.sidebar.slider(
    "Peak Load Multiplier",
    min_value=1.0,
    max_value=5.0,
    value=2.0,
    step=0.1,
    help="Multiplier for peak traffic (2.0 means 2x average load)"
)

# Calculations
def calculate_infrastructure_requirements():
    # Basic calculations
    daily_queries = total_users * queries_per_user_per_day
    queries_per_second = daily_queries / (24 * 60 * 60)  # Convert to QPS
    tokens_per_second = queries_per_second * tokens_per_query
    
    # GPU requirements
    gpus_needed_average = tokens_per_second / gpt4o_throughput
    gpus_needed_peak = gpus_needed_average * peak_load_multiplier
    
    # Power calculations
    power_consumption_kw_average = gpus_needed_average * (gpu_power_consumption / 1000)
    power_consumption_kw_peak = gpus_needed_peak * (gpu_power_consumption / 1000)
    
    # Energy consumption (kWh per day)
    energy_per_day_kwh = power_consumption_kw_average * 24
    
    # Cost calculations
    cost_per_day = energy_per_day_kwh * electricity_cost_per_kwh
    cost_per_week = cost_per_day * 7
    cost_per_month = cost_per_day * 30
    cost_per_year = cost_per_day * 365
    
    return {
        'daily_queries': daily_queries,
        'queries_per_second': queries_per_second,
        'tokens_per_second': tokens_per_second,
        'gpus_needed_average': gpus_needed_average,
        'gpus_needed_peak': gpus_needed_peak,
        'power_consumption_kw_average': power_consumption_kw_average,
        'power_consumption_kw_peak': power_consumption_kw_peak,
        'energy_per_day_kwh': energy_per_day_kwh,
        'cost_per_day': cost_per_day,
        'cost_per_week': cost_per_week,
        'cost_per_month': cost_per_month,
        'cost_per_year': cost_per_year
    }

# Calculate results
results = calculate_infrastructure_requirements()

# Helper function to format large numbers
def format_large_number(num):
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"

def format_currency(amount):
    if amount >= 1e9:
        return f"${amount/1e9:.2f}B"
    elif amount >= 1e6:
        return f"${amount/1e6:.2f}M"
    elif amount >= 1e3:
        return f"${amount/1e3:.2f}K"
    else:
        return f"${amount:.2f}"

# Display results
st.header("📈 Infrastructure Requirements")

# Create columns for metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Daily Queries",
        format_large_number(results['daily_queries']),
        delta=f"{results['queries_per_second']:.0f} QPS"
    )

with col2:
    st.metric(
        "Tokens per Second",
        format_large_number(results['tokens_per_second']),
        delta="Total throughput required"
    )

with col3:
    st.metric(
        "GPUs Required (Average)",
        f"{results['gpus_needed_average']:,.0f}",
        delta=f"Peak: {results['gpus_needed_peak']:,.0f}"
    )

with col4:
    st.metric(
        "Power Consumption",
        f"{results['power_consumption_kw_average']/1e6:.2f} GW",
        delta=f"Peak: {results['power_consumption_kw_peak']/1e6:.2f} GW"
    )

# Cost breakdown
st.header("💰 Cost Breakdown")

cost_col1, cost_col2, cost_col3, cost_col4 = st.columns(4)

with cost_col1:
    st.metric("Daily Cost", format_currency(results['cost_per_day']))

with cost_col2:
    st.metric("Weekly Cost", format_currency(results['cost_per_week']))

with cost_col3:
    st.metric("Monthly Cost", format_currency(results['cost_per_month']))

with cost_col4:
    st.metric("Annual Cost", format_currency(results['cost_per_year']))

# Visualizations
st.header("📊 Data Visualizations")

# Create two columns for visualizations
viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    # Cost breakdown over time periods
    cost_periods = ['Day', 'Week', 'Month', 'Year']
    cost_values = [
        results['cost_per_day'],
        results['cost_per_week'],
        results['cost_per_month'],
        results['cost_per_year']
    ]
    
    fig1 = px.bar(
        x=cost_periods,
        y=cost_values,
        title="Cost Breakdown by Time Period",
        labels={'x': 'Time Period', 'y': 'Cost ($)'},
        color=cost_values,
        color_continuous_scale='Blues'
    )
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with viz_col2:
    # Resource allocation pie chart
    total_power_mw = results['power_consumption_kw_average'] / 1000
    
    # Simulate different data center regions for demonstration
    regions = ['US West', 'US East', 'Europe', 'Asia Pacific']
    power_distribution = [0.35, 0.25, 0.25, 0.15]  # Proportional distribution
    power_by_region = [total_power_mw * dist for dist in power_distribution]
    
    fig2 = px.pie(
        values=power_by_region,
        names=regions,
        title="Power Distribution by Region"
    )
    st.plotly_chart(fig2, use_container_width=True)

# Sensitivity analysis
st.header("🔍 Sensitivity Analysis")

# Parameter ranges for sensitivity analysis
param_ranges = {
    'Users (Billions)': np.linspace(1, 15, 15),
    'Queries/User/Day': np.linspace(5, 20, 16),
    'Tokens/Query': np.linspace(300, 1200, 16),
    'Electricity Cost ($/kWh)': np.linspace(0.05, 0.30, 16)
}

# Calculate sensitivity for each parameter
sensitivity_data = []

for param_name, param_values in param_ranges.items():
    costs = []
    for value in param_values:
        # Create temporary parameters
        temp_users = total_users if param_name != 'Users (Billions)' else value * 1e9
        temp_queries = queries_per_user_per_day if param_name != 'Queries/User/Day' else value
        temp_tokens = tokens_per_query if param_name != 'Tokens/Query' else value
        temp_elec_cost = electricity_cost_per_kwh if param_name != 'Electricity Cost ($/kWh)' else value
        
        # Recalculate
        temp_daily_queries = temp_users * temp_queries
        temp_qps = temp_daily_queries / (24 * 60 * 60)
        temp_tps = temp_qps * temp_tokens
        temp_gpus = temp_tps / gpt4o_throughput
        temp_power_kw = temp_gpus * (gpu_power_consumption / 1000)
        temp_energy_day = temp_power_kw * 24
        temp_cost_day = temp_energy_day * temp_elec_cost
        
        costs.append(temp_cost_day)
        sensitivity_data.append({
            'Parameter': param_name,
            'Value': value,
            'Daily_Cost': temp_cost_day
        })

# Create sensitivity plot
sensitivity_df = pd.DataFrame(sensitivity_data)

fig3 = px.line(
    sensitivity_df,
    x='Value',
    y='Daily_Cost',
    color='Parameter',
    title="Parameter Sensitivity Analysis - Daily Cost Impact",
    labels={'Value': 'Parameter Value', 'Daily_Cost': 'Daily Cost ($)'}
)
fig3.update_layout(hovermode='x unified')
st.plotly_chart(fig3, use_container_width=True)

# Scaling analysis
st.header("📈 Scaling Analysis")

# Show how requirements scale with user base
user_scaling = np.logspace(8, 10, 20)  # From 100M to 10B users
scaling_data = []

for users in user_scaling:
    temp_daily_queries = users * queries_per_user_per_day
    temp_qps = temp_daily_queries / (24 * 60 * 60)
    temp_tps = temp_qps * tokens_per_query
    temp_gpus = temp_tps / gpt4o_throughput
    temp_power_gw = temp_gpus * (gpu_power_consumption / 1000) / 1e6  # Convert to GW
    
    scaling_data.append({
        'Users_Billions': users / 1e9,
        'GPUs_Millions': temp_gpus / 1e6,
        'Power_GW': temp_power_gw
    })

scaling_df = pd.DataFrame(scaling_data)

# Create subplot with secondary y-axis
fig4 = make_subplots(
    specs=[[{"secondary_y": True}]],
    subplot_titles=("Infrastructure Scaling with User Base",)
)

fig4.add_trace(
    go.Scatter(
        x=scaling_df['Users_Billions'],
        y=scaling_df['GPUs_Millions'],
        name="GPUs (Millions)",
        line=dict(color='blue')
    ),
    secondary_y=False,
)

fig4.add_trace(
    go.Scatter(
        x=scaling_df['Users_Billions'],
        y=scaling_df['Power_GW'],
        name="Power (GW)",
        line=dict(color='red')
    ),
    secondary_y=True,
)

fig4.update_xaxes(title_text="Users (Billions)")
fig4.update_yaxes(title_text="GPUs (Millions)", secondary_y=False)
fig4.update_yaxes(title_text="Power Consumption (GW)", secondary_y=True)

st.plotly_chart(fig4, use_container_width=True)

# Export functionality
st.header("📥 Export Results")

# Create export data
export_data = {
    'Parameter': [
        'Total Users', 'Queries per User per Day', 'Tokens per Query',
        'GPT-4o Throughput (tokens/s per H100)', 'H100 Power Consumption (W)',
        'Electricity Cost ($/kWh)', 'Peak Load Multiplier'
    ],
    'Value': [
        f"{total_users/1e9:.1f}B", queries_per_user_per_day, tokens_per_query,
        gpt4o_throughput, gpu_power_consumption, electricity_cost_per_kwh, peak_load_multiplier
    ]
}

results_data = {
    'Metric': [
        'Daily Queries', 'Queries per Second', 'Tokens per Second',
        'GPUs Required (Average)', 'GPUs Required (Peak)',
        'Power Consumption Average (GW)', 'Power Consumption Peak (GW)',
        'Energy per Day (GWh)', 'Daily Cost', 'Weekly Cost', 'Monthly Cost', 'Annual Cost'
    ],
    'Value': [
        f"{results['daily_queries']:,.0f}",
        f"{results['queries_per_second']:,.0f}",
        f"{results['tokens_per_second']:,.0f}",
        f"{results['gpus_needed_average']:,.0f}",
        f"{results['gpus_needed_peak']:,.0f}",
        f"{results['power_consumption_kw_average']/1e6:.3f}",
        f"{results['power_consumption_kw_peak']/1e6:.3f}",
        f"{results['energy_per_day_kwh']/1e6:.1f}",
        format_currency(results['cost_per_day']),
        format_currency(results['cost_per_week']),
        format_currency(results['cost_per_month']),
        format_currency(results['cost_per_year'])
    ]
}

export_col1, export_col2 = st.columns(2)

with export_col1:
    st.subheader("Parameters Used")
    params_df = pd.DataFrame(export_data)
    st.dataframe(params_df, use_container_width=True)

with export_col2:
    st.subheader("Calculated Results")
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)

# Download buttons
col_download1, col_download2 = st.columns(2)

with col_download1:
    params_csv = params_df.to_csv(index=False)
    st.download_button(
        label="Download Parameters CSV",
        data=params_csv,
        file_name="gpt4o_parameters.csv",
        mime="text/csv"
    )

with col_download2:
    results_csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results CSV",
        data=results_csv,
        file_name="gpt4o_results.csv",
        mime="text/csv"
    )

# Footer with additional information
st.markdown("---")
st.markdown("""
**Note**: This calculator provides estimates based on the assumptions entered. Actual infrastructure requirements 
may vary based on factors such as model optimization, hardware efficiency, cooling requirements, redundancy needs, 
and real-world usage patterns. Consider consulting with infrastructure specialists for production deployments.
""")
