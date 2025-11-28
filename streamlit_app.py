"""
Monte Carlo Portfolio Optimizer - Web Dashboard
Simple, beautiful, interactive web interface using Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path

# Import your modules
from src.data_fetcher import DataFetcher
from src.monte_carlo_engine import MonteCarloSimulator
from src.risk_calculator import RiskCalculator

# Page config
st.set_page_config(
    page_title="Monte Carlo Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    .success-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Monte Carlo Portfolio Optimizer")
st.markdown("**Advanced portfolio optimization with real-time analytics**")

# Sidebar - User Inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Stock Selection
    st.subheader("1. Select Assets")
    
    preset = st.selectbox(
        "Choose Preset",
        ["Custom", "Tech Giants", "Tech + Finance", "FAANG", "Dividend Stocks"]
    )
    
    if preset == "Tech Giants":
        default_tickers = "AAPL, GOOGL, MSFT, AMZN"
    elif preset == "Tech + Finance":
        default_tickers = "AAPL, GOOGL, JPM, GS, BAC"
    elif preset == "FAANG":
        default_tickers = "META, AAPL, AMZN, NFLX, GOOGL"
    elif preset == "Dividend Stocks":
        default_tickers = "JNJ, PG, KO, PEP, MCD"
    else:
        default_tickers = "AAPL, GOOGL, MSFT, AMZN"
    
    tickers_input = st.text_input(
        "Stock Tickers (comma-separated)",
        value=default_tickers
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    
    # Date Range
    st.subheader("2. Date Range")
    
    date_preset = st.selectbox(
        "Time Period",
        ["1 Year", "2 Years", "3 Years", "5 Years", "Custom"]
    )
    
    if date_preset == "1 Year":
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    elif date_preset == "2 Years":
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    elif date_preset == "3 Years":
        start_date = (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d')
    elif date_preset == "5 Years":
        start_date = (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d')
    else:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=730)
        ).strftime('%Y-%m-%d')
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Simulation Parameters
    st.subheader("3. Simulation Settings")
    
    initial_investment = st.number_input(
        "Initial Investment ($)",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=10000
    )
    
    num_simulations = st.slider(
        "Number of Simulations",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000
    )
    
    risk_free_rate = st.slider(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.1
    ) / 100
    
    # Run Button
    st.markdown("---")
    run_analysis = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

# Main Content
if run_analysis:
    with st.spinner("🔄 Fetching market data..."):
        try:
            # Fetch Data
            fetcher = DataFetcher(tickers, start_date, end_date, risk_free_rate=risk_free_rate)
            market_data = fetcher.fetch_historical_data(use_cache=True)
            stats = fetcher.calculate_statistics()
            
            st.success("✅ Data fetched successfully!")
            
            # Display Data Summary
            st.header("📈 Market Data Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Trading Days", market_data.trading_days)
            with col2:
                st.metric("Start Date", market_data.start_date)
            with col3:
                st.metric("End Date", market_data.end_date)
            
            # Statistics Table
            st.subheader("Asset Performance")
            
            perf_df = pd.DataFrame({
                'Ticker': stats.annualized_returns.index,
                'Annual Return': stats.annualized_returns.values * 100,
                'Volatility': stats.volatility.values * 100,
                'Sharpe Ratio': stats.sharpe_ratios.values
            })
            
            # Color code Sharpe ratios
            def color_sharpe(val):
                if val > 1:
                    return 'background-color: #d4edda'
                elif val > 0.5:
                    return 'background-color: #fff3cd'
                else:
                    return 'background-color: #f8d7da'
            
            st.dataframe(
                perf_df.style.format({
                    'Annual Return': '{:.2f}%',
                    'Volatility': '{:.2f}%',
                    'Sharpe Ratio': '{:.3f}'
                }).applymap(color_sharpe, subset=['Sharpe Ratio']),
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            st.stop()
    
    # Run Monte Carlo Simulation
    with st.spinner("🎲 Running Monte Carlo simulation..."):
        try:
            simulator = MonteCarloSimulator(
                mean_returns=stats.mean_returns,
                cov_matrix=stats.cov_matrix,
                risk_free_rate=risk_free_rate,
                num_simulations=num_simulations,
                random_seed=42
            )
            
            results = simulator.generate_random_portfolios()
            
            st.success(f"✅ Completed {num_simulations:,} simulations in {results.execution_time:.2f}s")
            
        except Exception as e:
            st.error(f"❌ Simulation error: {str(e)}")
            st.stop()
    
    # Display Optimal Portfolio
    st.header("🏆 Optimal Portfolio")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Expected Return",
            f"{results.optimal_portfolio.expected_return*100:.2f}%",
            delta=f"{(results.optimal_portfolio.expected_return - risk_free_rate)*100:.2f}% above risk-free"
        )
    
    with col2:
        st.metric(
            "Volatility",
            f"{results.optimal_portfolio.volatility*100:.2f}%"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{results.optimal_portfolio.sharpe_ratio:.3f}",
            delta="Higher is better"
        )
    
    with col4:
        expected_value = initial_investment * (1 + results.optimal_portfolio.expected_return)
        st.metric(
            "Expected Value (1Y)",
            f"${expected_value:,.0f}",
            delta=f"${expected_value - initial_investment:,.0f}"
        )
    
    # Asset Allocation
    st.subheader("📊 Asset Allocation")
    
    allocation_df = pd.DataFrame({
        'Ticker': tickers,
        'Weight': results.optimal_portfolio.weights * 100,
        'Dollar Amount': results.optimal_portfolio.weights * initial_investment
    }).sort_values('Weight', ascending=False)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Pie chart
        fig_pie = px.pie(
            allocation_df,
            values='Weight',
            names='Ticker',
            title='Portfolio Weights',
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart
        fig_bar = px.bar(
            allocation_df,
            x='Ticker',
            y='Weight',
            title='Asset Weights (%)',
            color='Weight',
            color_continuous_scale='Blues'
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Efficient Frontier
    st.header("🎯 Efficient Frontier")
    
    fig_frontier = go.Figure()
    
    # All portfolios
    fig_frontier.add_trace(go.Scatter(
        x=results.all_volatilities * 100,
        y=results.all_returns * 100,
        mode='markers',
        marker=dict(
            size=3,
            color=results.all_sharpe_ratios,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio")
        ),
        name='Simulated Portfolios',
        text=[f'Return: {r*100:.1f}%<br>Vol: {v*100:.1f}%<br>Sharpe: {s:.2f}'
              for r, v, s in zip(results.all_returns, results.all_volatilities, results.all_sharpe_ratios)],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Optimal portfolio
    fig_frontier.add_trace(go.Scatter(
        x=[results.optimal_portfolio.volatility * 100],
        y=[results.optimal_portfolio.expected_return * 100],
        mode='markers',
        marker=dict(size=20, color='gold', symbol='star', line=dict(color='red', width=2)),
        name='Optimal Portfolio',
        hovertemplate=f'<b>Optimal</b><br>Return: {results.optimal_portfolio.expected_return*100:.2f}%<br>Vol: {results.optimal_portfolio.volatility*100:.2f}%<br>Sharpe: {results.optimal_portfolio.sharpe_ratio:.3f}<extra></extra>'
    ))
    
    # Min volatility
    fig_frontier.add_trace(go.Scatter(
        x=[results.min_volatility_portfolio.volatility * 100],
        y=[results.min_volatility_portfolio.expected_return * 100],
        mode='markers',
        marker=dict(size=15, color='lightgreen', symbol='diamond', line=dict(color='darkgreen', width=2)),
        name='Min Volatility',
        hovertemplate=f'<b>Min Vol</b><br>Return: {results.min_volatility_portfolio.expected_return*100:.2f}%<br>Vol: {results.min_volatility_portfolio.volatility*100:.2f}%<extra></extra>'
    ))
    
    fig_frontier.update_layout(
        title=f'Efficient Frontier ({num_simulations:,} Simulations)',
        xaxis_title='Volatility (%)',
        yaxis_title='Expected Return (%)',
        hovermode='closest',
        height=600
    )
    
    st.plotly_chart(fig_frontier, use_container_width=True)
    
    # Risk Analysis
    with st.spinner("📉 Calculating risk metrics..."):
        portfolio_path = simulator.simulate_portfolio_paths(
            weights=results.optimal_portfolio.weights,
            initial_investment=initial_investment,
            time_horizon=252,
            num_paths=1000
        )
        
        portfolio_returns = (market_data.returns * results.optimal_portfolio.weights).sum(axis=1)
        
        risk_calc = RiskCalculator(
            portfolio_returns=portfolio_returns,
            portfolio_paths=portfolio_path.paths,
            initial_investment=initial_investment,
            risk_free_rate=risk_free_rate
        )
        
        risk_report = risk_calc.generate_comprehensive_report(
            weights=results.optimal_portfolio.weights,
            mean_returns=stats.mean_returns,
            cov_matrix=stats.cov_matrix
        )
    
    st.header("⚠️ Risk Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "VaR (95%)",
            f"${abs(risk_report.var_95.var_value):,.0f}",
            delta=f"{abs(risk_report.var_95.var_percentage)*100:.2f}%",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "CVaR (95%)",
            f"${abs(risk_report.cvar_95.cvar_value):,.0f}",
            delta=f"{abs(risk_report.cvar_95.cvar_percentage)*100:.2f}%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Max Drawdown",
            f"{abs(risk_report.drawdown_analysis.max_drawdown)*100:.2f}%",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "Probability of Loss",
            f"{portfolio_path.statistics['probability_of_loss']*100:.1f}%",
            delta_color="inverse"
        )
    
    # Portfolio Simulation Paths
    st.subheader("📈 Portfolio Value Simulation")
    
    fig_paths = go.Figure()
    
    # Sample paths (show 100)
    sample_indices = np.random.choice(portfolio_path.paths.shape[1], 100, replace=False)
    for idx in sample_indices:
        fig_paths.add_trace(go.Scatter(
            x=portfolio_path.timestamps,
            y=portfolio_path.paths[:, idx],
            mode='lines',
            line=dict(width=0.5, color='lightblue'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Percentiles
    fig_paths.add_trace(go.Scatter(
        x=portfolio_path.timestamps,
        y=portfolio_path.percentiles[50],
        mode='lines',
        line=dict(color='darkblue', width=3),
        name='Median (50th)'
    ))
    
    fig_paths.add_trace(go.Scatter(
        x=portfolio_path.timestamps,
        y=portfolio_path.percentiles[95],
        mode='lines',
        line=dict(color='green', width=2, dash='dash'),
        name='95th Percentile'
    ))
    
    fig_paths.add_trace(go.Scatter(
        x=portfolio_path.timestamps,
        y=portfolio_path.percentiles[5],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='5th Percentile'
    ))
    
    # Initial investment line
    fig_paths.add_hline(
        y=initial_investment,
        line_dash="dot",
        line_color="black",
        annotation_text="Initial Investment"
    )
    
    fig_paths.update_layout(
        title='1,000 Simulated Portfolio Paths (1 Year)',
        xaxis_title='Trading Days',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig_paths, use_container_width=True)
    
    # Stress Test Results
    st.subheader("🚨 Stress Test Scenarios")
    
    stress_df = pd.DataFrame([
        {
            'Scenario': st.scenario_name,
            'Loss ($)': abs(st.portfolio_loss),
            'Loss (%)': abs(st.portfolio_loss_percentage) * 100,
            'New Value ($)': st.new_portfolio_value
        }
        for st in risk_report.stress_test_results
    ]).sort_values('Loss ($)', ascending=False)
    
    fig_stress = px.bar(
        stress_df,
        x='Scenario',
        y='Loss (%)',
        color='Loss (%)',
        color_continuous_scale='Reds',
        title='Portfolio Loss Under Historical Crisis Scenarios'
    )
    
    fig_stress.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_stress, use_container_width=True)
    
    st.dataframe(
        stress_df.style.format({
            'Loss ($)': '${:,.0f}',
            'Loss (%)': '{:.2f}%',
            'New Value ($)': '${:,.0f}'
        }),
        use_container_width=True
    )
    
    # Download Results
    st.header("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export allocation
        allocation_csv = allocation_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Allocation (CSV)",
            data=allocation_csv,
            file_name=f"portfolio_allocation_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export risk metrics
        risk_data = {
            'VaR_95': risk_report.var_95.var_value,
            'CVaR_95': risk_report.cvar_95.cvar_value,
            'Max_Drawdown': risk_report.drawdown_analysis.max_drawdown,
            'Sharpe_Ratio': results.optimal_portfolio.sharpe_ratio,
            'Expected_Return': results.optimal_portfolio.expected_return,
            'Volatility': results.optimal_portfolio.volatility
        }
        risk_json = json.dumps(risk_data, indent=2)
        st.download_button(
            label="📉 Download Risk Report (JSON)",
            data=risk_json,
            file_name=f"risk_report_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    with col3:
        # Export performance data
        perf_csv = perf_df.to_csv(index=False)
        st.download_button(
            label="📈 Download Performance (CSV)",
            data=perf_csv,
            file_name=f"asset_performance_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

else:
    # Landing page when no analysis run
    st.info("👈 Configure your portfolio settings in the sidebar and click **Run Analysis** to start!")
    
    # Feature showcase
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎲 Monte Carlo Simulation")
        st.markdown("Run thousands of portfolio simulations to find optimal allocation")
    
    with col2:
        st.markdown("### 📊 Risk Analytics")
        st.markdown("Calculate VaR, CVaR, drawdowns, and stress test scenarios")
    
    with col3:
        st.markdown("### 📈 Interactive Charts")
        st.markdown("Visualize efficient frontier and portfolio paths in real-time")
    
    st.markdown("---")
    
    # Sample results preview
    st.markdown("### 🎯 What You'll Get:")
    st.markdown("""
    - **Optimal Portfolio Allocation** - Best weights for maximum Sharpe ratio
    - **Efficient Frontier Visualization** - 10,000+ simulated portfolios
    - **Risk Metrics** - VaR, CVaR, Maximum Drawdown, Sortino Ratio
    - **Stress Testing** - Performance under 6 historical crisis scenarios
    - **Portfolio Paths** - 1,000 simulated trajectories over 1 year
    - **Downloadable Reports** - CSV and JSON exports
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Python, Streamlit, and Modern Portfolio Theory | © 2025")
