# GPT-4o Infrastructure Calculator

## Overview

This is a Streamlit web application that calculates GPU, power, and cost requirements for serving GPT-4o to billions of users. The application provides an interactive dashboard where users can adjust various parameters (throughput, user count, queries per day) and see real-time calculations of infrastructure requirements. It's designed as an educational and planning tool for understanding the scale of infrastructure needed to serve large language models at global scale.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

**Frontend Framework**: Built using Streamlit, a Python-based web app framework that provides reactive UI components and automatic re-rendering when parameters change.

**Data Visualization**: Uses Plotly (both Express and Graph Objects) for creating interactive charts and visualizations. The application employs subplots for complex multi-metric dashboards.

**Application Structure**: Single-file architecture (app.py) following Streamlit's declarative programming model where the entire app reruns on user interaction.

**User Interface Design**: 
- Wide layout configuration for optimal dashboard viewing
- Sidebar-based parameter controls for easy adjustment
- Real-time calculation updates as users modify inputs
- Interactive number inputs with validation and help text

**Calculation Engine**: Built with NumPy and Pandas for numerical computations and data manipulation. The app calculates infrastructure requirements based on user-defined parameters like throughput per GPU, total users, and query frequency.

## External Dependencies

**Core Framework**: Streamlit for web application framework and UI components

**Data Processing**: 
- Pandas for data manipulation and analysis
- NumPy for numerical computations

**Visualization**: 
- Plotly Express for high-level plotting interface
- Plotly Graph Objects for detailed chart customization
- Plotly Subplots for complex multi-chart layouts

**Note**: This is a pure Python web application with no database backend or external API integrations. All calculations are performed client-side within the Streamlit session.