# --- CRITICAL FIX FOR STR SHADOWING ---
# We store a reference to the built-in str function immediately to prevent any
# external library (like ChatterBot) or code from shadowing it globally.
# This variable is ONLY used for robust error handling and when ChatterBot calls str() internally.
__py_str_func = str 
# ---------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import os
import sqlite3

# --- CORRECTED PAGE CONFIGURATION (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="LoopySmart Prototype",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CHATTERBOT AND DATA ANALYSIS IMPORTS ---
# Prophet is used for forecasting.
from prophet import Prophet 
# Pytrends is used for external trend data.
from pytrends.request import TrendReq 
# ChatterBot components
from chatterbot.logic import LogicAdapter 
from chatterbot.conversation import Statement 
from chatterbot import ChatBot
from chatterbot import languages
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer
    
# --- ACTUAL DATA ANALYSIS ADAPTER CLASS (Merged from data_analysis_adapter.py) ---
class DataAnalysisLogicAdapter(LogicAdapter):

    def __init__(self, chatbot, **kwargs):
        super().__init__(chatbot, **kwargs)
        
        # Initialize Pytrends
        self.pytrends = TrendReq(hl='en-US', tz=360) 
        
        # Mapping sectors to their KPI data files and relevant keywords
        self.sector_map = {
            'food': {'file': 'kpi_food_beverage.csv', 'keywords': ['food', 'beverage', 'restaurant', 'drink']},
            'fashion': {'file': 'kpi_fashion.csv', 'keywords': ['fashion', 'clothing', 'apparel', 'shoes']},
            'lifestyle': {'file': 'kpi_lifestyle_beauty.csv', 'keywords': ['lifestyle', 'beauty', 'cosmetic', 'home', 'fitness']},
            'tech': {'file': 'kpi_tech_gadget.csv', 'keywords': ['tech', 'gadget', 'electronic', 'phone', 'hardware']},
        }
        
        # Load all KPI data into a single dictionary
        self.data_store = self._load_all_data()

        # Keywords for TRENDING Queries (External Pytrends)
        self.external_keywords = ['trending', 'buzz', 'market trend', 'competitor', 'external', 'latest']
        self.kpi_keywords = ['revenue', 'margin', 'sales', 'value', 'kpi', 'how much']
        self.forecast_keywords = ['forecast', 'predict', 'future', 'trend for', 'what will be']


    @st.cache_resource 
    # FIX: Renamed 'self' to '_self' to tell Streamlit NOT to hash the instance.
    def _load_all_data(_self):
        """
        Creates mock data structures for Prophet and KPI logic.
        This function is cached to run only once during the app's lifetime.
        """
        data = {}
        # Iterate over the sector map using _self (the renamed instance reference)
        for sector in _self.sector_map: 
            # Mock a time series for Prophet (ds, y)
            mock_dates = pd.date_range(end=datetime.date.today(), periods=100)
            
            # Create slightly different mock patterns for variety
            if sector == 'tech':
                mock_values = np.linspace(50, 80, 100) + np.random.randn(100) * 5
            elif sector == 'fashion':
                mock_values = np.sin(np.arange(100) / 10) * 15 + 60 + np.random.randn(100) * 3
            else:
                mock_values = np.random.rand(100) * 30 + 100 

            # Mock KPI data (e.g., Revenue, Margin)
            mock_kpi = pd.DataFrame({
                'ds': mock_dates,
                'y': mock_values
            })
            
            # Mock a single current KPI value (Scaled for display)
            current_kpi = mock_values[-1] * 1000 
            
            data[sector] = {
                'kpi_df': mock_kpi,
                'current_kpi': current_kpi,
                'kpi_name': f"Revenue for {sector.title()}"
            }
        return data

    def _get_datetime_from_text(self, text):
        # Simplified mock date parsing
        text = text.lower()
        if 'today' in text:
            return datetime.date.today()
        if 'last week' in text:
            return datetime.date.today() - datetime.timedelta(weeks=1)
        if 'next month' in text:
            return datetime.date.today() + datetime.timedelta(days=30)
        return None

    # --- FORECAST PROCESSING METHOD (Prophet) ---
    def _process_forecast(self, text, sector_name):
        
        try:
            data = self.data_store.get(sector_name)
            if data is None:
                return Statement(f"I don't have enough data to forecast for {sector_name.title()}. Try food, fashion, lifestyle, or tech.", confidence=0.8)

            df = data['kpi_df']
            kpi_name = data['kpi_name']

            # Prophet instantiation
            m = Prophet()
            m.fit(df)

            future = m.make_future_dataframe(periods=7) # Forecast 7 days ahead
            forecast = m.predict(future)

            # Get the next forecast value
            next_forecast_date = forecast['ds'].iloc[-1].strftime('%Y-%m-%d')
            next_forecast_value = forecast['yhat'].iloc[-1]
            
            current_value = df['y'].iloc[-1]
            
            trend = "upward" if next_forecast_value > current_value else "downward"

            response_text = (
                f"**{sector_name.title()} Forecast Analysis (Prophet):**\n\n"
                f"The model predicts the **{kpi_name}** will trend **{trend}**.\n"
                f"The forecasted value for **{next_forecast_date}** is approximately **RM{next_forecast_value:,.2f}**.\n\n"
                "This forecast is based on historical time-series data."
            )
            return Statement(response_text, confidence=1.0)
        
        except Exception as e:
            # Safely convert exception to string using the built-in reference
            error_message = __py_str_func(e)
            return Statement(f"I encountered an error running the forecast for {sector_name}. Details: {error_message}", confidence=0.7)

    # --- EXTERNAL TRENDS PROCESSING METHOD (Pytrends) ---
    def _process_external_trends(self, text):
        
        try:
            sector_query = "general retail"
            for sector in self.sector_map.keys():
                if any(word in text for word in self.sector_map[sector]['keywords']):
                    sector_query = sector
                    break
            
            # Use pytrends.trending_searches for real-time global trends
            trending_df = self.pytrends.trending_searches(pn='united_states')
            
            if not trending_df.empty:
                top_topics = trending_df[0].head(5).tolist()
                
                response_text = (
                    f"**Global Consumer Engagement Trends (Google Trends):**\n\n"
                    f"For your query regarding **{sector_query.title()}**, the Top 5 trending search topics (all categories) are:\n\n"
                    f"1. **{top_topics[0]}**\n"
                    f"2. {top_topics[1]}\n"
                    f"3. {top_topics[2]}\n"
                    f"4. {top_topics[3]}\n"
                    f"5. {top_topics[4]}\n\n"
                    "**Actionable Insight:** These real-time searches indicate current consumer *interest* and *buzz*."
                )
                return Statement(response_text, confidence=1.0)
            
        except Exception as e:
            # Safely convert exception to string using the built-in reference
            error_message = __py_str_func(e)
            return Statement(f"I encountered an error fetching live Google Trends data. Please check my connection. Details: {error_message}", confidence=0.7)
        
        return Statement("I couldn't find any current trending data right now.", confidence=0.7)


    def can_process(self, statement):
        """Checks if the statement is related to data analysis, forecasting, or trends."""
        text = statement.text.lower()
        
        # Merge all data-related keywords
        all_data_keywords = self.kpi_keywords + self.forecast_keywords
        
        # Check if any data/forecast keyword is present
        data_hit = any(kw in text for kw in all_data_keywords)

        # Check if any sector keyword is present
        sector_hit = False
        for sector in self.sector_map.keys():
            # Check if any word is in the sector's keyword list
            if any(word in text for word in self.sector_map[sector]['keywords']):
                sector_hit = True
                break
        
        # External Trend Check (Pytrends)
        external_hit = any(kw in text for kw in self.external_keywords)
            
        # Processable if it's an external hit OR it's a data query AND a sector is specified
        return external_hit or (data_hit and sector_hit)

    def process(self, statement, additional_response_parameters=None):
        """Processes the statement based on the type of data request."""
        text = statement.text.lower()
        
        # 1. Identify Sector
        sector_name = None
        for sector in self.sector_map.keys():
            if any(word in text for word in self.sector_map[sector]['keywords']):
                sector_name = sector
                break
        
        # 2. Process External Trends
        if any(kw in text for kw in self.external_keywords):
            return self._process_external_trends(text)

        # 3. Process Forecast
        if any(kw in text for kw in self.forecast_keywords):
            if sector_name:
                return self._process_forecast(text, sector_name)
            else:
                return Statement("I can only run a forecast if you specify a sector like food, fashion, lifestyle, or tech. Which one?", confidence=0.9)


        # 4. Process KPI Query (Default)
        if any(kw in text for kw in self.kpi_keywords):
            if sector_name:
                data = self.data_store.get(sector_name)
                # Ensure data is not corrupted/empty
                if data and 'current_kpi' in data:
                    kpi_value = data['current_kpi']
                    kpi_name = data['kpi_name']
                    
                    # Mock simple percentage change
                    change = np.random.uniform(-5, 5)
                    change_sign = "up" if change > 0 else "down"
                    
                    response_text = (
                        f"**{sector_name.title()} KPI Summary:**\n\n"
                        f"The current **{kpi_name}** is **RM{kpi_value:,.2f}**.\n"
                        f"This figure is **{abs(change):.1f}% {change_sign}** compared to last week (Mock data).\n\n"
                        "Would you like me to run a forecast on this data?"
                    )
                    return Statement(response_text, confidence=1.0)
                else:
                    return Statement(f"I have mock data loaded for {sector_name.title()}, but the KPI value seems missing or corrupted.", confidence=0.7)

        # Fallback if no specific logic was matched but can_process returned true (should not happen)
        return Statement("I detected a data analysis query but couldn't process the specifics. Could you rephrase or specify a sector and KPI?", confidence=0.6)


# --- GLOBAL CONFIGURATION AND COLORS ---
# Standardized color palette
BABY_BLUE = "#89CFF0"
LIGHT_CORAL = "#F7A5A6"
DARK_BLUE_HEADER = "#0f4c75" # Used for a strong header contrast
BACKGROUND_COLOR = "#f8f9fa" 
CARD_BG = "#ffffff"


# Inject global CSS for a cohesive look and feel
st.markdown(
    f"""
    <style>
    /* Global Background and Typography */
    .stApp {{
        background-color: {BACKGROUND_COLOR};
        color: #333333;
        font-family: 'Inter', sans-serif;
    }}

    /* Main Tab Container - Spacing and Centering */
    .stTabs [data-baseweb="tab-list"] {{
        display: flex;
        justify-content: space-around;
        gap: 15px;
        padding-top: 10px;
        padding-bottom: 5px;
        border-bottom: 2px solid #e0e0e0;
    }}

    /* Individual Tabs */
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        color: #333333;
        border-radius: 8px 8px 0 0;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }}
    
    /* Active Tab */
    .stTabs [aria-selected="true"] {{
        color: {DARK_BLUE_HEADER} !important;
        border-bottom: 3px solid {BABY_BLUE} !important;
        background-color: {CARD_BG} !important;
    }}

    /* Standard Card Style */
    .stCard, .stAlert, .stSelectbox, .stTextInput, .stButton > button {{
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }}
    
    /* Button Styling */
    .stButton > button, .stForm > button {{ /* Added .stForm > button for compatibility */
        background-color: {BABY_BLUE};
        color: white;
        font-weight: bold;
        transition: background-color 0.3s;
        border: none; /* Ensure buttons in forms look nice too */
        padding: 10px 20px;
    }}
    .stButton > button:hover, .stForm > button:hover {{
        background-color: "#6a9ccf"; /* slightly darker blue */
    }}
    
    /* Header/Title Styling */
    .main-header {{
        background-color: {DARK_BLUE_HEADER};
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }}
    
    /* Loopy Chatbot Styling */
    .loopy-header {{
        background-image: linear-gradient(to right, {LIGHT_CORAL}, {BABY_BLUE});
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }}
    
    /* Retail Plan Card Styling */
    .plan-card {{
        background-color: {CARD_BG};
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }}
    
    /* Dropdown Styling for Explore Tab */
    .stSelectbox div[data-baseweb="select"] {{
        border-radius: 8px;
        border: 2px solid {BABY_BLUE} !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        background-color: white;
        margin-top: 5px;
        margin-bottom: 20px;
    }}
    /* Force the displayed value text to be dark black/gray */
    .stSelectbox div[data-baseweb="select"] div[data-baseweb="input"] {{
        color: #333333 !important; 
        background-color: white !important;
        padding: 10px;
    }}
    
    </style>
    """
    , unsafe_allow_html=True
)


# --- 1. HOME PAGE LOGIC ---

def home_page():
    """Displays the main dashboard content."""
    
    # Header Banner
    st.markdown(f"""
        <div class='main-header'>
            <h1 style='font-size:36px; margin-bottom:5px;'>LoopySmart Dashboard</h1>
            <p style='font-size:18px;'>Your unified hub for trends, planning, and AI support.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Welcome Back, Analyst")

    # Mock Data for Visualization
    data = {
        "KPI": ["Revenue (Food)", "Margin (Fashion)", "Customer Count (Lifestyle)", "Conversion Rate (Tech)"],
        "Value": [520000, 45, 12000, 8.5],
        "Change": ["+4.5%", "-1.2%", "+15%", "+0.5%"],
        "Color": [BABY_BLUE, LIGHT_CORAL, BABY_BLUE, LIGHT_CORAL]
    }
    df_kpi = pd.DataFrame(data)

    # Key Performance Indicators (KPIs)
    cols = st.columns(4)
    for i, row in df_kpi.iterrows():
        with cols[i]:
            st.markdown(
                f"""
                <div style="background-color:{CARD_BG}; 
                            padding:15px; 
                            border-radius:12px; 
                            border-left:5px solid {row['Color']}; 
                            box-shadow:0 4px 8px rgba(0,0,0,0.1); 
                            height:100px;">
                    <p style="font-size:14px; color:#6c757d; margin:0;">{row['KPI']}</p>
                    <h3 style="color:#333; margin:5px 0 0 0;">{row['Value']:,}</h3>
                    <p style="font-size:12px; color:{'green' if row['Change'].startswith('+') else 'red'}; margin:0;">{row['Change']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
    st.markdown("---")
    st.subheader("Forecast Visualizations")

    # Mock Data for Prophet/Forecasting (simplified visual)
    forecast_data = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'Actual': [100, 110, 150, 130, 160, 180],
        'Forecast': [105, 115, 145, 135, 170, 190]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_data['Month'], y=forecast_data['Actual'], mode='lines+markers', name='Actual Sales', line=dict(color=DARK_BLUE_HEADER)))
    fig.add_trace(go.Scatter(x=forecast_data['Month'], y=forecast_data['Forecast'], mode='lines', name='Prophet Forecast', line=dict(color=LIGHT_CORAL, dash='dash')))

    fig.update_layout(
        title='Next 6 Months Sales Forecast (Mock)',
        xaxis_title='Month',
        yaxis_title='Sales (RM)',
        plot_bgcolor=CARD_BG,
        paper_bgcolor=CARD_BG,
        font=dict(family="Inter", size=12, color="#333"),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("AI Suggestions")

    # AI Suggestion Bars
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="height:150px; 
                        padding:20px;
                        border:1px solid #ddd; 
                        border-radius:12px; 
                        background-color:{CARD_BG}; 
                        box-shadow:0 2px 6px rgba(0,0,0,0.1);">
                <h4 style="color:{LIGHT_CORAL}; margin-top:0;">Urgent Tasks</h4>
                <p style="color:#6c757d;">⚠️ Check 'Explore' tab: New competitor trend detected in F&B.</p>
                <p style="color:#6c757d;">✅ Run 'Plan' for Sunday forecast adjustments.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="height:150px; 
                        padding:20px;
                        border:1px solid #ddd; 
                        border-radius:12px; 
                        background-color:{CARD_BG}; 
                        box-shadow:0 2px 6px rgba(0,0,0,0.1);">
                <h4 style="color:{LIGHT_CORAL}; margin-top:0;">Opportunities</h4>
                <p style="color:#6c757d;">💡 Loopy suggests a new 'Tech' bundle promo based on emerging trend velocity.</p>
                <p style="color:#6c757d;">💡 High demand for 'Lifestyle' products near upcoming holiday.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
# --- 2. EXPLORE PAGE LOGIC ---

# Mock Data Generation
def generate_trend_mock_data():
    np.random.seed(42)
    months = [f"2023-{i:02d}" for i in range(1, 13)]
    
    # Matching categories from the user's latest provided list
    categories = ["F&B"] * 4 + ["Beauty"] * 4 + ["Fashion"] * 4 + ["Tech"] * 4
    
    data = {
        'Trend Item': [
            'Sustainable Packaging', 'Plant-Based Menus', 'Interactive Dining', 'Ghost Kitchens',
            'Minimalist Skincare', 'At-Home Fitness Tech', 'Personalized Fragrance', 'Microbiome Beauty',
            'Upcycled Clothing', 'Digital Try-Ons', 'Comfort Wear', 'Thrift Culture',
            'AI-Powered Assistants', 'Foldable Devices', 'Metaverse Commerce', 'NFT Collectibles'
        ],
        'Category': categories,
        'Description': [f"Description for {item}" for item in [
            'Sustainable Packaging', 'Plant-Based Menus', 'Interactive Dining', 'Ghost Kitchens',
            'Minimalist Skincare', 'At-Home Fitness Tech', 'Personalized Fragrance', 'Microbiome Beauty',
            'Upcycled Clothing', 'Digital Try-Ons', 'Comfort Wear', 'Thrift Culture',
            'AI-Powered Assistants', 'Foldable Devices', 'Metaverse Commerce', 'NFT Collectibles'
        ]],
        'Velocity': np.random.randint(5, 15, 16),
        'Engagement_Score': np.random.uniform(3.0, 5.0, 16).round(2),
        'Month': [np.random.choice(months) for _ in range(16)]
    }
    return pd.DataFrame(data)

def explore_page():
    """Displays trend analysis and market data, now using st.selectbox."""
    st.header("🌐 Explore: Market Trends & Insights")
    st.markdown(f"<p style='color:#6c757d;'>Analyze the velocity and engagement of emerging market trends across various categories.</p>", unsafe_allow_html=True)
    
    # Generate data
    trend_data = generate_trend_mock_data()
    
    # --- Category Selection (Dropdown Menu) ---
    st.subheader("Explore Latest Trends")
    st.caption("Filter by industry sector to narrow your search.")

    # Categories for the dropdown menu
    CATEGORIES = ["All", "F&B", "Beauty", "Fashion", "Tech"]
    
    # Use st.selectbox for a cleaner UI
    selected_category = st.selectbox(
        "Select Trend Category",
        CATEGORIES,
        index=0, # Default to 'All'
        placeholder="Choose an industry sector",
        label_visibility="collapsed" # Hide the default label for a cleaner look
    )

    # Filtering the data based on the dropdown selection
    if selected_category == "All":
        trend_data_filtered = trend_data.copy()
    else:
        # Filter the DataFrame where the 'Category' column matches the selection
        trend_data_filtered = trend_data[trend_data['Category'] == selected_category]
        
    st.markdown("---")


    # --- Trend Cards ---
    st.header(f"Top Trends for {selected_category}")
    st.caption(f"Displaying {len(trend_data_filtered)} trends.")

    COLUMNS_PER_ROW = 3
    cols = st.columns(COLUMNS_PER_ROW)
    col_index = 0

    # Define color map for consistency
    color_map = {
        "F&B": "#FFDDC1",      # Light Peach
        "Beauty": "#E6E6FA",   # Lavender
        "Fashion": "#D1E7DD",  # Light Green
        "Tech": "#B0E0E6",     # Powder Blue
        "All": "#F0F8FF"       # Alice Blue
    }

    for index, row in trend_data_filtered.iterrows():
        with cols[col_index]:
            card_color = color_map.get(row['Category'], CARD_BG)
            
            st.markdown(
                f"""
                <div style='background-color:{CARD_BG}; 
                            padding:15px; 
                            border-radius:12px; 
                            margin-bottom:20px;
                            border-top: 5px solid {card_color};
                            box-shadow:0 4px 8px rgba(0,0,0,0.1);
                            height: 220px; /* fixed height for alignment */
                            display: flex;
                            flex-direction: column;
                            justify-content: space-between;'>
                    <div>
                        <span style='background-color:{card_color}; color:{DARK_BLUE_HEADER}; padding:4px 8px; border-radius:5px; font-size:12px; font-weight:bold;'>{row['Category']}</span>
                        <h4 style='margin-top:10px; color:{DARK_BLUE_HEADER};'>{row['Trend Item']}</h4>
                        <p style='font-size:14px; color:#6c757d;'>{row['Description'][:70]}...</p>
                    </div>
                    <div>
                        <p style='margin:0; font-size:14px;'>📈 Velocity: <b>{row['Velocity']:.2f}</b></p>
                        <p style='margin:0; font-size:14px;'>💬 Engagement Score: <b>{row['Engagement_Score']:.1f}/10</b></p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
        col_index = (col_index + 1) % COLUMNS_PER_ROW

    st.markdown("---")

    # Trend Velocity (Bar Chart)
    st.header("Trend Velocity (Monthly)")
    st.caption("Showing the general market momentum over the last year.")

    # Aggregate data by month to show overall velocity (using the full data for context)
    velocity_by_month = trend_data.groupby('Month')['Velocity'].sum().reset_index()
    
    st.bar_chart(
        velocity_by_month.set_index('Month'),
        use_container_width=True,
        color=BABY_BLUE
    )


# --- 3. PLAN PAGE LOGIC ---

def plan_page():
    """Displays smart daily retail strategies based on context."""
    
    # Local color palette for retail cards 
    colors_slots = ["#89CFF0", "#F7A5A6", "#98FF98", "#D8BFD8", "#FFD580", "#A7C7E7", "#B0E0E6"]
    base_background = "#F2F0EF" # Using a placeholder light background color

    st.header("📊 Strategy Studio: Smart Planning")
    st.markdown(f"<p style='color:#6c757d;'>Daily operational opportunities tailored to weather, time, and events.</p>", unsafe_allow_html=True)

    today = datetime.date.today()
    day_of_week = today.strftime("%A")
    business_type = "Café"

    # Context Header
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-weight:bold; text-align:center;'>📅 Today’s Context</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    # Using the local base_background color for context cards
    col1.markdown(f"<div class='plan-card' style='background-color:{base_background}; border-left: 5px solid {BABY_BLUE};'><p style='margin:0;'><b>Day:</b> {day_of_week}</p></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='plan-card' style='background-color:{base_background}; border-left: 5px solid {LIGHT_CORAL};'><p style='margin:0;'><b>Date:</b> {today}</p></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='plan-card' style='background-color:{base_background}; border-left: 5px solid {BABY_BLUE};'><p style='margin:0;'><b>Business Type:</b> {business_type}</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Time Slot Action Plan")

    # Mock Timeline Data
    timelines = [
        {
            "time": "7:00 AM – 9:00 AM",
            "weather": "Sunny Morning",
            "temperature": "28°C",
            "humidity": "65%",
            "event": "Morning Rush Hour",
            "insights": [
                "Focus on speed and efficiency for coffee and quick breakfast items.",
                "Promote 'grab-and-go' deals on the signboard.",
                "Ensure baristas are fully staffed and focused on quality."
            ]
        },
        {
            "time": "9:00 AM – 12:00 PM",
            "weather": "Sunny Day",
            "temperature": "32°C",
            "humidity": "60%",
            "event": "Office Break Time",
            "insights": [
                "Push premium lunch options (sandwiches, wraps) and cold beverages.",
                "Offer a 'Mid-morning Power Up' combo deal.",
                "Clean up and reorganize tables for laptop users and meetings."
            ]
        },
        {
            "time": "12:00 PM – 2:00 PM",
            "weather": "Hot & Clear",
            "temperature": "35°C",
            "humidity": "55%",
            "event": "Lunch Peak",
            "insights": [
                "Maximize seating capacity and manage the queue efficiently.",
                "Prioritize pre-made salads and quick hot meals.",
                "Boost sales of chilled drinks/snacks."
            ]
        },
        {
            "time": "2:00 PM – 5:00 PM",
            "weather": "Partly Cloudy",
            "temperature": "33°C",
            "humidity": "68%",
            "event": "Afternoon Slowdown",
            "insights": [
                "Promote desserts and afternoon tea/coffee sets to increase average spend.",
                "Use this time for staff training and inventory checks.",
                "Run a social media post showcasing cozy seating and Wi-Fi."
            ]
        },
        {
            "time": "5:00 PM – 8:00 PM",
            "weather": "Rainy Evening",
            "temperature": "27°C",
            "humidity": "80%",
            "event": "After-Work Traffic",
            "insights": [
                "Focus on comfort food and hot beverages due to rain.",
                "Promote dinner menus and online delivery services heavily.",
                "Offer promo: spend RM20 = free sample for takeaway."
            ]
        },
        {
            "time": "8:00 PM – 11:00 PM",
            "weather": "Clear Night",
            "temperature": "25°C",
            "humidity": "76%",
            "event": "Late Night Crowd",
            "insights": [
                "Offer dessert deals and light evening snacks.",
                "Reduce staffing on dine-in; focus remaining staff on prep and cleaning.",
                "Promote loyalty program sign-ups with a small incentive."
            ]
        },
        {
            "time": "11:00 PM – 12:00 AM",
            "weather": "Cloudy Night",
            "temperature": "25°C",
            "humidity": "76%",
            "event": "No major event",
            "insights": [
                "Expect very low traffic at this hour.",
                "Focus on online delivery instead of dine-in.",
                "Promote midnight promo codes for loyal customers."
            ]
        }
    ]

    # Display Timelines 
    for i, slot in enumerate(timelines):
        # 1. Prepare dynamic content
        insights_html = ''.join([f"<li>{insight}</li>" for insight in slot['insights']])
        color = colors_slots[i % len(colors_slots)]
        
        # 2. Define the full HTML content string
        html_content = f"""
<div style='background-color:{color}40; 
            padding:20px; 
            border-radius:15px; 
            margin-bottom:15px; 
            color:#000000;'>
    
<div style="display:flex; justify-content:space-around; margin-bottom:15px; text-align:center;">
    <div>🌦<br><b>{slot['weather']}</b></div>
    <div>🌡<br><b>{slot['temperature']}</b></div>
    <div>💧<br><b>{slot['humidity']}</b></div>
</div>

<p>🎉 <b>Event:</b> {slot['event']}</p>

<h5 style='margin-top:20px; border-bottom: 2px solid {DARK_BLUE_HEADER}; padding-bottom: 5px;'>🎯 Actionable Insights:</h5>
<ul style='list-style-type: disc; padding-left: 20px;'>
{insights_html} 
</ul>
</div>
"""
        
        # 3. Display the content within the expander
        with st.expander(f"🕒 {slot['time']}", expanded=False):
            st.write(html_content, unsafe_allow_html=True) 


# --- 4. LOOPY PAGE LOGIC ---

# --- CHATBOT INITIALIZATION FUNCTION ---
@st.cache_resource 
def initialize_chatterbot():
    """Initializes and trains the ChatterBot instance with refined settings."""
    
    # Use a relative path for the database
    db_path = 'chatterbot_db.sqlite3'

    # Custom Conversational Data (FLATTENED for ListTrainer)
    # CRITICAL FIX: Must be a single flat list: [input1, response1, input2, response2, ...]
    custom_data_flat = [
        'Hello', 'Hi there! How can I loop in on your tasks today?',
        'What can you do?', 'I analyze KPIs for Food, Fashion, Lifestyle, and Tech, run forecasts using Prophet, and track external market trends using Google Trends.',
        'How are you?', "I'm a bot, running perfectly! How about you?",
        'Thank you', 'You are most welcome!',
        # Explicitly added simple conversational phrases that often trigger the bug
        'hi', 'Hello! Ask me about trends or a forecast.',
        'nothing', 'No problem! Just let me know when you have a question.',
    ]
    
    # 1. Initialize the bot instance
    try:
        # Use a locally scoped name for extra safety
        loopy_bot_instance = ChatBot( 
            'Loopy',
            tagger_language=languages.ENG,
            storage_adapter='chatterbot.storage.SQLStorageAdapter',
            database_uri='sqlite:///' + db_path,
            default_response_adapter='chatterbot.logic.BestMatch',
            logic_adapters=[
                # Use the string path '__main__.DataAnalysisLogicAdapter'
                {
                    'import_path': '__main__.DataAnalysisLogicAdapter',
                    'threshold': 0.70,  
                    'default_response': 'I need more information to analyze that data. Can you specify a sector or KPI?'
                },
                # 2. General conversation and Q&A (using import path string)
                {
                    'import_path': 'chatterbot.logic.MathematicalEvaluation',
                    'threshold': 0.90 
                },
                # 3. Fallback for general questions (must be last)
                {
                    'import_path': 'chatterbot.logic.BestMatch',
                    'threshold': 0.60,
                    'statement_comparison_function': 'chatterbot.comparisons.JaccardSimilarity', 
                    'response_selection_method': 'chatterbot.response_selection.get_random_response'
                }
            ],
            # CRITICAL FIX: Use read_only=False for training, then set to True after initial run
            read_only=False 
        )

        # 2. Training Data
        # Only train on the first run to utilize caching
        if not hasattr(st.session_state, 'is_trained') or not st.session_state.is_trained:
            st.info("Training Loopy... (This happens only on first run or database reset)")
            
            # Use ListTrainer for custom data
            trainer_list = ListTrainer(loopy_bot_instance)
            # Use the corrected flat list
            trainer_list.train(custom_data_flat)

            # Use Corpus Trainer for general knowledge
            trainer_corpus = ChatterBotCorpusTrainer(loopy_bot_instance)
            trainer_corpus.train(
                'chatterbot.corpus.english'
            )
            st.success("Loopy is trained and ready!")
            st.session_state.is_trained = True
            
            # After training, set to read_only=True to prevent constant retraining
            loopy_bot_instance.read_only = True


        # Ensure it is read_only when returning the cached object
        loopy_bot_instance.read_only = True
        return loopy_bot_instance

    except Exception as e:
        # Safely convert exception to string using the built-in reference
        error_message = __py_str_func(e)
        if 'str' in error_message and 'callable' in error_message:
            st.error(
                f"**CRITICAL ERROR DURING BOT SETUP (str corruption):** The 'str' object is not callable. "
                f"The safeguard (`__py_str_func`) is active, but a major library conflict occurred."
            )
        else:
            st.error(f"ChatBot Initialization Error: Could not set up the bot. Details: {error_message}")
        return None # Return None if initialization failed


def loopy_page():
    """Displays the interactive AI Chatbot interface."""
    
    st.markdown(
        f"""
        <div class='loopy-header'>
            <h1 style='color: white; margin-bottom: 5px;'>🤖 Loopy: Your AI Co-Pilot</h1>
            <p style='color: white; margin: 0;'>Ready to assist on your business planning</p>
        </div>
        """, unsafe_allow_html=True
    )
    
    global loopy_bot
    if loopy_bot is None:
        st.warning("Loopy bot is unavailable due to an initialization error. Please check the setup messages above.")
        return 

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_history.append((
            "assistant", 
            "Hello! I'm **Loopy**, your AI Co-Pilot. I can analyze KPIs across Food, Fashion, Lifestyle, and Tech. Ask me for a KPI forecast or the current market buzz!"
        ))


    try:
        # Display chat history
        for role, message in st.session_state.chat_history:
            # Use custom HTML for the chat messages for better styling
            if role == "user":
                st.markdown(f"<div style='text-align:right; margin-bottom:10px;'><span style='background-color:{BABY_BLUE}; color:white; padding:10px 15px; border-radius:15px 15px 0 15px; max-width:70%; display:inline-block;'>{message}</span></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:left; margin-bottom:10px;'><span style='background-color:{LIGHT_CORAL}40; color:#333; padding:10px 15px; border-radius:15px 15px 15px 0; max-width:70%; display:inline-block;'>{message}</span></div>", unsafe_allow_html=True)


        # Using st.form for input
        with st.form(key='chat_form', clear_on_submit=True):
            user_query = st.text_input(
                "Ask me! (e.g. 'What is the total revenue for food?' or 'What is the current market buzz?')", 
                key='user_query_input', # Unique key
                placeholder="Type your question here...",
                label_visibility="collapsed"
            )
            submit_button = st.form_submit_button(label='Send Message')
        
        # Process the input only when the form is submitted AND the text field has content
        if submit_button and user_query:
            # Add user input to history
            st.session_state.chat_history.append(("user", user_query))
            
            # Get response from bot
            with st.spinner('Loopy is analyzing data...'):
                # --- START OF CRITICAL CODE BLOCK ---
                # DEBUG LINE 1: Check execution flow start
                st.write("DEBUG: Attempting to get response from Loopy...") 
                
                chatterbot_statement = loopy_bot.get_response(user_query)
                ai_generated_text_response = chatterbot_statement.text
                
                # DEBUG LINE 2: Check execution flow end
                st.write("DEBUG: Response obtained successfully.") 
                # --- END OF CRITICAL CODE BLOCK ---
                
            # Add assistant response to history
            st.session_state.chat_history.append(("assistant", ai_generated_text_response))
            
            # Re-run the app to update the chat history visually
            st.rerun()

    except Exception as e:
        # Fallback error display for runtime exceptions
        # Use the globally safeguarded __py_str_func for robust error display here too
        st.error(f"Could not run the ChatBot application. Runtime Details: {__py_str_func(e)}")

# --- MAIN APP EXECUTION ---

# Initialize bot outside the main function so it's cached
# This must be done here, AFTER the initialization function is defined
loopy_bot = None
try:
    # This call is cached and will only run once, preventing re-initialization cost
    loopy_bot = initialize_chatterbot() 
except Exception as e:
    # Safely convert exception to string for display
    error_message = __py_str_func(e) 
    st.error(f"Failed to load cached ChatBot: {error_message}")


def main_app():
    """The main function to structure the Streamlit application using tabs."""
    
    # Tab creation and assignment
    tab_home, tab_explore, tab_plan, tab_loopy = st.tabs(["Home", "Explore", "Plan", "Loopy"])

    with tab_home:
        home_page()
    with tab_explore:
        explore_page()
    with tab_plan:
        plan_page()
    with tab_loopy:
        loopy_page()

if __name__ == "__main__":
    main_app()
