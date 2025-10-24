import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import joblib
import gradio as gr
import os
from openai import OpenAI
from datetime import datetime

# --- Timestamp for Logging ---
print(f"Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} PKT")

# --- GPT-4o API Integration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("OpenAI client initialized.")
except Exception as e:
    print(f"Warning: OpenAI API key not found or client initialization failed: {e}")
    print("GPT-4o features will be disabled. Please ensure the API key is set in Hugging Face Spaces secrets.")
    client = None

# --- File Paths ---
data_path = './Census_UHI_US_Urbanized_recalculated.csv'
model_filename = './uhi_summer_day_predictor_rf (1).joblib'

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')

FRIENDLY_NAMES = {
    'DelNDVI_summer': 'Summer Greenness Deficit (NDVI Diff)',
    'DEM_urb_CT_act': 'Urban Elevation (Meters)',
    'NDVI_urb_CT_act_summer': 'Urban Greenness Index (Summer NDVI)',
    'LST_rur_day_summer': 'Rural Daytime Temperature (Summer °C)',
    'Area': 'Census Tract Area (sq. km)',
    'Total_Count': 'Total Density Count (e.g., Population)',
    'Urban_Count': 'Urban Density Count (e.g., Population)',
    'UHI_summer_day': 'Summer Daytime UHI (°C)',
    'UHI_annual_day_city': 'Annual Daytime UHI (City Avg. °C)',
    'UHI_annual_night_city': 'Annual Nighttime UHI (City Avg. °C)',
    'UHI_summer_day_city': 'Summer Daytime UHI (City Avg. °C)',
    'UHI_summer_night_city': 'Summer Nighttime UHI (City Avg. °C)',
    'UHI_winter_day_city': 'Winter Daytime UHI (City Avg. °C)',
    'UHI_winter_night_city': 'Winter Nighttime UHI (City Avg. °C)',
    'DelNDVI_annual_city': 'Annual Greenness Deficit (City Avg.)',
    'DelNDVI_summer_city': 'Summer Greenness Deficit (City Avg.)',
    'DelNDVI_winter_city': 'Winter Greenness Deficit (City Avg.)',
    'DelNDVI_annual': 'Annual Greenness Deficit',
    'DelNDVI_winter': 'Winter Greenness Deficit',
    'Urban_name': 'Urbanized Area Name',
    'Census_geoid': 'Census Tract ID',
    'system:index': 'Record Index',
}

df = None
df_cleaned = pd.DataFrame()
df_ml = pd.DataFrame()

try:
    df = pd.read_csv(data_path, encoding='latin1')
except FileNotFoundError:
    raise FileNotFoundError(f"Error: '{data_path}' not found.")
except UnicodeDecodeError:
    raise UnicodeDecodeError(f"An encoding error occurred while loading the dataset.")
except Exception:
    raise Exception(f"An unexpected error occurred while loading the dataset.")

if df is not None and not df.empty:
    key_uhi_cols = [
        'UHI_annual_day', 'UHI_annual_night', 'UHI_summer_day', 'UHI_summer_night',
        'UHI_winter_day', 'UHI_winter_night', 'DelNDVI_annual', 'DelNDVI_summer', 'DelNDVI_winter',
        'DEM_urb_CT_act', 'LST_rur_day_summer', 'NDVI_urb_CT_act_summer', 'Area',
        'Urban_name', 'Census_geoid', 'Total_Count', 'Urban_Count',
        'UHI_annual_day_city', 'UHI_annual_night_city', 'UHI_summer_day_city', 'UHI_summer_night_city',
        'UHI_winter_day_city', 'UHI_winter_night_city', 'DelNDVI_annual_city', 'DelNDVI_summer_city', 'DelNDVI_winter_city'
    ]
    existing_key_uhi_cols = [col for col in key_uhi_cols if col in df.columns]
    if not existing_key_uhi_cols:
        raise ValueError("No relevant UHI/NDVI columns found in the dataset after loading.")
    
    df_cleaned = df.dropna(subset=existing_key_uhi_cols).copy()
    if df_cleaned.empty:
        raise ValueError("Dataset is empty after dropping rows with missing critical UHI/NDVI data.")

    if 'Census_geoid' in df_cleaned.columns:
        df_cleaned.drop_duplicates(subset=['Census_geoid'], inplace=True)

    columns_to_convert_to_numeric = [col for col in existing_key_uhi_cols if col not in ['Urban_name', 'Census_geoid']]
    for col in columns_to_convert_to_numeric:
        if col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            if df_cleaned[col].isnull().any():
                df_cleaned.dropna(subset=[col], inplace=True)

    model_features = [
        'DelNDVI_summer', 'DEM_urb_CT_act', 'NDVI_urb_CT_act_summer', 'LST_rur_day_summer',
        'Area', 'Total_Count', 'Urban_Count'
    ]
    target_variable = 'UHI_summer_day'
    existing_model_features = [f for f in model_features if f in df_cleaned.columns]
    if target_variable not in df_cleaned.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in the cleaned dataset.")
    if not existing_model_features:
        raise ValueError("No valid model features found in the cleaned dataset.")

    df_ml = df_cleaned[existing_model_features + [target_variable]].copy()
    df_ml.dropna(subset=existing_model_features + [target_variable], inplace=True)
    if df_ml.empty:
        raise ValueError("Error: No valid data available for ML model training/prediction after final cleaning and feature selection.")

model = None
if os.path.exists(model_filename):
    try:
        model = joblib.load(model_filename)
        print(f"Model loaded successfully. Feature names: {model.feature_names_in_ if hasattr(model, 'feature_names_in_') else 'Not available'}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

UHI_THRESHOLDS = {'q25': 2.0, 'q75': 5.0, 'median': 3.5}
UHI_MIN_VAL = -5.0
UHI_MAX_VAL = 15.0

if 'UHI_summer_day' in df_ml.columns and not df_ml['UHI_summer_day'].empty:
    try:
        UHI_THRESHOLDS['q25'] = float(df_ml['UHI_summer_day'].quantile(0.25))
        UHI_THRESHOLDS['q75'] = float(df_ml['UHI_summer_day'].quantile(0.75))
        UHI_THRESHOLDS['median'] = float(df_ml['UHI_summer_day'].median())
        UHI_MIN_VAL = float(df_ml['UHI_summer_day'].min() - 1)
        UHI_MAX_VAL = float(df_ml['UHI_summer_day'].max() + 1)
        if UHI_MIN_VAL > UHI_MAX_VAL:
            UHI_MIN_VAL = UHI_THRESHOLDS['median'] - 5
            UHI_MAX_VAL = UHI_THRESHOLDS['median'] + 5
    except Exception as e:
        print(f"Error calculating UHI thresholds: {e}")

def get_friendly_name(col_name):
    return FRIENDLY_NAMES.get(col_name, col_name)

def get_original_col_name(friendly_name):
    for k, v in FRIENDLY_NAMES.items():
        if v == friendly_name:
            return k
    return friendly_name

def get_all_urban_names():
    if not df_cleaned.empty and 'Urban_name' in df_cleaned.columns:
        return sorted(df_cleaned['Urban_name'].dropna().unique().tolist())
    return []

def get_feature_ranges():
    ranges = {}
    for feature in model_features:
        if not df_ml.empty and feature in df_ml.columns and not df_ml[feature].empty:
            min_val = float(df_ml[feature].min())
            max_val = float(df_ml[feature].max())
            if min_val == max_val:
                ranges[feature] = (min_val * 0.9, min_val * 1.1 if min_val != 0 else 0.1)
                if ranges[feature][0] == ranges[feature][1]:
                    ranges[feature] = (min_val - 0.01, min_val + 0.01)
            else:
                ranges[feature] = (min_val, max_val)
        else:
            if feature == 'DelNDVI_summer': ranges[feature] = (-1.0, 1.0)
            elif feature == 'DEM_urb_CT_act': ranges[feature] = (0.0, 1000.0)
            elif feature == 'NDVI_urb_CT_act_summer': ranges[feature] = (0.0, 1.0)
            elif feature == 'LST_rur_day_summer': ranges[feature] = (20.0, 45.0)
            elif feature == 'Area': ranges[feature] = (0.1, 500.0)
            elif feature == 'Total_Count': ranges[feature] = (100.0, 100000.0)
            elif feature == 'Urban_Count': ranges[feature] = (100.0, 100000.0)
            else: ranges[feature] = (0.0, 1.0)
    return ranges

FEATURE_RANGES = get_feature_ranges()

def classify_uhi(predicted_uhi_value):
    if not UHI_THRESHOLDS or 'q25' not in UHI_THRESHOLDS or 'q75' not in UHI_THRESHOLDS:
        return "N/A", "UHI categorization thresholds are not set or invalid. Cannot provide specific advice.", "gray"
    q25 = UHI_THRESHOLDS['q25']
    q75 = UHI_THRESHOLDS['q75']
    if isinstance(predicted_uhi_value, (int, float, np.number)):
        predicted_uhi_value = float(predicted_uhi_value)
        if predicted_uhi_value < q25:
            category = "Low"
            suggestion = "This UHI is relatively low compared to typical urban areas in the dataset. Focus on maintaining green spaces and cool materials to preserve this advantage."
            color = "#27ae60"
        elif predicted_uhi_value >= q75:
            category = "High"
            suggestion = "This UHI is significantly high, indicating potential heat stress. Prioritize aggressive mitigation strategies like extensive green infrastructure, cool pavements, and reflective surfaces."
            color = "#c0392b"
        else:
            category = "Average"
            suggestion = "This UHI is typical for urban areas in the dataset. Consider continuous greening efforts and optimizing urban design to prevent future increases and improve climate resilience."
            color = "#e67e22"
        return category, suggestion, color
    return "N/A", "Invalid UHI value provided.", "gray"

def create_uhi_gauge_plot(value, minimum, maximum, q25_threshold, q75_threshold):
    value = max(float(minimum), min(float(value), float(maximum)))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "<b>Predicted Summer Daytime UHI</b><br><span style='color: grey; font-size:0.8em'>Lower = Cooler & More Resilient</span>", 'font': {'size': 20}},
        gauge={
            'shape': "angular",
            'axis': {'range': [float(minimum), float(maximum)], 'tickwidth': 1, 'tickcolor': "#e0e0e0"},
            'bar': {'color': "#3498db", 'thickness': 0.3},
            'bgcolor': "#2c2c2c",
            'borderwidth': 0,
            'steps': [
                {'range': [float(minimum), float(q25_threshold)], 'color': "#27ae60", 'name': "Low UHI"},
                {'range': [float(q25_threshold), float(q75_threshold)], 'color': "#e67e22", 'name': "Average UHI"},
                {'range': [float(q75_threshold), float(maximum)], 'color': "#c0392b", 'name': "High UHI"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.75,
                'value': float(q75_threshold)
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#e0e0e0", 'family': "Roboto"},
        margin=dict(l=20, r=20, t=80, b=20),
        height=300
    )
    return fig

def get_gpt4o_mitigation_advice(predicted_uhi_value, category, input_params_dict):
    if client is None:
        return "GPT-4o not available. Cannot provide detailed mitigation advice."
    formatted_params = ", ".join([f"{get_friendly_name(k)}: {v:.2f}" for k, v in input_params_dict.items()])
    prompt = f"""
    You are an expert urban climate resilience advisor specializing in Urban Heat Island (UHI) mitigation.
    A user has simulated an urban scenario with the following characteristics:
    - Predicted Summer Daytime UHI: {predicted_uhi_value:.2f}°C
    - UHI Category: {category}
    - Key Scenario Parameters: {formatted_params}
    Based on this information, provide concise, actionable, and specific recommendations for mitigating or managing the UHI for this urban area.
    Focus on strategies relevant to the given parameters.
    - If the UHI is 'Low', focus on maintaining existing advantages and long-term resilience.
    - If the UHI is 'Average', suggest balanced interventions to improve conditions.
    - If the UHI is 'High', prioritize aggressive and immediate mitigation strategies.
    Start with a brief summary of the situation. Provide 3-5 bullet points of advice.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful urban climate resilience expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return completion.choices[0].message.content
    except Exception:
        return "An error occurred while fetching AI-powered advice. Please try again later or check API key."

def get_gpt4o_plot_interpretation(plot_type, plot_details):
    if client is None:
        return "AI Interpretation: GPT-4o not available. Please check API key."
    system_message = "You are an urban climate data analyst, skilled at interpreting visualizations and explaining findings concisely."
    user_prompt = f"Please provide a concise interpretation of the following plot:\n\n"
    user_prompt += f"Plot Type: {plot_type}\n"
    user_prompt += f"Details: {plot_details}\n\n"
    user_prompt += "Highlight the key insights, trends, or important observations in 2-4 sentences. Start with 'AI Interpretation:'"
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=150
        )
        return completion.choices[0].message.content
    except Exception:
        return "AI Interpretation: An error occurred while generating AI-powered plot interpretation."

def predict_uhi_scenario(*args):
    if model is None:
        return (
            create_uhi_gauge_plot(UHI_THRESHOLDS['median'], UHI_MIN_VAL, UHI_MAX_VAL, UHI_THRESHOLDS['q25'], UHI_THRESHOLDS['q75']),
            "<div class='prediction-result-box' style='border-color:gray; color:gray;'>Model Not Loaded</div>",
            "<div class='uhi-category-label' style='color:gray;'>UHI Category: N/A</div>",
            "<div class='uhi-suggestion'>The UHI prediction model is not available. Please ensure the model file is in the correct path.</div>",
            "<div class='uhi-suggestion'>AI Advice: Model not available.</div>"
        )
    input_params_dict = dict(zip(model_features, args))
    input_values = list(args)
    try:
        input_data = pd.DataFrame([input_values], columns=model_features)
        print(f"Input data shape: {input_data.shape}, types: {input_data.dtypes}")  # Debug print
    except ValueError as e:
        return (
            create_uhi_gauge_plot(UHI_THRESHOLDS['median'], UHI_MIN_VAL, UHI_MAX_VAL, UHI_THRESHOLDS['q25'], UHI_THRESHOLDS['q75']),
            "<div class='prediction-result-box' style='border-color:red; color:red;'>Input Error</div>",
            f"<div class='uhi-category-label' style='color:red;'>Input Error</div>",
            f"<div class='uhi-suggestion' style='color:red;'>Mismatch in input parameters for prediction: {e}</div>",
            "<div class='uhi-suggestion' style='color:red;'>AI Advice: Input error.</div>"
        )
    try:
        predicted_uhi = float(model.predict(input_data)[0])  # Ensure scalar value as float
        print(f"Predicted UHI: {predicted_uhi}, type: {type(predicted_uhi)}")  # Debug print
        category, base_suggestion, color = classify_uhi(predicted_uhi)
        gpt_advice = get_gpt4o_mitigation_advice(predicted_uhi, category, input_params_dict)
        gauge_fig = create_uhi_gauge_plot(predicted_uhi, UHI_MIN_VAL, UHI_MAX_VAL, UHI_THRESHOLDS['q25'], UHI_THRESHOLDS['q75'])
        return (
            gauge_fig,
            f"<div class='prediction-result-box' style='border-color:{color}; color:{color};'>{predicted_uhi:.2f} °C</div>",
            f"<div class='uhi-category-label' style='color:{color};'>UHI Category: **{category}**</div>",
            f"<div class='uhi-suggestion'>{base_suggestion}</div>",
            f"<div class='uhi-suggestion' style='border-color:{color};'><h4>AI-Powered Mitigation Advice:</h4>{gpt_advice}</div>"
        )
    except Exception as e:
        print(f"Prediction error: {e}")  # Debug print
        return (
            create_uhi_gauge_plot(UHI_THRESHOLDS['median'], UHI_MIN_VAL, UHI_MAX_VAL, UHI_THRESHOLDS['q25'], UHI_THRESHOLDS['q75']),
            "<div class='prediction-result-box' style='border-color:red; color:red;'>Prediction Error</div>",
            f"<div class='uhi-category-label' style='color:red;'>UHI Category: Error</div>",
            f"<div class='uhi-suggestion' style='color:red;'>An error occurred during prediction: {e}. Please check the model and input data.</div>",
            "<div class='uhi-suggestion' style='color:red;'>AI Advice: Prediction error.</div>"
        )

def natural_language_query(query):
    if df_cleaned.empty:
        return "The dataset is not loaded or is empty. Cannot answer queries."
    if client:
        try:
            column_info = "\n".join([f"- {k} (e.g., {v})" for k, v in FRIENDLY_NAMES.items() if k in df_cleaned.columns or v in df_cleaned.columns])
            data_summary = f"The dataset contains urban heat island (UHI) and greenness (NDVI) data for various US urbanized areas. It includes metrics like 'UHI_summer_day', 'DelNDVI_summer', 'Urban_name', and city-level averages like 'UHI_summer_day_city'."
            prompt = f"""
            You are a helpful assistant that answers questions about urban climate data, specifically Urban Heat Island (UHI) and greenness (NDVI) in US urbanized areas.
            The dataset has columns with original names (e.g., 'UHI_summer_day', 'DelNDVI_summer', 'Urban_name') and user-friendly names (e.g., 'Summer Daytime UHI (°C)', 'Summer Greenness Deficit (NDVI Diff)', 'Urbanized Area Name').
            You can also provide general explanations of UHI concepts.
            Here's a summary of the data: {data_summary}
            Please answer the following question as concisely as possible, using information from the dataset if applicable, or general knowledge about UHI if the question is conceptual. If you need to query the data, mention what kind of data you'd look for.
            Question: {query}
            """
            query_lower = query.lower()
            response = ""
            if "average" in query_lower and "uhi" in query_lower:
                if "summer day" in query_lower and 'UHI_summer_day_city' in df_cleaned.columns:
                    avg_val = float(df_cleaned['UHI_summer_day_city'].mean(numeric_only=True))
                    response = f"The average Summer Daytime UHI across all available cities is approximately {avg_val:.2f}°C."
                elif "annual day" in query_lower and 'UHI_annual_day_city' in df_cleaned.columns:
                    avg_val = float(df_cleaned['UHI_annual_day_city'].mean(numeric_only=True))
                    response = f"The average Annual Daytime UHI across all available cities is approximately {avg_val:.2f}°C."
                elif "annual night" in query_lower and 'UHI_annual_night_city' in df_cleaned.columns:
                    avg_val = float(df_cleaned['UHI_annual_night_city'].mean(numeric_only=True))
                    response = f"The average Annual Nighttime UHI across all available cities is approximately {avg_val:.2f}°C."
                else:
                    response = "Could you please specify which UHI average (e.g., 'summer day', 'annual night') you are interested in?"
            elif "top" in query_lower and "cities" in query_lower and "uhi" in query_lower:
                if 'UHI_summer_day_city' in df_cleaned.columns:
                    top_cities = df_cleaned.groupby('Urban_name')['UHI_summer_day_city'].mean(numeric_only=True).nlargest(5)
                    response = "Top 5 cities by Summer Daytime UHI:\n" + "\n".join([f"- {name}: {uhi:.2f}°C" for name, uhi in top_cities.items()])
                else:
                    response = "Cannot rank cities by UHI as 'UHI_summer_day_city' column is not available."
            elif "explain" in query_lower:
                if "uhi" in query_lower:
                    response = "The Urban Heat Island (UHI) effect refers to urban areas being significantly warmer than their surrounding rural areas. This phenomenon is primarily caused by heat absorption from dark surfaces like roads and buildings, reduced vegetation, and heat generated by human activities."
                elif "ndvi" in query_lower:
                    response = "NDVI (Normalized Difference Vegetation Index) is a satellite-derived measure of vegetation health. Higher NDVI values indicate more dense and healthy vegetation, which generally correlates with cooler surface temperatures due to evapotranspiration."
                elif "dem" in query_lower:
                    response = "DEM (Digital Elevation Model) refers to the elevation of the urban area. While not a direct UHI driver, elevation can influence local climate patterns and ventilation, indirectly affecting UHI intensity."
                elif "lst" in query_lower:
                    response = "LST (Land Surface Temperature) is the actual temperature of the Earth's surface. In UHI studies, rural LST is often used as a baseline to calculate the temperature difference that defines the UHI effect."
                else:
                    response = "Please specify what term you'd like me to explain (e.g., UHI, NDVI, DEM, LST)."
            else:
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions about urban heat island data. Provide concise answers."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=200
                )
                response = completion.choices[0].message.content
            return response
        except Exception as e:
            print(f"Natural language query error: {e}")
            return "An AI error occurred while trying to answer your question. Please ensure your API key is valid and check console for errors."
    else:
        return "AI services are not available to answer complex natural language queries. Please check your OpenAI API key."

def create_empty_plotly_fig(title="No Data Available", height=500, width=700):
    fig = go.Figure()
    fig.add_annotation(
        text=title,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color="#b0b0b0")
    )
    fig.update_layout(
        height=height, width=width,
        margin=dict(t=50, b=20, l=20, r=20),
        plot_bgcolor="#2c2c2c",
        paper_bgcolor="#2c2c2c",
        font=dict(color="#e0e0e0"),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        title_font_color="#e0e0e0"
    )
    return fig

def create_uhi_distribution_plot(urban_name_selection, selected_metric_col_friendly):
    original_metric_col = get_original_col_name(selected_metric_col_friendly)
    interpretation = ""
    if df_cleaned.empty or original_metric_col not in df_cleaned.columns or df_cleaned[original_metric_col].empty:
        interpretation = f"AI Interpretation: No data for '{selected_metric_col_friendly}' available to plot or interpret."
        return create_empty_plotly_fig(f"No data for '{selected_metric_col_friendly}' available to plot."), interpretation
    df_plot = df_cleaned.copy()
    if urban_name_selection:
        df_plot = df_plot[df_plot['Urban_name'].isin(urban_name_selection)]
        if df_plot.empty:
            interpretation = f"AI Interpretation: No data for selected urban areas for '{selected_metric_col_friendly}'."
            return create_empty_plotly_fig(f"No data for selected urban areas for '{selected_metric_col_friendly}'."), interpretation
    fig = px.box(
        df_plot,
        y=original_metric_col,
        x='Urban_name' if urban_name_selection else None,
        color='Urban_name' if urban_name_selection else None,
        points='outliers' if not urban_name_selection else False,
        title=f'{selected_metric_col_friendly} Distribution' + (f' for {", ".join(urban_name_selection)}' if urban_name_selection else ' Across All Urban Areas'),
        labels={original_metric_col: selected_metric_col_friendly, 'Urban_name': get_friendly_name('Urban_name')},
        template='plotly_dark'
    )
    fig.update_layout(height=500, margin=dict(t=50, b=20, l=20, r=20), showlegend=False)
    if client:
        stats = {}
        if not df_plot[original_metric_col].empty:
            stats['min'] = float(df_plot[original_metric_col].min())
            stats['max'] = float(df_plot[original_metric_col].max())
            stats['median'] = float(df_plot[original_metric_col].median())
            stats['q25'] = float(df_plot[original_metric_col].quantile(0.25))
            stats['q75'] = float(df_plot[original_metric_col].quantile(0.75))
        plot_details = {
            "metric": selected_metric_col_friendly,
            "urban_areas": urban_name_selection if urban_name_selection else "all urban areas",
            "statistics": stats,
            "plot_type_description": "A box or violin plot showing the distribution and spread of a chosen metric."
        }
        interpretation = get_gpt4o_plot_interpretation("Distribution Box/Violin Plot", plot_details)
    else:
        interpretation = "AI Interpretation: Not available. Please check API key."
    return fig, interpretation

def create_greenness_uhi_scatter(urban_name_selection, uhi_metric_col_friendly, greenness_metric_col_friendly):
    original_uhi_metric = get_original_col_name(uhi_metric_col_friendly)
    original_greenness_metric = get_original_col_name(greenness_metric_col_friendly)
    interpretation = ""
    if df_cleaned.empty or original_uhi_metric not in df_cleaned.columns or original_greenness_metric not in df_cleaned.columns or \
       df_cleaned[original_uhi_metric].empty or df_cleaned[original_greenness_metric].empty:
        interpretation = f"AI Interpretation: No data for '{uhi_metric_col_friendly}' or '{greenness_metric_col_friendly}' available to plot or interpret."
        return create_empty_plotly_fig(f"No data for '{uhi_metric_col_friendly}' or '{greenness_metric_col_friendly}' available to plot."), interpretation
    df_plot = df_cleaned.copy()
    if urban_name_selection:
        df_plot = df_plot[df_plot['Urban_name'].isin(urban_name_selection)]
        if df_plot.empty:
            interpretation = f"AI Interpretation: No data for selected urban areas for scatter plot."
            return create_empty_plotly_fig(f"No data for selected urban areas for scatter plot."), interpretation
    df_plot = df_plot.dropna(subset=[original_uhi_metric, original_greenness_metric])
    if df_plot.empty:
        interpretation = "AI Interpretation: No complete data points for selected metrics after dropping NaNs for scatter plot."
        return create_empty_plotly_fig("No complete data points for selected metrics after dropping NaNs."), interpretation
    fig = px.scatter(
        df_plot,
        x=original_greenness_metric,
        y=original_uhi_metric,
        color='Urban_name' if urban_name_selection else None,
        hover_name='Census_geoid',
        trendline='ols',
        title=f'{uhi_metric_col_friendly} vs. {greenness_metric_col_friendly}',
        labels={original_uhi_metric: uhi_metric_col_friendly, original_greenness_metric: greenness_metric_col_friendly},
        template='plotly_dark'
    )
    fig.update_layout(height=500, margin=dict(t=50, b=20, l=20, r=20))
    if client:
        correlation = None
        if len(df_plot) > 1:
            correlation = float(df_plot[[original_greenness_metric, original_uhi_metric]].corr().iloc[0, 1])
        plot_details = {
            "x_axis": greenness_metric_col_friendly,
            "y_axis": uhi_metric_col_friendly,
            "urban_areas": urban_name_selection if urban_name_selection else "all urban areas",
            "correlation": f"{correlation:.2f}" if correlation is not None else "N/A",
            "plot_type_description": "A scatter plot showing the relationship between two variables, with an optional Ordinary Least Squares (OLS) trendline."
        }
        interpretation = get_gpt4o_plot_interpretation("Scatter Plot (Greenness vs. UHI)", plot_details)
    else:
        interpretation = "AI Interpretation: Not available. Please check API key."
    return fig, interpretation

def create_feature_importance_plot():
    interpretation = ""
    if model is None or not hasattr(model, 'feature_importances_') or not existing_model_features:
        interpretation = "AI Interpretation: Model not loaded or feature importances not available for interpretation."
        return create_empty_plotly_fig("Model not loaded or feature importances not available.", height=400, width=500), interpretation
    importance_df = pd.DataFrame({
        'Feature': [get_friendly_name(f) for f in existing_model_features],
        'Importance': [float(x) for x in model.feature_importances_]
    }).sort_values('Importance', ascending=False)
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance for UHI Prediction',
        labels={'Importance': 'Relative Importance', 'Feature': 'Model Feature'},
        template='plotly_dark',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    fig.update_layout(height=400, margin=dict(t=50, b=20, l=20, r=20))
    if client:
        top_features_summary = importance_df.head(3).to_dict('records')
        plot_details = {
            "top_features": top_features_summary,
            "plot_type_description": "A horizontal bar chart displaying the relative importance of features in predicting UHI."
        }
        interpretation = get_gpt4o_plot_interpretation("Feature Importance Bar Chart", plot_details)
    else:
        interpretation = "AI Interpretation: Not available. Please check API key."
    return fig, interpretation

def create_seasonal_uhi_comparison(urban_name_selection):
    city_avg_cols = [
        'UHI_summer_day_city', 'UHI_winter_day_city',
        'UHI_summer_night_city', 'UHI_winter_night_city',
        'Urban_name'
    ]
    interpretation = ""
    available_city_avg_cols = [col for col in city_avg_cols if col in df_cleaned.columns]
    if 'Urban_name' not in available_city_avg_cols:
        interpretation = "AI Interpretation: Urban name column not found for seasonal comparison."
        return create_empty_plotly_fig("Urban name column not found for seasonal comparison."), interpretation
    df_plot = df_cleaned[available_city_avg_cols].drop_duplicates(subset=['Urban_name']).copy()
    if df_plot.empty:
        interpretation = "AI Interpretation: No city-level data available for seasonal comparison."
        return create_empty_plotly_fig("No city-level data available for seasonal comparison."), interpretation
    current_selection_for_title = 'Overall Average'
    data_for_interpretation = {}
    if urban_name_selection and urban_name_selection in df_plot['Urban_name'].values:
        df_plot = df_plot[df_plot['Urban_name'] == urban_name_selection]
        current_selection_for_title = urban_name_selection
        data_for_interpretation = df_plot.iloc[0].drop('Urban_name').to_dict()
    else:
        numeric_df_for_avg = df_plot.select_dtypes(include=np.number)
        if not numeric_df_for_avg.empty:
            avg_row = numeric_df_for_avg.mean(numeric_only=True).to_frame().T
            avg_row['Urban_name'] = 'Overall Average'
            df_plot = avg_row
            data_for_interpretation = avg_row.iloc[0].drop('Urban_name').to_dict()
        else:
            interpretation = "AI Interpretation: No numeric data for overall averages for seasonal comparison."
            return create_empty_plotly_fig("No numeric data for overall averages for seasonal comparison."), interpretation
    df_melted = df_plot.melt(
        id_vars=['Urban_name'],
        value_vars=[col for col in available_city_avg_cols if col != 'Urban_name'],
        var_name='UHI Metric',
        value_name='UHI Value (°C)'
    )
    if df_melted.empty:
        interpretation = "AI Interpretation: No melted data for seasonal comparison after aggregation."
        return create_empty_plotly_fig("No melted data for seasonal comparison after aggregation."), interpretation
    df_melted['UHI Metric'] = df_melted['UHI Metric'].map(get_friendly_name)
    fig = px.bar(
        df_melted,
        x='UHI Metric',
        y='UHI Value (°C)',
        color='UHI Metric',
        barmode='group',
        title=f'Seasonal & Daily UHI Comparison for: {current_selection_for_title}',
        labels={'UHI Value (°C)': 'UHI Intensity (°C)'},
        template='plotly_dark',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(height=500, margin=dict(t=50, b=20, l=20, r=20), showlegend=True)
    if client:
        plot_details = {
            "city": current_selection_for_title,
            "uhi_values": {get_friendly_name(k): float(v) for k, v in data_for_interpretation.items()},
            "plot_type_description": "A grouped bar chart comparing Urban Heat Island (UHI) values across different seasons and times of day (e.g., Summer Day, Winter Night)."
        }
        interpretation = get_gpt4o_plot_interpretation("Seasonal UHI Comparison Bar Chart", plot_details)
    else:
        interpretation = "AI Interpretation: Not available. Please check API key."
    return fig, interpretation

def create_ranked_uhi_bar_chart(uhi_metric_friendly, top_n=20, sort_ascending=False):
    original_uhi_metric = get_original_col_name(uhi_metric_friendly)
    interpretation = ""
    if df_cleaned.empty or original_uhi_metric not in df_cleaned.columns:
        interpretation = f"AI Interpretation: No data for '{uhi_metric_friendly}' available to rank urban areas or interpret."
        return create_empty_plotly_fig(f"No data for '{uhi_metric_friendly}' available to rank urban areas."), interpretation
    current_ranking_metric = original_uhi_metric
    if '_city' not in original_uhi_metric:
        potential_city_col = original_uhi_metric + '_city'
        if potential_city_col in df_cleaned.columns:
            current_ranking_metric = potential_city_col
            uhi_metric_friendly = get_friendly_name(current_ranking_metric)
    if current_ranking_metric not in df_cleaned.columns:
        interpretation = f"AI Interpretation: Required column '{current_ranking_metric}' not found for ranking or interpretation."
        return create_empty_plotly_fig(f"Required column '{current_ranking_metric}' not found for ranking."), interpretation
    df_ranked = df_cleaned.groupby('Urban_name')[current_ranking_metric].mean(numeric_only=True).reset_index()
    df_ranked.columns = ['Urban_name', 'Average UHI (°C)']
    if df_ranked.empty:
        interpretation = f"AI Interpretation: No aggregated data for '{uhi_metric_friendly}' to rank urban areas or interpret."
        return create_empty_plotly_fig(f"No aggregated data for '{uhi_metric_friendly}' to rank urban areas."), interpretation
    df_ranked = df_ranked.sort_values(by='Average UHI (°C)', ascending=sort_ascending)
    if sort_ascending:
        title_suffix = f' (Lowest {top_n})'
        df_display = df_ranked.head(top_n)
    else:
        title_suffix = f' (Highest {top_n})'
        df_display = df_ranked.tail(top_n)
        df_display = df_display.sort_values(by='Average UHI (°C)', ascending=False)
    if df_display.empty:
        interpretation = f"AI Interpretation: No data to display after ranking and selecting top/bottom {top_n} for interpretation."
        return create_empty_plotly_fig(f"No data to display after ranking and selecting top/bottom {top_n}."), interpretation
    fig = px.bar(
        df_display,
        x='Average UHI (°C)',
        y='Urban_name',
        orientation='h',
        title=f'Ranked Urban Areas by {uhi_metric_friendly}{title_suffix}',
        labels={'Urban_name': 'Urbanized Area Name', 'Average UHI (°C)': uhi_metric_friendly},
        template='plotly_dark',
        color='Average UHI (°C)',
        color_continuous_scale=px.colors.sequential.Plasma_r if sort_ascending else px.colors.sequential.Plasma
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    fig.update_layout(height=600, margin=dict(t=50, b=20, l=20, r=20), coloraxis_showscale=True)
    if client:
        ranking_type = "lowest" if sort_ascending else "highest"
        ranked_data = df_display.to_dict('records')
        plot_details = {
            "metric": uhi_metric_friendly,
            "ranking_type": ranking_type,
            "top_n": top_n,
            "ranked_areas": ranked_data,
            "plot_type_description": "A horizontal bar chart displaying urban areas ranked by a specific UHI metric."
        }
        interpretation = get_gpt4o_plot_interpretation("Ranked Urban Areas Bar Chart", plot_details)
    else:
        interpretation = "AI Interpretation: Not available. Please check API key."
    return fig, interpretation

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&family=Roboto:wght@400;700&display=swap');
body {
    font-family: 'Montserrat', sans-serif !important;
    background-color: #1a1a1a;
    color: #e0e0e0;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Roboto', sans-serif !important;
    color: #e0e0e0;
}
.gradient-header {
    background: linear-gradient(90deg, #3498db, #2ecc71);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
    font-family: 'Roboto', sans-serif !important;
    font-weight: 700;
    font-size: 2.2em;
    margin-bottom: 20px;
    display: inline-block;
    text-shadow: 2px 2px 5px rgba(0,0,0,0.5);
}
.home-banner {
    text-align: center;
    padding: 40px 20px;
    background: linear-gradient(135deg, #3498db, #2980b9);
    color: white;
    border-radius: 15px;
    margin-bottom: 40px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.5);
    position: relative;
    overflow: hidden;
}
.home-banner h1 {
    color: white;
    font-size: 3.2em;
    margin-bottom: 15px;
    letter-spacing: 1px;
}
.home-banner p {
    font-size: 1.3em;
    opacity: 0.95;
    max-width: 800px;
    margin: 0 auto;
}
.metric-card-container {
    display: flex;
    justify-content: space-around;
    flex-wrap: wrap;
    margin-top: 30px;
    gap: 25px;
}
.metric-card {
    background-color: #2c2c2c;
    border-radius: 15px;
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.35);
    padding: 30px;
    text-align: center;
    flex: 1;
    min-width: 280px;
    max-width: 350px;
    transition: transform 0.3s ease-in-out, box-shadow 0.3s ease;
    border: 1px solid #3a3a3a;
}
.metric-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 25px rgba(0, 0, 0, 0.5);
}
.metric-card .value {
    font-family: 'Roboto', sans-serif;
    font-size: 3em;
    font-weight: 700;
    color: #e67e22;
    margin-bottom: 8px;
}
.metric-card .label {
    font-size: 1.05em;
    color: #b0b0b0;
    font-weight: 600;
}
.gradio-container {
    max-width: 1500px;
    margin: auto;
    padding: 30px;
    background-color: #1a1a1a;
}
.gradio-accordion {
    border-radius: 12px !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.25) !important;
    background-color: #2c2c2c !important;
    border: none !important;
    margin-bottom: 20px;
}
.gradio-accordion h2.label {
    color: #2ecc71 !important;
    font-size: 1.3em !important;
    font-weight: 700 !important;
}
.gradio-accordion .accordion-content {
    background-color: #222222 !important;
    border-top: 1px solid #3a3a3a !important;
}
.gradio-input-field label {
    font-weight: 600 !important;
    color: #b0b0b0 !important;
    margin-bottom: 5px !important;
}
.gradio-input-field input[type="range"] {
    -webkit-appearance: none;
    width: 100%;
    height: 10px;
    border-radius: 5px;
    background: #3a3a3a;
    outline: none;
    opacity: 0.9;
    transition: opacity .2s;
}
.gradio-input-field input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #3498db;
    cursor: pointer;
    border: 2px solid #1a1a1a;
    box-shadow: 0 0 5px rgba(0,0,0,0.5);
}
.gradio-input-field .gr-input,
.gradio-input-field .gr-dropdown {
    background-color: #3a3a3a !important;
    border: 1px solid #555 !important;
    color: #e0e0e0 !important;
    border-radius: 8px !important;
    padding: 10px 15px !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
}
.gradio-input-field .gr-input:focus,
.gradio-input-field .gr-dropdown:focus {
    border-color: #3498db !important;
    box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.4) !important;
    outline: none !important;
}
.gradio-button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 12px 25px !important;
    transition: background-color 0.3s ease, transform 0.15s ease, box-shadow 0.3s ease !important;
    background-color: #2ecc71 !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}
.gradio-button:hover {
    background-color: #27ae60 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 12px rgba(0,0,0,0.4) !important;
}
.gradio-button:active {
    transform: translateY(1px) !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
}
.prediction-result-box {
    font-size: 2em;
    font-weight: 700;
    text-align: center;
    padding: 20px 25px;
    border-radius: 15px;
    margin-top: 30px;
    border-width: 3px;
    border-style: solid;
    background-color: #2c2c2c;
    box-shadow: 0 6px 15px rgba(0,0,0,0.35);
}
.uhi-category-label {
    font-size: 1.3em;
    font-weight: 700;
    margin-top: 15px;
    text-align: center;
    letter-spacing: 0.5px;
}
.uhi-suggestion {
    font-size: 1em;
    text-align: center;
    line-height: 1.6;
    padding: 15px;
    background-color: #222222;
    border-radius: 10px;
    margin-top: 20px;
    border: 1px solid #3a3a3a;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
    color: #c0c0c0;
}
.plotly-container {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
    margin-bottom: 10px;
    background-color: #2c2c2c;
    border: 1px solid #3a3a3a;
}
.plot-interpretation {
    font-size: 0.95em;
    line-height: 1.5;
    padding: 15px;
    background-color: #222222;
    border-radius: 10px;
    margin-bottom: 25px;
    border: 1px solid #3a3a3a;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
    color: #c0c0c0;
}
@media (max-width: 768px) {
    .home-banner h1 {
        font-size: 2.5em;
    }
    .home-banner p {
        font-size: 1em;
    }
    .metric-card {
        min-width: 90%;
        margin-bottom: 20px;
    }
    .gradio-container {
        padding: 15px;
    }
    .gradient-header {
        font-size: 1.8em;
    }
}
"""

with gr.Blocks(theme=gr.themes.Default(), css=custom_css) as demo:
    gr.Markdown("""
        <div class="home-banner">
            <h1>Urban Heat Island Mitigation Dashboard</h1>
            <p>An interactive dashboard for understanding Urban Heat Islands (UHI) and exploring mitigation strategies across U.S. urbanized areas.</p>
        </div>
    """)
    avg_uhi_summer_day_city = df_cleaned['UHI_summer_day_city'].mean(numeric_only=True) if 'UHI_summer_day_city' in df_cleaned.columns and not df_cleaned['UHI_summer_day_city'].empty else 0.0
    avg_uhi_annual_night_city = df_cleaned['UHI_annual_night_city'].mean(numeric_only=True) if 'UHI_annual_night_city' in df_cleaned.columns and not df_cleaned['UHI_annual_night_city'].empty else 0.0
    avg_del_ndvi_annual_city = df_cleaned['DelNDVI_annual_city'].mean(numeric_only=True) if 'DelNDVI_annual_city' in df_cleaned.columns and not df_cleaned['DelNDVI_annual_city'].empty else 0.0
    with gr.Tabs() as tabs:
        with gr.Tab(label="Home & Overview", id="home_tab"):
            gr.Markdown('<h2 class="gradient-header">Key Urban Heat Metrics (National Averages)</h2>')
            with gr.Row(elem_classes="metric-card-container"):
                with gr.Column(scale=1, elem_classes="metric-card"):
                    gr.Markdown(f"<div class='value'>{avg_uhi_summer_day_city:.2f}°C</div><div class='label'>{get_friendly_name('UHI_summer_day_city')}</div>")
                with gr.Column(scale=1, elem_classes="metric-card"):
                    gr.Markdown(f"<div class='value'>{avg_uhi_annual_night_city:.2f}°C</div><div class='label'>{get_friendly_name('UHI_annual_night_city')}</div>")
                with gr.Column(scale=1, elem_classes="metric-card"):
                    gr.Markdown(f"<div class='value'>{avg_del_ndvi_annual_city:.2f}</div><div class='label'>{get_friendly_name('DelNDVI_annual_city')}</div>")
            gr.Markdown("""
                ---
            """)
            gr.Markdown('<h2 class="gradient-header">About This Project</h2>')
            gr.Markdown('<h3 class="gradient-header">Methodology</h3>')
            gr.Markdown("""
                Our analysis employs robust data cleaning, feature engineering, and a **Random Forest Regressor** model to predict summer daytime UHI. Visualizations are powered by **Plotly**, enabling interactive exploration of trends and relationships.
            """)
            gr.Markdown('<h3 class="gradient-header">AI Integration</h3>')
            gr.Markdown("""
                The core of our AI integration lies in the UHI Predictor. This tool allows users to simulate "what-if" scenarios by adjusting key environmental and urban parameters, providing data-driven forecasts of UHI intensity. This helps in understanding the potential impact of various urban planning and mitigation strategies.
                **New!** This version also integrates **GPT-4o** to provide intelligent, context-aware mitigation advice for your predicted UHI scenarios, and offers a natural language interface to query the dataset. You'll also find AI-powered interpretations below most charts, helping you quickly grasp key insights.
            """)
            gr.Markdown('<h3 class="gradient-header">Data Source</h3>')
            gr.Markdown("""
                The data for this dashboard is derived from the `Census_UHI_US_Urbanized_recalculated` dataset, focusing on various geographic, thermal, and vegetation-related features within U.S. urbanized areas.
                ---
            """)
            gr.Markdown('<h3 class="gradient-header">How to Use:</h3>')
            gr.Markdown("""
                Navigate through the tabs above to explore different aspects of Urban Heat Islands:
                * **Data Explorer & Controls:** Select urban areas and metrics to see distribution and correlations.
                * **Insights & Trends:** Discover key drivers of UHI and compare seasonal impacts.
                * **Ranked Urban Areas:** See which urban areas have the highest or lowest UHI.
                * **UHI Predictor & Scenario Builder:** Use our AI model to predict UHI based on custom scenarios and get actionable mitigation advice.
                * **Ask the AI (New!):** Ask natural language questions about the dataset or UHI concepts.
            """)
        with gr.Tab(label="Data Explorer & Controls", id="explorer_tab"):
            gr.Markdown('<h2 class="gradient-header">Explore UHI Data & Correlations</h2>')
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown('<h4>Filter Data & Select Metrics:</h4>')
                    urban_names_choices = get_all_urban_names()
                    metric_dist_choices = [get_friendly_name(col) for col in [
                        'UHI_summer_day', 'UHI_annual_day', 'UHI_annual_night',
                        'DelNDVI_summer', 'DelNDVI_annual'
                    ] if col in df_cleaned.columns]
                    uhi_scatter_choices = [get_friendly_name(col) for col in [
                        'UHI_summer_day', 'UHI_annual_day', 'UHI_annual_night'
                    ] if col in df_cleaned.columns]
                    greenness_scatter_choices = [get_friendly_name(col) for col in [
                        'DelNDVI_summer', 'NDVI_urb_CT_act_summer', 'DelNDVI_annual'
                    ] if col in df_cleaned.columns]
                    urban_name_input = gr.Dropdown(
                        choices=urban_names_choices,
                        label=get_friendly_name('Urban_name'),
                        multiselect=True,
                        elem_classes="gradio-input-field"
                    )
                    metric_select_dist = gr.Dropdown(
                        choices=metric_dist_choices,
                        label="Select Metric for Distribution",
                        value=metric_dist_choices[0] if metric_dist_choices else None,
                        elem_classes="gradio-input-field"
                    )
                    uhi_metric_select_scatter = gr.Dropdown(
                        choices=uhi_scatter_choices,
                        label="Select UHI Metric for Scatter Plot",
                        value=uhi_scatter_choices[0] if uhi_scatter_choices else None,
                        elem_classes="gradio-input-field"
                    )
                    greenness_metric_select_scatter = gr.Dropdown(
                        choices=greenness_scatter_choices,
                        label="Select Greenness Metric for Scatter Plot",
                        value=greenness_scatter_choices[0] if greenness_scatter_choices else None,
                        elem_classes="gradio-input-field"
                    )
                    update_explorer_btn = gr.Button("Update Charts", elem_classes="gradio-button")
                with gr.Column(scale=2):
                    gr.Markdown('<h4>Visualizations:</h4>')
                    uhi_dist_plot = gr.Plot(label="UHI/Greenness Distribution", elem_classes="plotly-container")
                    uhi_dist_interpretation = gr.Markdown("AI Interpretation will appear here.", elem_classes="plot-interpretation")
                    greenness_uhi_scatter = gr.Plot(label="Greenness vs. UHI Relationship", elem_classes="plotly-container")
                    greenness_uhi_interpretation = gr.Markdown("AI Interpretation will appear here.", elem_classes="plot-interpretation")
            update_explorer_btn.click(
                fn=create_uhi_distribution_plot,
                inputs=[urban_name_input, metric_select_dist],
                outputs=[uhi_dist_plot, uhi_dist_interpretation]
            ).then(
                fn=create_greenness_uhi_scatter,
                inputs=[urban_name_input, uhi_metric_select_scatter, greenness_metric_select_scatter],
                outputs=[greenness_uhi_scatter, greenness_uhi_interpretation]
            )
        with gr.Tab(label="Insights & Trends", id="trends_tab"):
            gr.Markdown('<h2 class="gradient-header">Deeper Insights into UHI Drivers & Seasonal Trends</h2>')
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown('<h3 class="gradient-header">Model Feature Importance</h3>')
                    feature_importance_plot = gr.Plot(label="Feature Importance", elem_classes="plotly-container")
                    feature_importance_interpretation = gr.Markdown("AI Interpretation will appear here.", elem_classes="plot-interpretation")
                    gr.Markdown("This chart shows which factors the AI model considers most influential in predicting Summer Daytime UHI. Higher bars indicate greater impact.")
                    demo.load(fn=create_feature_importance_plot, inputs=[], outputs=[feature_importance_plot, feature_importance_interpretation])
                with gr.Column(scale=1):
                    gr.Markdown('<h3 class="gradient-header">Seasonal UHI Comparison</h3>')
                    urban_names_for_seasonal = get_all_urban_names()
                    urban_name_seasonal = gr.Dropdown(
                        choices=urban_names_for_seasonal,
                        label=get_friendly_name('Urban_name'),
                        multiselect=False,
                        value=urban_names_for_seasonal[0] if urban_names_for_seasonal else None,
                        elem_classes="gradio-input-field"
                    )
                    seasonal_uhi_plot = gr.Plot(label="Seasonal UHI Variations", elem_classes="plotly-container")
                    seasonal_uhi_interpretation = gr.Markdown("AI Interpretation will appear here.", elem_classes="plot-interpretation")
                    gr.Markdown("Compare how UHI intensity changes across different seasons and times of day for a selected urban area. This helps in understanding peak UHI periods.")
                    urban_name_seasonal.change(
                        fn=create_seasonal_uhi_comparison,
                        inputs=[urban_name_seasonal],
                        outputs=[seasonal_uhi_plot, seasonal_uhi_interpretation]
                    )
                    if urban_names_for_seasonal:
                        demo.load(fn=create_seasonal_uhi_comparison, inputs=[urban_name_seasonal], outputs=[seasonal_uhi_plot, seasonal_uhi_interpretation])
        with gr.Tab(label="Ranked Urban Areas", id="ranked_tab"):
            gr.Markdown('<h2 class="gradient-header">Ranked Urban Areas by UHI Intensity</h2>')
            gr.Markdown("""
                This chart allows you to see which urbanized areas in the dataset have the highest or lowest average UHI for a selected metric.
                This provides an alternative way to understand the 'geographic distribution' by ranking areas.
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    ranked_uhi_metric_choices = [get_friendly_name(col) for col in [
                        'UHI_summer_day_city', 'UHI_annual_day_city', 'UHI_annual_night_city',
                        'UHI_winter_day_city', 'UHI_winter_night_city'
                    ] if col in df_cleaned.columns]
                    ranked_uhi_metric_select = gr.Dropdown(
                        choices=ranked_uhi_metric_choices,
                        label="Select UHI Metric to Rank By",
                        value=ranked_uhi_metric_choices[0] if ranked_uhi_metric_choices else None,
                        elem_classes="gradio-input-field"
                    )
                    ranked_top_n_slider = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=20,
                        step=1,
                        label="Number of Top/Bottom Areas to Display",
                        interactive=True,
                        elem_classes="gradio-input-field"
                    )
                    ranked_sort_order_radio = gr.Radio(
                        choices=["Highest UHI (Descending)", "Lowest UHI (Ascending)"],
                        label="Sort Order",
                        value="Highest UHI (Descending)",
                        interactive=True,
                        elem_classes="gradio-input-field"
                    )
                    update_ranked_btn = gr.Button("Update Ranked Chart", elem_classes="gradio-button")
                with gr.Column(scale=2):
                    ranked_uhi_plot = gr.Plot(label="Urban Area UHI Ranking", elem_classes="plotly-container")
                    ranked_uhi_interpretation = gr.Markdown("AI Interpretation will appear here.", elem_classes="plot-interpretation")
            update_ranked_btn.click(
                fn=lambda metric, n, order: create_ranked_uhi_bar_chart(metric, n, sort_ascending=(order == "Lowest UHI (Ascending)")),
                inputs=[ranked_uhi_metric_select, ranked_top_n_slider, ranked_sort_order_radio],
                outputs=[ranked_uhi_plot, ranked_uhi_interpretation]
            )
            if ranked_uhi_metric_choices:
                demo.load(
                    fn=lambda: create_ranked_uhi_bar_chart(
                        ranked_uhi_metric_choices[0],
                        20,
                        False
                    ),
                    inputs=[],
                    outputs=[ranked_uhi_plot, ranked_uhi_interpretation]
                )
        with gr.Tab(label="UHI Predictor & Scenario Builder", id="predictor_tab"):
            gr.Markdown('<h2 class="gradient-header">UHI Predictor: Forecast & Simulate Mitigation Scenarios</h2>')
            if model is None:
                gr.Warning("The UHI prediction model could not be loaded. Please ensure 'uhi_summer_day_predictor_rf (1).joblib' exists and is accessible. Prediction functionality is disabled.")
            elif df_ml.empty:
                gr.Warning("No valid data available for UHI prediction after cleaning. Please check your dataset.")
            else:
                gr.Markdown("""
                    Adjust the sliders below to define a specific urban scenario. The AI model will then predict the Summer Daytime UHI for that scenario. Use this to explore 'what-if' questions about urban design and environmental factors.
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown('<h3 class="gradient-header">Input Scenario Parameters</h3>')
                        gr.Markdown("""
                            **Parameter Descriptions:**
                            - **Summer Greenness Deficit**: Difference between rural and urban greenness. Lower values (more negative) mean more urban greening, which typically reduces UHI.
                            - **Urban Elevation**: The average elevation of the urban area. Higher elevations can sometimes lead to better air circulation, potentially reducing UHI (complex relationship).
                            - **Urban Greenness Index (Summer NDVI)**: A measure of vegetation health and density within the urban area. Higher values indicate more green spaces, which strongly reduces UHI.
                            - **Rural Daytime Temperature (Summer °C)**: The baseline rural temperature. Higher rural temperatures indicate a hotter regional climate, which will lead to generally higher UHI (as UHI is a delta).
                            - **Census Tract Area**: The size of the urbanized area being analyzed. Larger areas might have more complex UHI patterns; its direct impact on UHI can vary but often relates to urban sprawl/intensity.
                            - **Total Density Count**: Represents overall density (e.g., population or building count). Higher density usually means more impervious surfaces and heat-generating activities, strongly increasing UHI.
                            - **Urban Density Count**: Similar to Total Density, focusing on urban elements. Higher values typically imply more built environment, which strongly increases UHI.
                        """)
                        feature_inputs = []
                        for feature_name in model_features:
                            min_val, max_val = FEATURE_RANGES.get(feature_name, (0, 1))
                            avg_val = df_ml[feature_name].median() if feature_name in df_ml.columns and not df_ml[feature_name].empty else (min_val + max_val) / 2
                            avg_val = max(float(min_val), min(float(avg_val), float(max_val)))
                            if (max_val - min_val) <= 0.001:
                                step_size = 0.001
                                min_val = avg_val - 0.05 if avg_val - 0.05 < min_val else min_val
                                max_val = avg_val + 0.05 if avg_val + 0.05 > max_val else max_val
                            else:
                                step_size = (max_val - min_val) / 100
                                if step_size < 1:
                                    step_size = round(step_size, 3)
                                    if step_size == 0: step_size = 0.001
                                else:
                                    step_size = int(step_size) if step_size == int(step_size) else round(step_size, 2)
                            feature_inputs.append(
                                gr.Slider(
                                    minimum=float(min_val),
                                    maximum=float(max_val),
                                    value=float(avg_val),
                                    step=float(step_size),
                                    label=get_friendly_name(feature_name),
                                    interactive=True,
                                    info=f"Typical range: {min_val:.2f} to {max_val:.2f}, Median: {avg_val:.2f}",
                                    elem_classes="gradio-input-field"
                                )
                            )
                        predict_btn = gr.Button("Predict Summer Daytime UHI", elem_classes="gradio-button")
                    with gr.Column(scale=1):
                        gr.Markdown("<h3 class='gradient-header'>Predicted UHI & Mitigation Advice</h3>")
                        predicted_uhi_gauge_plot = gr.Plot(
                            value=create_uhi_gauge_plot(UHI_THRESHOLDS['median'], UHI_MIN_VAL, UHI_MAX_VAL, UHI_THRESHOLDS['q25'], UHI_THRESHOLDS['q75']),
                            label="Predicted UHI Gauge",
                            elem_classes="plotly-container"
                        )
                        predicted_uhi_output_value = gr.HTML(
                            value="<div class='prediction-result-box' style='border-color:gray; color:gray;'>Adjust parameters and click 'Predict'</div>",
                            elem_classes="prediction-output-container"
                        )
                        uhi_category_label = gr.HTML(
                            value="<div class='uhi-category-label'>UHI Category: N/A</div>",
                            elem_classes="uhi-category-label-container"
                        )
                        base_suggestion_html = gr.Markdown(
                            value="<div class='uhi-suggestion'>This section will provide general mitigation advice.</div>",
                            elem_classes="uhi-suggestion-container"
                        )
                        gpt_powered_advice = gr.Markdown(
                            value="<div class='uhi-suggestion'>AI-Powered Mitigation Advice will appear here.</div>",
                            elem_classes="uhi-suggestion-container"
                        )
                predict_btn.click(
                    fn=predict_uhi_scenario,
                    inputs=feature_inputs,
                    outputs=[predicted_uhi_gauge_plot, predicted_uhi_output_value, uhi_category_label, base_suggestion_html, gpt_powered_advice]
                )
        with gr.Tab(label="Ask the AI", id="ai_qa_tab"):
            gr.Markdown('<h2 class="gradient-header">Ask Your Urban Heat Island Questions</h2>')
            if client is None:
                gr.Warning("GPT-4o API is not configured. This feature is disabled.")
            else:
                gr.Markdown("""
                    Type your question about the UHI data, concepts, or general UHI phenomena below.
                    Examples:
                    - "What is the average summer UHI in New York?"
                    - "Explain the urban heat island effect."
                    - "Which city has the highest annual night UHI?"
                    - "What is NDVI?"
                """)
                with gr.Row():
                    nl_query_input = gr.Textbox(label="Your Question", placeholder="e.g., What is the average summer UHI?", lines=2, elem_classes="gradio-input-field")
                with gr.Row():
                    nl_query_btn = gr.Button("Get Answer", elem_classes="gradio-button")
                with gr.Row():
                    nl_query_output = gr.Markdown("The AI's answer will appear here.", elem_classes="uhi-suggestion")
                nl_query_btn.click(
                    fn=natural_language_query,
                    inputs=[nl_query_input],
                    outputs=[nl_query_output]
                )

if __name__ == "__main__":
    demo.launch(debug=True, share=True)