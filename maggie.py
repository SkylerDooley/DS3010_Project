import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import shap
import xgboost as xgb
import plotly.graph_objects as go


from sklearn.linear_model import LinearRegression, Ridge, PoissonRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score


from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from scipy import stats



def plot_migration(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['sighting_date'], df['latitude'], alpha=0.3, s=10)
    plt.title('Northward Hummingbird Migration (2015–2020)')
    plt.xlabel('Date of First Sighting')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("migration_plot.png", dpi=300)
    plt.show()

def plot_arrival_histogram(df):
    df['week'] = df['sighting_date'].dt.isocalendar().week
    plt.figure(figsize=(10, 6))
    plt.hist(df['week'], bins=range(1, 54), edgecolor='black')
    plt.title('Hummingbird First Sightings by Week (2015–2020)')
    plt.xlabel('Week of Year')
    plt.ylabel('Number of Sightings')
    plt.tight_layout()
    plt.savefig("arrival_histogram_by_week.png", dpi=300)
    plt.show()

def plot_arrival_histogram_by_species(df):
    df['week'] = df['sighting_date'].dt.isocalendar().week

    # Keep only top 2 species, group the rest
    top_species = df['species_old'].value_counts().nlargest(2).index.tolist()
    df['species_grouped'] = df['species_old'].apply(lambda x: x if x in top_species else 'Other')

    plt.figure(figsize=(10, 6))
    for species in df['species_grouped'].unique():
        weekly_counts = df[df['species_grouped'] == species]['week']
        plt.hist(weekly_counts, bins=range(1, 54), alpha=0.5, label=species, edgecolor='black')

    plt.title('Arrival Timing by Week – Top 2 Species')
    plt.xlabel('Week of Year')
    plt.ylabel('Sightings')
    plt.legend()
    plt.tight_layout()
    plt.savefig("arrival_histogram_top2_species.png", dpi=300)
    plt.show()


def plot_migration_map(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['longitude'], df['latitude'], alpha=0.3, s=10)
    plt.title('Hummingbird Sightings (2015–2020)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("migration_map.png", dpi=300)
    plt.show()


def plot_interactive_map(df):
    # Sample data if it's large (improves performance)
    if len(df) > 5000:
        df_sample = df.sample(n=5000, random_state=42)
    else:
        df_sample = df

    # Reduce data columns to only what's needed
    plot_df = df_sample[['latitude', 'longitude', 'species_old', 'sighting_date', 'number']].copy()

    # Convert date to string format to reduce size
    plot_df['sighting_date'] = plot_df['sighting_date'].dt.strftime('%Y-%m-%d')

    # Create the plot with optimized settings
    fig = px.scatter_geo(
        plot_df,
        lat='latitude',
        lon='longitude',
        color='species_old',
        hover_name='species_old',
        hover_data=['sighting_date', 'number'],
        title='Interactive Map of Hummingbird Sightings (2015–2020)',
        opacity=0.4,
        height=600
    )

    # Optimize rendering
    fig.update_traces(marker=dict(size=5))

    # Set the map scope once
    fig.update_layout(
        geo=dict(
            scope='north america',
            showland=True,
            landcolor='rgb(217, 217, 217)',
            projection_scale=1  # Optimize zoom level
        ),
        template='plotly_white',
        width=1000,
        height=800

    )

    # Save with full plotly.js included
    fig.write_html("interactive_migration_map.html", include_plotlyjs=True)

    # Instead of fig.show(), just display a message
    print("Map saved to 'interactive_migration_map.html'. Please open this file in your browser.")

def plot_migration_with_seasonal_regression(df):
    # Filter valid data
    df = df.dropna(subset=['sighting_date', 'latitude'])

    # Extract day of year (DOY) to capture seasonality
    df['doy'] = df['sighting_date'].dt.dayofyear

    # Add year as a feature to track long-term trends
    df['year'] = df['sighting_date'].dt.year

    # Create features for seasonal model
    # Using sine and cosine transforms to capture cyclical nature
    df['sin_doy'] = np.sin(2 * np.pi * df['doy']/365)
    df['cos_doy'] = np.cos(2 * np.pi * df['doy']/365)

    # Create model features
    X = df[['year', 'sin_doy', 'cos_doy']].values
    y = df['latitude'].values

    # Fit model
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=0.5)  # Ridge regression for better stability
    model.fit(X, y)

    # Generate prediction dates for smooth curve
    years = sorted(df['year'].unique())
    days = np.arange(1, 366)
    pred_df = pd.DataFrame([(y, d) for y in years for d in days if d % 5 == 0],
                          columns=['year', 'doy'])
    pred_df['sin_doy'] = np.sin(2 * np.pi * pred_df['doy']/365)
    pred_df['cos_doy'] = np.cos(2 * np.pi * pred_df['doy']/365)

    # Generate predictions
    pred_X = pred_df[['year', 'sin_doy', 'cos_doy']].values
    predictions = model.predict(pred_X)

    # Create date column for plotting
    base_date = pd.Timestamp(str(pred_df['year'].min()))
    pred_df['date'] = pd.to_datetime(
        [base_date + pd.Timedelta(days=int(d-1) + 365*(y-years[0]))
         for y, d in zip(pred_df['year'], pred_df['doy'])]
    )
    pred_df['predicted_latitude'] = predictions

    # Plot with improved styling
    import seaborn as sns
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    # Plot actual data
    plt.scatter(df['sighting_date'], df['latitude'], alpha=0.2, s=8,
               color='skyblue', label='Sightings', rasterized=True)

    # Plot model for each year
    for year in years:
        year_pred = pred_df[pred_df['year'] == year]
        plt.plot(year_pred['date'], year_pred['predicted_latitude'],
                linewidth=2, label=f'Model {year}' if year == years[0] else None)

    # Calculate metrics
    from sklearn.metrics import mean_squared_error, r2_score
    pred_y = model.predict(X)
    mse = mean_squared_error(y, pred_y)
    r2 = r2_score(y, pred_y)

    # Format plot
    plt.title('Northward Hummingbird Migration with Seasonal Model (2015–2020)', fontsize=14)
    plt.xlabel('Date of Sighting', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.legend(frameon=True)
    plt.tight_layout()

    # Add model info on plot
    coef_text = f"Year coef: {model.coef_[0]:.4f}\nR² score: {r2:.4f}"
    plt.annotate(coef_text, xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # Save and show
    plt.savefig("migration_with_seasonal_model.png", dpi=300, bbox_inches='tight')
    print(f"Seasonal Model MSE: {mse:.4f}")
    print(f"Seasonal Model R² score: {r2:.4f}")
    print(f"Year coefficient: {model.coef_[0]:.6f} (latitude degrees/year)")
    plt.show()

    # Optional: Plot the seasonal component separately
    plt.figure(figsize=(10, 6))
    sample_year = years[0]
    year_pred = pred_df[pred_df['year'] == sample_year].copy()
    year_pred['month'] = pd.to_datetime(year_pred['date']).dt.month

    plt.plot(year_pred['doy'], year_pred['predicted_latitude'], linewidth=2)
    plt.title(f'Seasonal Migration Pattern (Year {sample_year})', fontsize=14)
    plt.xlabel('Day of Year', fontsize=12)
    plt.ylabel('Predicted Latitude', fontsize=12)

    # Add month labels
    month_positions = [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(month_positions, month_names)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("seasonal_pattern.png", dpi=300)
    plt.show()

def plot_run_xgboost_with_shap(df):

    # Drop missing values
    df = df.dropna(subset=['longitude', 'sighting_date', 'number'])

    # Feature engineering
    df['doy'] = df['sighting_date'].dt.dayofyear
    df['year'] = df['sighting_date'].dt.year
    df['sin_doy'] = np.sin(2 * np.pi * df['doy'] / 365)
    df['cos_doy'] = np.cos(2 * np.pi * df['doy'] / 365)
    df['lat_lon_interaction'] = df['longitude'] * df['cos_doy']

    # Features & target
    features = ['longitude', 'year', 'sin_doy', 'cos_doy', 'lat_lon_interaction']
    X = df[features]
    y = df['number']  # Predicting count of hummingbirds

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")

    # SHAP
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X)

    # SHAP summary bar plot
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP Summary Bar Plot")
    plt.tight_layout()
    plt.savefig("shap_summary_bar.png", dpi=300)
    plt.show()

    # SHAP beeswarm plot
    shap.summary_plot(shap_values, X, plot_type="dot", show=False)
    plt.title("SHAP Beeswarm Plot")
    plt.tight_layout()
    plt.savefig("shap_beeswarm.png", dpi=300)
    plt.show()

    print("Saved SHAP plots: shap_summary_bar.png and shap_beeswarm.png")


def plot_time_series_forecast_with_prophet_advanced(df):
    # Step 1: Clean + aggregate daily data
    df = df.dropna(subset=['sighting_date', 'number'])
    df_daily = df.groupby('sighting_date')['number'].sum().reset_index()

    # Step 2: Handle outliers using IQR
    Q1 = df_daily['number'].quantile(0.25)
    Q3 = df_daily['number'].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    df_daily['number'] = np.where(df_daily['number'] > upper_limit, upper_limit, df_daily['number'])

    # Step 3: Ensure continuous daily index
    full_range = pd.date_range(df_daily['sighting_date'].min(), df_daily['sighting_date'].max())
    df_daily = df_daily.set_index('sighting_date').reindex(full_range).fillna(0).rename_axis('ds').reset_index()
    df_daily.rename(columns={'number': 'y'}, inplace=True)

    # Step 4: Initialize Prophet model with tuning
    model = Prophet(
        yearly_seasonality=True,
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10.0,
        interval_width=0.9
    )
    model.fit(df_daily)

    # Step 5: Forecast
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    # Step 6: Cross-validation & metrics
    df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='90 days')
    df_perf = performance_metrics(df_cv)
    print(df_perf[['mae', 'rmse']].mean())  # Removed 'mape'


    # Step 7: Forecast plot with CI shading
    fig1 = model.plot(forecast)
    plt.title("Hummingbird Sightings Forecast (with 90% Confidence Interval)", fontsize=14)
    plt.tight_layout()
    plt.savefig("forecast_prophet.png", dpi=300)
    plt.show()

    # Step 8: Component plot (trend/seasonality)
    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig("forecast_components_prophet.png", dpi=300)
    plt.show()

    # Step 9: Actual vs predicted (zoom-in)
    merged = df_daily.merge(forecast[['ds', 'yhat']], on='ds', how='left')
    plt.figure(figsize=(12, 5))
    plt.plot(merged['ds'], merged['y'], label='Actual', alpha=0.6)
    plt.plot(merged['ds'], merged['yhat'], label='Predicted', color='red', linewidth=2)
    plt.xlim(forecast['ds'].min(), forecast['ds'].max())
    plt.title("Actual vs Predicted Hummingbird Counts", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=300)
    plt.show()

    print("Saved: forecast_prophet.png, forecast_components_prophet.png, actual_vs_predicted.png")

def plot_kmeans_clusters_advanced(df, max_k=10):
    
    # Step 1: Clean data
    df = df.dropna(subset=['latitude', 'longitude', 'sighting_date', 'number']).copy()
    df['week'] = df['sighting_date'].dt.isocalendar().week

    # Step 2: Normalize spatial + temporal features
    features = df[['latitude', 'longitude', 'week']]
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Step 3: Determine optimal number of clusters using silhouette score
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)

    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters (K): {best_k}")

    # Step 4: Final KMeans clustering with best K
    kmeans = KMeans(n_clusters=best_k, n_init='auto', random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    # Inverse-transform centroids to original space
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    df['centroid_lat'] = df['cluster'].map({i: c[0] for i, c in enumerate(centroids)})
    df['centroid_lon'] = df['cluster'].map({i: c[1] for i, c in enumerate(centroids)})

    # Step 5: Cluster statistics summary
    stats = df.groupby('cluster').agg({
        'latitude': ['mean', 'count'],
        'number': 'mean'
    }).rename(columns={'mean': 'avg', 'count': 'n_obs'})
    print("\nCluster summary statistics:")
    print(stats)

    # Step 6: Interactive map with Plotly
    fig = px.scatter_geo(
        df,
        lat='latitude',
        lon='longitude',
        color='cluster',
        hover_name='cluster',
        hover_data=['week', 'number'],
        title='Hummingbird Migration Clusters (Interactive)',
        opacity=0.6
    )

    # Add centroid markers
    fig.add_trace(go.Scattergeo(
        lat=centroids[:, 0],
        lon=centroids[:, 1],
        mode='markers+text',
        marker=dict(size=10, color='black', symbol='x'),
        text=[f'C{i}' for i in range(best_k)],
        textposition='top center',
        name='Centroids'
    ))

    fig.update_layout(
        geo=dict(scope='north america'),
        legend_title="Cluster",
        height=700,
        template="plotly_white"
    )

    fig.write_html("kmeans_clusters_interactive.html", include_plotlyjs='cdn')
    print("Interactive map saved as: kmeans_clusters_interactive.html")

    # Step 7: Elbow plot
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Elbow Method - Silhouette Score vs K")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("elbow_silhouette.png", dpi=300)
    plt.show()


def main():
    df = pd.read_csv('filtered_hummingbirds_2015.csv', parse_dates=['sighting_date'])
    plot_migration(df)
    plot_arrival_histogram(df)
    plot_arrival_histogram_by_species(df)
    plot_migration_map(df)
    plot_interactive_map(df)
    plot_migration_with_seasonal_regression(df)
    plot_run_xgboost_with_shap(df)
    plot_time_series_forecast_with_prophet_advanced(df)
    plot_kmeans_clusters_advanced(df, max_k=10)



if __name__ == "__main__":
    main()