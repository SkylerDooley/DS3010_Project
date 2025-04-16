import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


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



def main():
    df = pd.read_csv('filtered_hummingbirds_2015.csv', parse_dates=['sighting_date'])
    plot_migration(df)
    plot_arrival_histogram(df)
    plot_arrival_histogram_by_species(df)
    plot_migration_map(df)
    plot_interactive_map(df)

if __name__ == "__main__":
    main()
