import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sl_utils.logger import streamlit_logger
from sl_utils.logger import log_function_call


@log_function_call(streamlit_logger)
def display_maps():
    # Load dataset
    with st.spinner('Loading data...'):
        # try to get data from session state
        articles = st.session_state.data_for_map
        if articles is None:
            articles = pd.read_csv("data/articlesformap.csv",
                                    dtype={"country": str,
                                            "continent": str,
                                            "subcontinent": str,
                                            "year": int,
                                            "month": int,
                                            "day": int,
                                            "fake_count": int,
                                            "real_count": int}, parse_dates=["date"])
    if articles is None or articles.empty:
        st.error("Failed to load data. Check ETL process.")
        streamlit_logger.error("Failed to load data. Check ETL process.")
        return
    # **Ensure 'date' column is in proper datetime format**
    if "date" not in articles or articles["date"].isnull().all():
        # **Fallback: Construct date from year, month, day**
        articles["date"] = pd.to_datetime(
            articles[["year", "month", "day"]]
            .astype(str).agg('-'.join, axis=1),
            errors='coerce'
        )
    # If there are still NaT values, fill with a default valid date
    articles["date"].fillna(pd.Timestamp("2000-01-01"), inplace=True)

    # **Convert date column to datetime.date for
    # Streamlit slider compatibility**
    articles["date"] = articles["date"].dt.date

    # Get min/max dates
    min_date, max_date = articles["date"].min(), articles["date"].max()

    col1, col2 = st.columns([0.20, 0.80], vertical_alignment="center")

    with col1:
        # **Date Range Slider**
        start_date, end_date = st.slider(
            "Select Date Range:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM"
        )

        # **User Selection for Geographic Level**
        geo_level = st.selectbox(
            "Select Geographic Level:",
            ["Continent", "Subcontinent", "Country", "State"],
            index=2  # Default to Country
        )

        # **Filter Data Based on Selected Date Range**
        filtered_data = (
            articles[(articles["date"] >= start_date) &
                    (articles["date"] <= end_date)])

        if geo_level == "Continent":
            group_col = "continent"
            location_mode = "country names"
            map_countries = True
        elif geo_level == "Subcontinent":
            group_col = "subcontinent"
            location_mode = "country names"
            map_countries = True
        elif geo_level == "Country":
            group_col = "country"
            location_mode = "country names"
            map_countries = False
        else:  # State Level
            # Use "ISO-3166-2" for international state codes
            group_col = "state"
            location_mode = "USA-states"
            map_countries = False

        # **Aggregate Data by Country (Summing Fake and Real Articles)**
        aggregated_data = filtered_data.groupby([group_col]).agg(
            {"fake_count": "sum", "real_count": "sum"}).reset_index()

        if map_countries:
            country_mapping = articles[["country", group_col]].drop_duplicates()
            aggregated_data = aggregated_data.merge(
                country_mapping, on=group_col, how="left")

        # **Calculate Percentage of Fake Articles**
        aggregated_data["total_articles"] = (
            aggregated_data["fake_count"] + aggregated_data["real_count"]
        )
        aggregated_data["fake_percentage"] = ((
            aggregated_data["fake_count"] /
            aggregated_data["total_articles"]) * 100
        ).fillna(0)  # Fill NaN with 0 if division fails

        # Generate a blended color value:
        # - Red = Fake-heavy
        # - Blue = Real-heavy
        # - Purple = Balanced mix
        aggregated_data["merged_scale"] = aggregated_data.apply(
            lambda row: row["fake_count"] / row["total_articles"]
            if row["total_articles"] > 0 else 0.5,
            axis=1
        )

        # **Option to Switch Between Absolute Count, Percentage, and Merged**
        display_mode = st.radio(
            "Select Display Mode:",
            options=["Fake Articles",
                    "Real Articles",
                    "All Articles",
                    "% Fake of Total"],
            index=0
        )
    with col2:
        # **Create Choropleth Map**
        fig = go.Figure()

        location_column = "country" if map_countries else group_col

        if display_mode == "Fake Articles":
            # **Fake News Choropleth**
            fig.add_trace(go.Choropleth(
                locations=aggregated_data[location_column],
                z=aggregated_data["fake_count"],
                locationmode=location_mode,
                colorscale='Reds',
                colorbar=dict(title="Fake Articles", x=1.02),
                text=aggregated_data[['fake_count', 'real_count']],
                hovertemplate=(
                    '<b>%{location}</b><br>'
                    'Fake Articles: %{z}<br>'
                    'Real Articles: %{text[1]}<br>'
                ),
                name="Fake Articles"
            ))
        elif display_mode == "Real Articles":
            # **Real News Choropleth**
            fig.add_trace(go.Choropleth(
                locations=aggregated_data[location_column],
                z=aggregated_data["real_count"],
                locationmode=location_mode,
                colorscale='Blues',
                colorbar=dict(title="Real Articles", x=1.02),

                text=aggregated_data[['fake_count', 'real_count']],
                hovertemplate=(
                    '<b>%{location}</b><br>'
                    'Fake Articles: %{text[0]}<br>'
                    'Real Articles: %{z}<br>'
                ),
                name="Real Articles"
            ))
        elif display_mode == "All Articles":
            fig.add_trace(go.Choropleth(
                locations=aggregated_data[location_column],
                z=aggregated_data["merged_scale"],
                locationmode=location_mode,
                # Blue (Real) → Purple (Mix) → Red (Fake)
                colorscale=[(0, "blue"), (0.5, "purple"), (1, "red")],
                colorbar=dict(title="Fake-Real Mix", x=1.02),
                text=aggregated_data[['fake_count', 'real_count',
                                    'fake_percentage']],
                hovertemplate=(
                    '<b>%{location}</b><br>'
                    'Fake Articles: %{text[0]}<br>'
                    'Real Articles: %{text[1]}<br>'
                    '% Fake Articles: %{text[2]:.2f}%<br>'
                ),
                name="Fale-Real Mix"
            ))

        else:  # **Percentage of Fake Articles**
            fig.add_trace(go.Choropleth(
                locations=aggregated_data[location_column],
                z=aggregated_data["fake_percentage"],
                locationmode=location_mode,
                colorscale='OrRd',  # Orange-Red for percentage
                colorbar=dict(title="% Fake Articles", x=1.02),  # Center legend
                text=aggregated_data[['fake_percentage']],
                hovertemplate=(
                    '<b>%{location}</b><br>'
                    '% Fake Articles: %{z:.2f}%<br>'
                ),
                name="% Fake Articles"
            ))

        # **Update Layout**
        fig.update_layout(
            title_text=(f'Article Mentions ({geo_level} level)'
                        f' (From {start_date.strftime("%Y-%m")} '
                        f'to {end_date.strftime("%Y-%m")})'),
            geo=dict(showframe=True,
                    showcoastlines=True,
                    projection_type='equirectangular'),
            showlegend=False,
            width=800,
            height=800
        )

        # **Display Map**
        st.plotly_chart(fig)
