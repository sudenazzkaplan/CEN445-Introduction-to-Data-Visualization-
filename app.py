import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import numpy as np
import networkx as nx

# Page Configuration
st.set_page_config(page_title="Real Estate Analysis Dashboard", layout="wide")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# Navigation in Sidebar
st.sidebar.title("🏠 Navigation")
page = st.sidebar.radio("Select Page:", [
    "Treemap Analysis", 
    "Histogram Analysis", 
    "Proportional Symbol Map Analysis",
    "Sankey Diagram",
    "3D Scatter Plot",
    "Correlation Analysis",
    "Parallel Coordinates",
    "Box Plot Analysis",
    "Network Graph"
])

# Common Filters
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 General Filters")

# Price filter
p_min, p_max = int(df.price.min()), int(df.price.max())
pr = st.sidebar.slider("Price Range", p_min, p_max, (p_min, p_max))

# City filter
all_cities = sorted(df['city'].unique()) if 'city' in df.columns else []
selected_cities = st.sidebar.multiselect("Cities", all_cities, default=all_cities)

# Bedrooms filter
all_bedrooms = sorted(df['bedrooms'].unique()) if 'bedrooms' in df.columns else []
selected_bedrooms = st.sidebar.multiselect("Bedrooms", all_bedrooms, default=all_bedrooms)

# Condition filter
all_conditions = sorted(df['condition'].unique()) if 'condition' in df.columns else []
selected_conditions = st.sidebar.multiselect("Condition", all_conditions, default=all_conditions)

# Area filter
if 'sqft_living' in df.columns:
    min_sqft = int(df["sqft_living"].min())
    max_sqft = int(df["sqft_living"].max())
    sqft_range = st.sidebar.slider("Living Area (sqft)", min_sqft, max_sqft, (min_sqft, max_sqft))

# Filtered Data
d = df[
    df.price.between(*pr) & 
    df.city.isin(selected_cities) & 
    df.bedrooms.isin(selected_bedrooms) & 
    df.condition.isin(selected_conditions)
]

if 'sqft_living' in df.columns:
    d = d[d.sqft_living.between(*sqft_range)]

# Treemap Analysis Page
if page == "Treemap Analysis":
    st.title("🌳 Treemap Analysis")
    
    # Treemap specific filters
    col1, col2, col3 = st.columns(3)
    with col1:
        path = st.multiselect("Hierarchy", ['city', 'statezip', 'condition', 'bedrooms'], default=['city', 'statezip'])
    with col2:
        size = st.selectbox("Size", ['sqft_living', 'price', 'sqft_lot', 'sqft_above'])
    with col3:
        color = st.selectbox("Color", ['price', 'sqft_living', 'yr_built', 'condition'])
    
    if path:
        fig = px.treemap(d, path=path, values=size, color=color, 
                        title=f"Analysis: {size} - {color}", 
                        color_continuous_scale='RdBu_r' if color == 'price' else 'Viridis')
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        st.info(f"🏠 **Total:** {len(d)} Houses | 💰 **Avg. Price:** ${d.price.mean():,.0f} | 📏 **Avg. Area:** {d.sqft_living.mean():,.0f} sqft")
    else:
        st.warning("Please select a hierarchy.")

    if st.checkbox("Show Filtered Data"):
        st.dataframe(d)

# Histogram Analysis Page
elif page == "Histogram Analysis":
    st.title('📊 Histogram Analysis')
    
    # Histogram specific filters
    col1, col2 = st.columns(2)
    with col1:
        col = st.selectbox('Select Column', d.select_dtypes('number').columns)
    with col2:
        color = st.selectbox('Color (Group By)', [None] + [c for c in d.columns if d[c].nunique() < 20])
    
    bins = st.slider('Bin Count', 5, 100, 30)
    
    # Chart Drawing
    fig = px.histogram(d, x=col, color=color, nbins=bins, marginal="box", title=f"{col} Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics and Data Display
    col1, col2 = st.columns(2)
    with col1:
        if st.checkbox("Show Statistics"):
            st.write(d[col].describe())
    with col2:
        if st.checkbox("Show Raw Data"):
            st.write(d.head())

# Proportional Symbol Map Analysis Page
elif page == "Proportional Symbol Map Analysis":
    st.title("🗺️ Proportional Symbol Map Analysis")
    
    # Coordinate check
    if 'lat' not in d.columns or 'lon' not in d.columns:
        np.random.seed(42)
        d['lat'] = np.random.uniform(47.4, 47.8, len(d))
        d['lon'] = np.random.uniform(-122.4, -122.1, len(d))
    
    # Map specific filters
    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("City", ['All'] + list(d.city.unique()))
    with col2:
        bed = st.selectbox("Bedrooms", ['All'] + sorted(d.bedrooms.unique()))
    
    # Filtering
    filtered_data = d[((d.city == city) | (city == 'All')) & ((d.bedrooms == bed) | (bed == 'All'))]
    
    # Map
    if not filtered_data.empty:
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=47.6, longitude=-122.3, zoom=9),
            layers=[pdk.Layer("ScatterplotLayer", filtered_data, get_position=['lon', 'lat'], 
                             get_radius="price/5000", get_fill_color=[255, 0, 0, 160], 
                             pickable=True, auto_highlight=True)],
            tooltip={"html": "<b>Price:</b> ${price}<br><b>City:</b> {city}<br><b>Bedrooms:</b> {bedrooms}", 
                    "style": {"backgroundColor": "steelblue", "color": "white"}}
        ))
    else:
        st.warning("No data found for the selected criteria.")
    
    # Metrics
    cols = st.columns(4)
    metrics = [
        ("Total", len(filtered_data)), 
        ("Avg. Price", f"${filtered_data.price.mean():,.0f}"), 
        ("Avg. Bedrooms", f"{filtered_data.bedrooms.mean():.1f}"), 
        ("Avg. Sqft", f"{filtered_data.sqft_living.mean():.0f}")
    ]
    for col, (label, val) in zip(cols, metrics):
        col.metric(label, val)
    
    if st.checkbox("Show Filtered Data"):
        st.dataframe(filtered_data.head())

# Sankey Diagram Page
elif page == "Sankey Diagram":
    st.title("📊 Sankey Diagram Analysis")
    
    st.write(f"Filtered record count: {len(d)}")
    
    if d.empty:
        st.warning("No data found matching filters.")
    else:
        # Sankey Diagram
        st.header("Sankey Diagram: Bedrooms → Condition")
        
        if "bedrooms" in d.columns and "condition" in d.columns:
            grouped = (
                d.groupby(["bedrooms", "condition"])
                .size()
                .reset_index(name="count")
            )

            grouped["bed_label"] = "Bedrooms: " + grouped["bedrooms"].astype(str)
            grouped["cond_label"] = "Condition: " + grouped["condition"].astype(str)

            labels = pd.concat([grouped["bed_label"], grouped["cond_label"]]).unique().tolist()
            label_to_index = {label: i for i, label in enumerate(labels)}

            sources = grouped["bed_label"].map(label_to_index).tolist()
            targets = grouped["cond_label"].map(label_to_index).tolist()
            values = grouped["count"].tolist()

            sankey_fig = go.Figure(data=[go.Sankey(
                node=dict(
                    label=labels,
                    pad=20,
                    thickness=15
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values
                )
            )])

            sankey_fig.update_layout(
                title_text="House Flow: Bedrooms to Condition",
                font=dict(size=12)
            )

            st.plotly_chart(sankey_fig, use_container_width=True)
        else:
            st.warning("Required columns (bedrooms, condition) not found for Sankey diagram.")

# 3D Scatter Plot Page
elif page == "3D Scatter Plot":
    st.title("🔮 3D Scatter Plot Analysis")
    
    if d.empty:
        st.warning("No data found matching filters.")
    else:
        highlight_bedroom = st.selectbox("Bedrooms to Highlight", ["All"] + sorted(d.bedrooms.unique()))

        if all(col in d.columns for col in ["sqft_living", "sqft_lot", "price"]) and "bedrooms" in d.columns:

            if highlight_bedroom != "All":
                df_highlight = d[d["bedrooms"] == highlight_bedroom]
                df_other = d[d["bedrooms"] != highlight_bedroom]

                scatter_fig = go.Figure()

                scatter_fig.add_trace(go.Scatter3d(
                    x=df_other["sqft_living"],
                    y=df_other["sqft_lot"],
                    z=df_other["price"],
                    mode="markers",
                    marker=dict(size=3, opacity=0.2),
                    name="Other Bedroom Counts"
                ))

                scatter_fig.add_trace(go.Scatter3d(
                    x=df_highlight["sqft_living"],
                    y=df_highlight["sqft_lot"],
                    z=df_highlight["price"],
                    mode="markers",
                    marker=dict(size=6, opacity=1),
                    name=f"Highlighted: {highlight_bedroom}"
                ))

                scatter_fig.update_layout(
                    title="3D Scatter: Living Area vs Lot Size vs Price",
                    scene=dict(
                        xaxis_title="Living Area",
                        yaxis_title="Lot Size",
                        zaxis_title="Price"
                    )
                )
            else:
                scatter_fig = px.scatter_3d(
                    d,
                    x="sqft_living",
                    y="sqft_lot",
                    z="price",
                    color="bedrooms",
                    opacity=0.7,
                    title="3D Scatter: Living Area vs Lot Size vs Price"
                )
                scatter_fig.update_traces(marker=dict(size=3))

            st.plotly_chart(scatter_fig, use_container_width=True)
        else:
            st.warning("Required columns not found for 3D scatter plot.")

# Correlation Analysis Page
elif page == "Correlation Analysis":
    st.title("📈 Correlation Analysis")
    
    if d.empty:
        st.warning("No data found matching filters.")
    else:
        # Correlation Heatmap
        st.header("Correlation Heatmap")

        numeric_df = d.select_dtypes(include=["int64", "float64"])
        if not numeric_df.empty:
            corr = numeric_df.corr()

            fig_heatmap = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap"
            )

            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Correlation statistics
            st.subheader("Highest Correlations")
            corr_pairs = corr.unstack().sort_values(ascending=False)
            high_corr = corr_pairs[(corr_pairs < 1.0) & (corr_pairs > 0.5)]
            if not high_corr.empty:
                st.write(high_corr)
            else:
                st.info("No correlations greater than 0.5 found.")
        else:
            st.warning("No numeric data found for correlation analysis.")

# Parallel Coordinates Page
elif page == "Parallel Coordinates":
    st.title("📐 Parallel Coordinates Analysis")
    
    col1, col2 = st.columns(2)
    col1.metric("Total Records", len(df))
    col2.metric("Filtered Records", len(d))

    if d.empty:
        st.warning("⚠️ No data found with current filters. Please adjust filters in the sidebar.")
    else:
        # Parallel Coordinates
        st.header("Parallel Coordinate Analysis")
        st.write("Analyze relationships between Price, Living Area, Floors, and Bedrooms.")

        if all(col in d.columns for col in ['price', 'sqft_living', 'floors', 'bedrooms']):
            fig_parallel = px.parallel_coordinates(
                d,
                color="price",
                dimensions=['price', 'sqft_living', 'floors', 'bedrooms'],
                color_continuous_scale='viridis',
                title="Multivariate Analysis"
            )
            st.plotly_chart(fig_parallel, use_container_width=True)
        else:
            st.warning("Required columns not found for Parallel Coordinates.")

# Box Plot Analysis Page
elif page == "Box Plot Analysis":
    st.title("📦 Box Plot Analysis")
    
    if d.empty:
        st.warning("No data found matching filters.")
    else:
        st.header("Distribution Analysis (Box Plot)")
        
        b_col1, b_col2 = st.columns(2)

        with b_col1:
            st.subheader("Price by Floors")
            if 'floors' in d.columns and 'price' in d.columns:
                fig_box1 = px.box(d, x='floors', y='price', color='floors', points="all")
                st.plotly_chart(fig_box1, use_container_width=True)
            else:
                st.warning("Required columns (floors, price) not found.")

        with b_col2:
            st.subheader("Price by City")
            if 'city' in d.columns and 'price' in d.columns:
                fig_box2 = px.box(d, x='city', y='price', color='city')
                st.plotly_chart(fig_box2, use_container_width=True)
            else:
                st.warning("Required columns (city, price) not found.")

# Network Graph Page
elif page == "Network Graph":
    st.title("🕸️ Network Graph Analysis")
    
    # Network Graph Simulation
    st.header("Network Graph Simulation")
    
    # Network settings
    col_net1, col_net2 = st.columns([1, 3])
    
    with col_net1:
        n_nodes = st.slider("Number of Nodes", 5, 50, 15)
        prob_edge = st.slider("Edge Probability", 0.1, 1.0, 0.3)

    with col_net2:
        # Create Graph
        G = nx.erdos_renyi_graph(n=n_nodes, p=prob_edge, seed=42)
        pos = nx.spring_layout(G, seed=42)

        # Edges
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Nodes
        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f'Node {node}')

        # Graph Figure
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=20,
                color=[],
                line_width=2
            )
        )
        
        # Color nodes by number of connections
        node_adjacencies = []
        for node, adacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adacencies[1]))
        node_trace.marker.color = node_adjacencies

        fig_net = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=500
                        ))

        st.plotly_chart(fig_net, use_container_width=True)

    # Network Statistics
    st.info(f"""
    **Network Statistics:**
    - Nodes: {n_nodes}
    - Probability: {prob_edge}
    - Estimated Edges: {len(G.edges())}
    - Average Degree: {np.mean([deg for _, deg in G.degree()]):.2f}
    """)
