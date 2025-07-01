# Imports
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn import tree
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import plotly.express as px

# Dashboard im Browser anzeigen
# http://127.0.0.1:8050/

df = pd.read_csv("data/smoker_train.csv")


# load images
def encode_image(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"


tree = encode_image("images/dashboard_tree.jpeg")
weight_height = encode_image("images/dashboard_weight_height.png")
age = encode_image("images/dashboard_age.png")
smoker = encode_image("images/dashboard_smoker.png")
FeatureImportance_RandomForest = encode_image("images/dashboard_FeatureImportances_RandomForest.png")


# Diagramme erzeugen -----------------------------------------------------------------------------------------------------------
# Scatterplot erzeugen
scatter_fig = px.scatter(
    df,
    x="height(cm)",
    y="weight(kg)",
    opacity=0.3,
    labels={"height(cm)": "Height (cm)", "weight(kg)": "Weight (kg)"},
)

# Histogramm für Age Distribution erzeugen
age_count = df["age"].value_counts().sort_index().reset_index()
age_count.columns = ["age", "count"]

age_hist_fig = px.bar(
    age_count,
    x="age",
    y="count",
    labels={"age": "Age", "count": "Count"},
)

# Smoker - Non-Smoker Verteilung
smoker_counts = df["smoking"].value_counts().sort_index()

smoker_pie_fig = px.pie(
    names=["Non-Smoker", "Smoker"],
    values=smoker_counts,
    # color_discrete_sequence=px.colors.qualitative.Pastel,
    color_discrete_sequence=["#4c96df", "#e82626"],
)

# Hemoglobin-Verteilung nach Raucherstatus erstellen
def create_hemoglobin_chart():
    plt.figure(figsize=(10, 5))
    for status in df["smoking"].unique():
        subset = df[df["smoking"] == status]
        plt.hist(subset["hemoglobin"], bins=100, alpha=0.5, label=f"Smoking: {status}")
    plt.title("Hemoglobin Distribution by Smoking Status")
    plt.xlabel("Hemoglobin")
    plt.ylabel("Count")
    plt.legend()

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{img_base64}"

hemoglobin_chart = create_hemoglobin_chart()


# Dash-Anwendung initialisieren -----------------------------------------------------------------------------------------------------------
app = dash.Dash(__name__)

# Layout des Dash-Dashboards definieren
app.layout = html.Div(
    [
        # Header -----------------------------------------------------------------------------------------------------------
        html.Div(
            [
                # Oberer Bereich mit Logo und Kassen-Identifikation
                html.Div(
                    [
                        # Linke Seite
                        html.Div(
                            [
                                html.Img(
                                    src="https://img.icons8.com/ios-filled/50/4a90e2/caduceus.png",
                                    style={"height": "40px"},
                                ),
                                html.Div(
                                    [
                                        html.H3(
                                            "HealthCare Assurance",
                                            style={"margin": "0", "color": "#2c3e50"},
                                        ),
                                        html.P(
                                            "Gesundheit im Fokus",
                                            style={
                                                "margin": "0",
                                                "fontSize": "0.8em",
                                                "color": "#7f8c8d",
                                            },
                                        ),
                                    ],
                                    style={"marginLeft": "15px"},
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center"},
                        ),
                        # Rechte Seite
                        html.Div(
                            [
                                html.P(
                                    "Internes System",
                                    style={
                                        "margin": "0",
                                        "fontSize": "0.85em",
                                        "color": "#7f8c8d",
                                    },
                                ),
                                html.P(
                                    "Version 2.1",
                                    style={
                                        "margin": "0",
                                        "fontSize": "0.75em",
                                        "color": "#95a5a6",
                                    },
                                ),
                            ],
                            style={"textAlign": "right"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "width": "100%",
                    },
                ),
                # Horizontale Linie zur Trennung
                html.Hr(
                    style={
                        "margin": "15px 0",
                        "border": "none",
                        "height": "1px",
                        "backgroundColor": "#4c96df",
                    }
                ),

                # Unterer Bereich mit Dashboard-Titel und Beschreibung
                html.Div(
                    [
                        html.H2(
                            "Raucher-Verifizierung",
                            style={
                                "margin": "0",
                                "color": "#4c96df",
                                "fontWeight": "500",
                                "letterSpacing": "0.5px",
                            },
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Status: ",
                                    style={"fontWeight": "bold", "color": "#7f8c8d"},
                                ),
                                html.Span(
                                    "Aktiv",
                                    style={
                                        "color": "#27ae60",
                                        "fontWeight": "bold",
                                        "backgroundColor": "#e8f5e9",
                                        "padding": "3px 8px",
                                        "borderRadius": "12px",
                                        "fontSize": "0.85em",
                                    },
                                ),
                            ],
                            style={"marginTop": "5px"},
                        ),
                    ]
                ),
            ],
            style={
                "backgroundColor": "#ffffff",
                "padding": "20px 30px",
                "borderBottom": "1px solid #e0e6ed",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
                "marginBottom": "25px",
            },
        ),

        # Info-Leiste
        html.Div(
            [
                html.Div(
                    [
                        html.Span("ℹ️", style={"marginRight": "8px"}),
                        html.Span(
                            # "Dieses System dient ausschließlich zur internen Verwendung gemäß DSGVO §15. Bitte behandeln Sie alle Daten vertraulich."
                            "Dieses System dient ausschließlich zur internen Verwendung und befindet sich aktuell in einer Testversion. Bitte behandeln Sie alle Daten vertraulich."
                        ),
                    ]
                )
            ],
            style={
                "backgroundColor": "#f1f8ff",
                "color": "#2471a3",
                "padding": "10px 30px",
                "fontSize": "0.9em",
                "marginBottom": "25px",
                "borderLeft": "4px solid #3498db",
            },
        ),

        # Main Content -----------------------------------------------------------------------------------------------------------
        html.Div(
            [
                # Übersicht über die Teilnehmerdaten -----------------------------------------------------------------------------------------------------------
                html.Details(
                    [
                        html.Summary(
                            "Teilnehmerdaten ausblenden/anzeigen",
                            style={
                                "cursor": "pointer",
                                "fontWeight": "bold",
                                "padding": "10px",
                                "backgroundColor": "#f1f8ff",
                                "borderRadius": "5px",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.P(
                                            "Verteilung von Gewicht und Größe",
                                            style={
                                                "textAlign": "center",
                                                "fontWeight": "bold",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        # html.Img(
                                        #     src=weight_height,
                                        #     style={"width": "100%", "height": "120px", "objectFit": "contain", "marginBottom": "8px"},
                                        # ),
                                        dcc.Graph(
                                            figure=scatter_fig,
                                            config={"displayModeBar": False},
                                            style={
                                                "height": "calc(100% - 40px)",
                                                "width": "100%",
                                                "margin": "0 auto",
                                            },
                                        ),
                                    ],
                                    style={
                                        "flex": "1 1 0",
                                        "minWidth": "0",
                                        "height": "340px",
                                        "boxShadow": "0 4px 8px rgba(0,0,0,0.2)",
                                        "borderRadius": "8px",
                                        "overflow": "hidden",
                                        "backgroundColor": "#ffffff",
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "justifyContent": "flex-start",
                                        "alignItems": "stretch",
                                        "margin": "10px",
                                        "padding": "10px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.P(
                                            "Verteilung von Raucher und Nichtraucher",
                                            style={
                                                "textAlign": "center",
                                                "fontWeight": "bold",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        # html.Img(
                                        #     src=smoker,
                                        #     style={"width": "100%", "height": "120px", "objectFit": "contain", "marginBottom": "8px"},
                                        # ),
                                        dcc.Graph(
                                            figure=smoker_pie_fig,
                                            config={"displayModeBar": False},
                                            style={
                                                "height": "calc(100% - 40px)",
                                                "width": "100%",
                                                "margin": "0 auto",
                                            },
                                        ),
                                    ],
                                    style={
                                        "flex": "1 1 0",
                                        "minWidth": "0",
                                        "height": "340px",
                                        "boxShadow": "0 4px 8px rgba(0,0,0,0.2)",
                                        "borderRadius": "8px",
                                        "overflow": "hidden",
                                        "backgroundColor": "#ffffff",
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "justifyContent": "flex-start",
                                        "alignItems": "stretch",
                                        "margin": "10px",
                                        "padding": "10px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.P(
                                            "Verteilung nach Alter",
                                            style={
                                                "textAlign": "center",
                                                "fontWeight": "bold",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        # html.Img(
                                        #     src=age,
                                        #     style={"width": "100%", "height": "120px", "objectFit": "contain", "marginBottom": "8px"},
                                        # ),
                                        dcc.Graph(
                                            figure=age_hist_fig,
                                            config={"displayModeBar": False},
                                            style={
                                                "height": "calc(100% - 40px)",
                                                "width": "100%",
                                                "margin": "0 auto",
                                            },
                                        ),
                                    ],
                                    style={
                                        "flex": "1 1 0",
                                        "minWidth": "0",
                                        "height": "340px",
                                        "boxShadow": "0 4px 8px rgba(0,0,0,0.2)",
                                        "borderRadius": "8px",
                                        "overflow": "hidden",
                                        "backgroundColor": "#ffffff",
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "justifyContent": "flex-start",
                                        "alignItems": "stretch",
                                        "margin": "10px",
                                        "padding": "10px",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexWrap": "nowrap",
                                "justifyContent": "space-between",
                                "alignItems": "stretch",
                                "width": "100%",
                                "marginBottom": "30px",
                                "marginTop": "20px",
                                "gap": "0",
                            },
                        ),
                    ],
                    open=True,
                ),
                # Tabs für verschiedene Abschnitte -----------------------------------------------------------------------------------------------------------
                dcc.Tabs(
                    [
                        dcc.Tab(
                            label="🩸 Blutwerte",
                            children=[
                                html.Div(
                                    [
                                        html.P(
                                            "Hämoglobin-Verteilung im Blut von Rauchern und Nichtrauchern",
                                            style={
                                                "textAlign": "center",
                                                "fontWeight": "bold",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        html.Img(
                                            src=hemoglobin_chart,
                                            style={
                                                "width": "100%",
                                                "height": "400px",
                                                "objectFit": "contain",
                                                "display": "block",
                                                "margin": "0 auto",
                                            },
                                        ),
                                    ],
                                    style={
                                        "width": "calc(100% - 20px)",
                                        "boxShadow": "0 4px 8px rgba(0,0,0,0.2)",
                                        "borderRadius": "8px",
                                        "overflow": "hidden",
                                        "backgroundColor": "#ffffff",
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "justifyContent": "flex-start",
                                        "alignItems": "stretch",
                                        "margin": "10px auto 30px auto",
                                        "padding": "20px",
                                    },
                                ),
                            ],
                            style={
                                "backgroundColor": "#f9f9f9",
                                "padding": "10px",
                                "fontSize": "16px",
                                "fontWeight": "500",
                                "color": "#555",
                                "height": "48px",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                            },
                            selected_style={
                                "backgroundColor": "#fde3e3",
                                "borderTop": "3px solid #e82626",
                                "fontSize": "16px",
                                "fontWeight": "500",
                                "color": "#e82626",
                                "height": "48px",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                            },
                        ),
                        dcc.Tab(
                            label="📊 Modellierung",
                            children=[
                                html.Div(
                                    [
                                        html.P(
                                            "Feature Importance - Random Forest",
                                            style={
                                                "textAlign": "center",
                                                "fontWeight": "bold",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        html.Img(
                                            src=FeatureImportance_RandomForest,
                                            style={
                                                "width": "100%",
                                                "height": "400px",
                                                "objectFit": "contain",
                                                "display": "block",
                                                "margin": "0 auto",
                                            },
                                        ),
                                    ],
                                    style={
                                        "width": "calc(100% - 20px)",
                                        "boxShadow": "0 4px 8px rgba(0,0,0,0.2)",
                                        "borderRadius": "8px",
                                        "overflow": "hidden",
                                        "backgroundColor": "#ffffff",
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "justifyContent": "flex-start",
                                        "alignItems": "stretch",
                                        "margin": "10px auto 20px auto",
                                        "padding": "20px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.P(
                                            "Decision Tree Model",
                                            style={
                                                "textAlign": "center",
                                                "fontWeight": "bold",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        html.Img(
                                            src=tree,
                                            style={
                                                "width": "100%",
                                                "height": "400px",
                                                "objectFit": "contain",
                                                "display": "block",
                                                "margin": "0 auto",
                                            },
                                        ),
                                    ],
                                    style={
                                        "width": "calc(100% - 20px)",
                                        "boxShadow": "0 4px 8px rgba(0,0,0,0.2)",
                                        "borderRadius": "8px",
                                        "overflow": "hidden",
                                        "backgroundColor": "#ffffff",
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "justifyContent": "flex-start",
                                        "alignItems": "stretch",
                                        "margin": "10px auto 30px auto",
                                        "padding": "20px",
                                    },
                                ),
                            ],
                            style={
                                "backgroundColor": "#f9f9f9",
                                "padding": "10px",
                                "fontSize": "16px",
                                "fontWeight": "500",
                                "color": "#555",
                                "height": "48px",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                            },
                            selected_style={
                                "backgroundColor": "#e4fde3",
                                "borderTop": "3px solid #66bb6a",
                                "fontSize": "16px",
                                "fontWeight": "500",
                                "color": "#2e7d32",
                                "height": "48px",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                            },
                        ),
                        dcc.Tab(
                            label="🖥️ Simulation",
                            children=[
                                html.Div(
                                    [
                                        html.H3(
                                            "Hier gerne Simulationen einfügen",
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "20px",
                                            },
                                        ),
                                    ],
                                    style={
                                        "width": "calc(100% - 20px)",
                                        "padding": "20px",
                                        "margin": "10px auto 30px auto",
                                    },
                                ),
                            ],
                            style={
                                "backgroundColor": "#f9f9f9",
                                "padding": "10px",
                                "fontSize": "16px",
                                "fontWeight": "500",
                                "color": "#555",
                                "height": "48px",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                            },
                            selected_style={
                                "backgroundColor": "#f1f8ff",
                                "borderTop": "3px solid #4c96df",
                                "fontSize": "16px",
                                "fontWeight": "500",
                                "color": "#4c96df",
                                "height": "48px",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                            },
                        ),
                    ],
                    style={
                        "marginTop": "20px",
                        "boxShadow": "0 2px 10px rgba(0,0,0,0.1)",
                        "borderRadius": "10px",
                        "overflow": "hidden",
                        "backgroundColor": "#fff",
                    },
                ),
            ],
            style={"padding": "0 30px"},
        ),

        # Footer -----------------------------------------------------------------------------------------------------------
        html.Div(
            [
                html.Hr(
                    style={
                        "margin": "0",
                        "border": "none",
                        "height": "1px",
                        "backgroundColor": "#4c96df",
                    }
                ),
                html.Div(
                    [
                        html.P(
                            "© 2025 Dashboard zur Raucher-Verifizierung",
                            style={
                                "margin": "4px 0",
                                "color": "#4c96df",
                                "fontSize": "0.85em",
                            },
                        ),
                        html.P(
                            "Data Science Portfolio, Gruppe 7",
                            style={
                                "margin": "0",
                                "fontSize": "0.8em",
                                "color": "#7f8c8d",
                            },
                        ),
                    ],
                    style={"textAlign": "center", "padding": "10px 0"},
                ),
            ],
            style={
                "backgroundColor": "#ffffff",
                "borderTop": "1px solid #e0e6ed",
                "boxShadow": "0 -2px 8px rgba(0,0,0,0.1)",
                "marginTop": "40px",
            },
        ),
    ],
    style={"fontFamily": "'Inter', sans-serif", "margin": "0", "padding": "0"},
)

# Dash-Anwendung ausführen -----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)  # Debug Modus für Ausgabe von Fehlermeldungen im Dashboard uvm
