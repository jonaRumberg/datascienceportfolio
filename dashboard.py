# Imports
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import plotly.express as px
import joblib
import io
import sys

# Dashboard im Browser anzeigen
# http://127.0.0.1:8050/

df = pd.read_csv("data/smoker_train.csv")


# load images
def encode_image(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"


tree = encode_image("images/dashboard_tree.jpeg")
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
    color_discrete_sequence=["#4c96df", "#e82626"],
)

# Hemoglobin-Verteilung nach Raucherstatus erstellen
def create_hemoglobin_chart():
    plt.figure(figsize=(10, 5))
    for status in df["smoking"].unique():
        subset = df[df["smoking"] == status]
        plt.hist(subset["hemoglobin"], bins=100, alpha=0.5, label=f"Smoking: {status}")
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

# Boxplot für Gewicht nach Raucherstatus erstellen
def create_weight_boxplot():
    plt.figure(figsize=(8, 5))
    df.boxplot(column='weight(kg)', by='smoking')
    plt.suptitle('')
    plt.xlabel('Raucherstatus (0 = Nichtraucher, 1 = Raucher)')
    plt.ylabel('Gewicht (kg)')
    
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{img_base64}"

hemoglobin_chart = create_hemoglobin_chart()
weight_boxplot_chart = create_weight_boxplot()


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
                            label="🩸💪 Blut- & Körperwerte",
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
                                        "margin": "10px auto 20px auto",
                                        "padding": "20px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.P(
                                            "Gewichtsverteilung: Vergleich zwischen Rauchern und Nichtrauchern",
                                            style={
                                                "textAlign": "center",
                                                "fontWeight": "bold",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        html.Img(
                                            src=weight_boxplot_chart,
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
                                            "Vorhersagegenauigkeit",
                                            style={
                                                "textAlign": "center",
                                                "fontWeight": "bold",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        html.P(
                                            "74%",
                                            style={
                                                "textAlign": "center",
                                                "fontSize": "56px",
                                                "fontWeight": "bold",
                                                "color": "#4c96df",
                                                "margin": "20px 0",
                                                "textShadow": "0 2px 4px rgba(39, 174, 96, 0.3)",
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
                                        "justifyContent": "center",
                                        "alignItems": "stretch",
                                        "margin": "10px auto 20px auto",
                                        "padding": "30px 20px",
                                        "minHeight": "150px",
                                    },
                                ),
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
                                            "Simulation der Raucher-Wahrscheinlichkeit",
                                            style={
                                                "textAlign": "center",
                                                "fontWeight": "bold",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        
                                        # Input Bereich
                                        html.Div(
                                            [
                                                html.H4(
                                                    "📁 CSV-Datei hochladen",
                                                    style={
                                                        "fontWeight": "bold",
                                                        "marginBottom": "10px",
                                                        "color": "#4c96df",
                                                    },
                                                ),
                                                dcc.Upload(
                                                    id='upload-csv',
                                                    children=html.Div([
                                                        html.P("CSV-Datei hier hineinziehen oder klicken zum Auswählen"),
                                                ]),
                                                    style={
                                                        'width': '100%',
                                                        'height': '100px',
                                                        'lineHeight': '60px',
                                                        'borderWidth': '2px',
                                                        'borderStyle': 'dashed',
                                                        'borderColor': '#4c96df',
                                                        'borderRadius': '8px',
                                                        'textAlign': 'center',
                                                        'backgroundColor': '#f8f9fa',
                                                        'cursor': 'pointer',
                                                        'marginBottom': '20px',
                                                        'display': 'flex',
                                                        'alignItems': 'center',
                                                        'justifyContent': 'center',
                                                        'padding': '10px',
                                                    },
                                                    multiple=False
                                                ),
                                            ],
                                            style={
                                                "backgroundColor": "#f8f9fa",
                                                "padding": "20px",
                                                "borderRadius": "8px",
                                                "marginBottom": "20px",
                                                "border": "1px solid #e9ecef",
                                            }
                                        ),
                                        
                                        # Output Bereich
                                        html.Div(
                                            [
                                                html.H4(
                                                    "Einschätzung",
                                                    style={
                                                        "fontSize": "20px",
                                                        "fontWeight": "bold",
                                                        "marginBottom": "20px",
                                                        "color": "#4c96df",
                                                        "textAlign": "center",
                                                    },
                                                ),
                                                html.Div(
                                                    id="prediction-output",
                                                    children=[
                                                        html.P(
                                                            "Laden Sie eine CSV-Datei hoch, um eine Vorhersage zu erhalten.",
                                                            style={
                                                                "textAlign": "center",
                                                                "color": "#7f8c8d",
                                                                "fontSize": "16px",
                                                            }
                                                        )
                                                    ],
                                                    style={
                                                        "minHeight": "100px",
                                                        "display": "flex",
                                                        "flexDirection": "column",
                                                        "alignItems": "stretch",
                                                        "justifyContent": "flex-start",
                                                        "gap": "0",
                                                    }
                                                )
                                            ],
                                            style={
                                                "backgroundColor": "#ffffff",
                                                "padding": "30px",
                                                "borderRadius": "8px",
                                                "border": "1px solid #e9ecef",
                                                "textAlign": "center",
                                            }
                                        ),
                                    ],
                                    style={
                                        "width": "calc(100% - 20px)",
                                        "boxShadow": "0 4px 8px rgba(0,0,0,0.2)",
                                        "borderRadius": "8px",
                                        "overflow": "hidden",
                                        "backgroundColor": "#ffffff",
                                        "margin": "10px auto 30px auto",
                                        "padding": "30px",
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

# Modell laden (nur einmal beim Start des Dashboards)
rfc_model = joblib.load('smoker_rfc_model.joblib')

# Callback für CSV-Upload und Vorhersage hinzufügen (nach dem Layout)
@app.callback(
    Output('prediction-output', 'children'),
    Input('upload-csv', 'contents'),
    State('upload-csv', 'filename')
)
def update_prediction(contents, filename):
    if contents is None:
        return html.P(
            "Laden Sie eine CSV-Datei hoch, um eine Vorhersage zu erhalten.",
            style={
                "textAlign": "center",
                "color": "#7f8c8d",
                "fontSize": "16px",
            }
        )
    
    try:
        print("try startet", file=sys.stderr)
        # CSV-Datei verarbeiten
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # CSV in DataFrame laden
        df_uploaded = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # Vorhersagen berechnen
        pred_probs = rfc_model.predict_proba(df_uploaded)[:,1]
        prediction_probability = round(pred_probs.mean() * 100, 2)
        
        # Datei-Informationen anzeigen
        file_info = [
            # Erste Kachel: Raucher-Wahrscheinlichkeit
            html.Div([
                html.P(
                    f"🚬 Raucher-Wahrscheinlichkeit 🚬",
                    style={
                        "textAlign": "center",
                        "fontWeight": "bold",
                        "marginBottom": "10px",
                        "color": "#4c96df",
                        "fontSize": "18px",
                    }
                ),
                html.P(
                    f"{prediction_probability}%",
                    style={
                        "textAlign": "center",
                        "fontSize": "48px",
                        "fontWeight": "bold",
                        "color": "#e82626" if prediction_probability > 50 else "#27ae60",
                        "margin": "10px 0",
                        "textShadow": "0 2px 4px rgba(0,0,0,0.2)",
                    }
                ),
                html.P(
                    f"{'⚠️ Hohe Raucher-Wahrscheinlichkeit' if prediction_probability > 50 else '✅ Niedrige Raucher-Wahrscheinlichkeit'}",
                    style={
                        "textAlign": "center",
                        "color": "#e82626" if prediction_probability > 50 else "#27ae60",
                        "fontSize": "16px",
                        "fontWeight": "bold",
                        "marginTop": "10px",
                    }
                )
            ], style={
                "backgroundColor": "#ffffff",
                "padding": "30px",
                "borderRadius": "8px",
                "marginBottom": "10px",
                "border": "1px solid #e9ecef",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
                "flex": "1",
                "minHeight": "200px",
                "display": "flex",
                "flexDirection": "column",
                "justifyContent": "center",
            }),
            
            # Zweite Kachel: Datenvorschau mit Datei-Informationen
            html.Div([
                html.H5(
                    f"Datenvorschau: {filename}",
                    style={
                        "color": "#4c96df",
                        "fontSize": "18px",
                        "fontWeight": "bold",
                        "marginBottom": "15px",
                        "textAlign": "center"
                    }
                ),
                html.Div([
                    html.P(f"Anzahl Zeilen: {len(df_uploaded)} | Anzahl Spalten: {len(df_uploaded.columns)}", 
                           style={"margin": "10px 0", "fontWeight": "bold", "textAlign": "center"}),
                ], style={
                    "backgroundColor": "#f8f9fa",
                    "padding": "10px",
                    "borderRadius": "8px",
                    "border": "1px solid #e9ecef",
                    "marginBottom": "15px"
                }),
                html.Div([
                    html.Table([
                        html.Thead([
                            html.Tr([html.Th(col, style={"padding": "8px", "backgroundColor": "#f1f8ff", "fontWeight": "bold"}) for col in df_uploaded.columns[:10]])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(str(df_uploaded.iloc[i][col]), style={"padding": "8px", "borderBottom": "1px solid #dee2e6"}) 
                                for col in df_uploaded.columns[:10]
                            ]) for i in range(min(5, len(df_uploaded)))
                        ])
                    ], style={
                        "width": "100%",
                        "borderCollapse": "collapse",
                        "border": "1px solid #dee2e6",
                        "fontSize": "14px"
                    })
                ], style={"overflowX": "auto"})
            ], style={
                "backgroundColor": "#ffffff",
                "padding": "20px",
                "borderRadius": "8px",
                "marginBottom": "0px",
                "border": "1px solid #e9ecef",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
                "flex": "1",
                "minHeight": "200px",
                "display": "flex",
                "flexDirection": "column",
                "justifyContent": "flex-start",
            })
        ]
        
        return file_info
    
    except Exception as e:
        return [
            html.Div([
                html.P(
                    f"❌ Fehler beim Verarbeiten der Datei:",
                    style={
                        "textAlign": "center",
                        "color": "#e74c3c",
                        "fontSize": "18px",
                        "fontWeight": "bold",
                        "marginBottom": "10px"
                    }
                ),
                html.P(
                    f"Datei: {filename}",
                    style={
                        "textAlign": "center",
                        "color": "#7f8c8d",
                        "fontSize": "14px",
                        "marginBottom": "10px"
                    }
                ),
                html.P(
                    f"Fehlerdetails: {str(e)}",
                    style={
                        "textAlign": "center",
                        "color": "#e74c3c",
                        "fontSize": "14px",
                        "backgroundColor": "#ffeaea",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "border": "1px solid #f5c6cb"
                    }
                )
            ])
        ]

# Zusätzlicher Callback für Upload-Status
@app.callback(
    Output('upload-csv', 'children'),
    Input('upload-csv', 'contents'),
    State('upload-csv', 'filename')
)
def update_upload_status(contents, filename):
    if contents is None:
        return html.Div([
            html.P("CSV-Datei hier hineinziehen oder klicken zum Auswählen"),
        ], style={"textAlign": "center", "margin": "0"})
    else:
        return html.Div([
            html.Div([
                html.Span("✅", style={"fontSize": "24px", "marginRight": "10px", "color": "#27ae60"}),
                html.Span(f"Datei hochgeladen: {filename}", style={"fontWeight": "bold", "color": "#27ae60"}),
            ], style={
                "display": "flex", 
                "alignItems": "center", 
                "justifyContent": "center",
                "marginBottom": "8px"
            }),
            html.P("Klicken Sie hier, um eine neue Datei hochzuladen", 
                   style={
                       "fontSize": "12px", 
                       "color": "#7f8c8d", 
                       "margin": "0",
                       "textAlign": "center"
                   })
        ], style={"display": "flex", "flexDirection": "column", "justifyContent": "center", "height": "100%"})

# Dash-Anwendung ausführen -----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)  # Debug Modus für Ausgabe von Fehlermeldungen im Dashboard uvm
