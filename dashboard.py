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


# Dash-Anwendung initialisieren
app = dash.Dash(__name__)

# # Layout des Dash-Dashboards definieren
# app.layout = html.Div([
#     html.H1("Smoker prediction", style={'textAlign': 'center'}),
#     html.Details([
#         html.Summary("Show/Hide Participant Data"),
#         html.Div([
#             html.Img(src=weight_height, style={'width': '400px', 'height': 'auto', 'marginRight': '20px'}),
#             html.Img(src=smoker, style={'width': '400px', 'height': 'auto', 'marginRight': '20px'}),
#             html.Img(src=age, style={'width': '400px', 'height': 'auto'})
#         ], style={'display': 'flex', 'justifyContent': 'center', 'gap': '20px', 'marginBottom': '30px'})
#     ]),
#     dcc.Tabs([
#         dcc.Tab(label='Blood Values', children=[
#             html.Div([
#             ])
#         ]),
#         dcc.Tab(label='Modeling', children=[
#             html.Div([
#                 html.H3("Modeling"),
#                 html.Img(src=tree, style={'width': '600px', 'height': 'auto'})
#             ])
#         ])
#     ])
# ])


# Layout des Dash-Dashboards definieren
app.layout = html.Div(
    [
        # Header
        html.Div(
            [
                # Oberer Bereich mit Logo und Kassen-Identifikation
                html.Div(
                    [
                        # Linke Seite: Logo und Name
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
                        # Rechte Seite: Interne Dashboard-Kennzeichnung
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
        # Informationsleiste (optional)
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
        
        # Main Content
        html.Div(
            [
                # Übersicht über die Teilnehmerdaten
                html.Details(
                    [
                        html.Summary(
                            "Show/Hide Participant Data",
                            style={
                                "cursor": "pointer",
                                "fontWeight": "bold",
                                "padding": "10px",
                                "backgroundColor": "#ededed",
                                "borderRadius": "5px",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.P(
                                            "Weight vs Height Distribution",
                                            style={
                                                "textAlign": "center",
                                                "fontWeight": "bold",
                                            },
                                        ),
                                        html.Img(
                                            src=weight_height,
                                            style={"width": "100%", "height": "auto"},
                                        ),
                                    ],
                                    style={
                                        "width": "30%",
                                        "margin": "10px",
                                        "boxShadow": "0 4px 8px rgba(0,0,0,0.2)",
                                        "borderRadius": "8px",
                                        "overflow": "hidden",
                                        "padding": "10px",
                                        "backgroundColor": "#ffffff",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.P(
                                            "Smoker Distribution",
                                            style={
                                                "textAlign": "center",
                                                "fontWeight": "bold",
                                            },
                                        ),
                                        html.Img(
                                            src=smoker,
                                            style={"width": "100%", "height": "auto"},
                                        ),
                                    ],
                                    style={
                                        "width": "30%",
                                        "margin": "10px",
                                        "boxShadow": "0 4px 8px rgba(0,0,0,0.2)",
                                        "borderRadius": "8px",
                                        "overflow": "hidden",
                                        "padding": "10px",
                                        "backgroundColor": "#ffffff",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.P(
                                            "Age Distribution",
                                            style={
                                                "textAlign": "center",
                                                "fontWeight": "bold",
                                            },
                                        ),
                                        html.Img(
                                            src=age,
                                            style={"width": "100%", "height": "auto"},
                                        ),
                                    ],
                                    style={
                                        "width": "30%",
                                        "margin": "10px",
                                        "boxShadow": "0 4px 8px rgba(0,0,0,0.2)",
                                        "borderRadius": "8px",
                                        "overflow": "hidden",
                                        "padding": "10px",
                                        "backgroundColor": "#ffffff",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexWrap": "wrap",
                                "justifyContent": "center",
                                "marginBottom": "30px",
                                "marginTop": "20px",
                            },
                        ),
                    ],
                    open=True,
                ),
                
                # Tabs für verschiedene Abschnitte
                dcc.Tabs(
                    [
                        dcc.Tab(
                            label="🩸 Blood Values",
                            children=[
                                html.Div(
                                    [
                                        html.H3(
                                            "Blood Value Analysis",
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "20px",
                                            },
                                        ),
                                        html.P(
                                            "This section contains analysis of blood-related parameters.",
                                            style={
                                                "textAlign": "center",
                                                "color": "#666",
                                            },
                                        ),
                                        html.Div(
                                            "Weitere Blutwertanalysen werden hier angezeigt.",
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "40px",
                                                "color": "#aaa",
                                                "fontStyle": "italic",
                                            },
                                        ),
                                    ],
                                    style={"padding": "20px"},
                                ),
                            ],
                            style={
                                "backgroundColor": "#f9f9f9",
                                "padding": "10px",
                                "fontSize": "16px",
                                "fontWeight": "500",
                                "color": "#555",
                                "height": "48px",  # Einheitliche Höhe
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                            },
                            selected_style={
                                "backgroundColor": "#fde3e3",
                                "borderTop": "3px solid #c01515",
                                "fontSize": "16px",
                                "fontWeight": "500",
                                "color": "#c01515",
                                "height": "48px",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                            },
                        ),
                        dcc.Tab(
                            label="📊 Modeling",
                            children=[
                                html.Div(
                                    [
                                        html.H3(
                                            "Decision Tree Model",
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "20px",
                                            },
                                        ),
                                        html.P(
                                            "Visual representation of the decision tree used for prediction.",
                                            style={
                                                "textAlign": "center",
                                                "color": "#666",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Img(
                                                    src=tree,
                                                    style={
                                                        "width": "600px",
                                                        "height": "auto",
                                                    },
                                                )
                                            ],
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "20px",
                                            },
                                        ),
                                    ],
                                    style={"padding": "20px"},
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
        
        # Footer
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
                            "© 2025 Smoker Prediction Dashboard",
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

# Dash-Anwendung ausführen
if __name__ == "__main__":
    app.run(debug=True)  # Debug Modus für Ausgabe von Fehlermeldungen im Dashboard uvm
