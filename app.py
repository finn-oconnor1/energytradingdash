import pandas as pd
import numpy as np
import re
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import dash
from dash import dash_table

# ========== Load Data ==========
elec_prices = pd.read_csv("electricity-prices-forwa.csv")
gas_prices = pd.read_csv("gas-prices-forward-deliv.csv")
elec_vol = pd.read_csv("electricity-trading-volu.csv")
gas_vol = pd.read_csv("gas-trading-volumes-and.csv")
elec_gen = pd.read_csv("electricity-generation-m.csv")
gas_demand = pd.read_csv("gas-demand-and-supply-so.csv")

# ========== Date Parsing ==========
def parse_dates(df):
    first_col = df.columns[0]
    sample = str(df[first_col].iloc[0])
    if re.match(r"Q[1-4]\s+\d{4}", sample):
        def q_to_date(q_str):
            q, year = q_str.split()
            quarter = int(q[1])
            return pd.Timestamp(year=int(year), month=(quarter-1)*3+1, day=1)
        df[first_col] = df[first_col].apply(q_to_date)
    else:
        df[first_col] = pd.to_datetime(df[first_col], errors='coerce')
    return df

datasets = [elec_prices, gas_prices, elec_vol, gas_vol, elec_gen, gas_demand]
datasets = [parse_dates(df) for df in datasets]
elec_prices, gas_prices, elec_vol, gas_vol, elec_gen, gas_demand = datasets

dataset_dict = {
    "Electricity Prices": elec_prices,
    "Gas Prices": gas_prices,
    "Electricity Trading Volumes": elec_vol,
    "Gas Trading Volumes": gas_vol,
    "Electricity Generation": elec_gen,
    "Gas Demand/Supply": gas_demand
}

# ========== Trading Simulation ==========
def run_strategy(buy_thresh, sell_thresh, position_pct):
    df = pd.merge(elec_prices, gas_prices,
                  left_on=elec_prices.columns[0],
                  right_on=gas_prices.columns[0],
                  suffixes=('_elec', '_gas'))
    df.rename(columns={df.columns[0]: 'Date', df.columns[1]: 'ElecPrice', df.columns[2]: 'GasPrice'}, inplace=True)
    df['Spread'] = df['ElecPrice'] - df['GasPrice']

    capital = 100000
    position_size = capital * (position_pct / 100)
    balance = capital
    trades = []

    for i in range(1, len(df)):
        spread = df.loc[i, 'Spread']
        prev_spread = df.loc[i-1, 'Spread']
        pnl = 0
        position = 0

        if spread > buy_thresh:
            pnl = (spread - prev_spread) * (position_size / prev_spread)
            position = 1
        elif spread < sell_thresh:
            pnl = (prev_spread - spread) * (position_size / prev_spread)
            position = -1

        balance += pnl
        trades.append({
            'Date': df.loc[i, 'Date'],
            'ElecPrice': df.loc[i, 'ElecPrice'],
            'GasPrice': df.loc[i, 'GasPrice'],
            'Spread': spread,
            'Position': position,
            'PnL': pnl,
            'Balance': balance
        })

    trades_df = pd.DataFrame(trades)

    # Stats
    total_pnl = trades_df['PnL'].sum()
    return_pct = (trades_df['Balance'].iloc[-1] - capital) / capital * 100
    wins = trades_df[trades_df['PnL'] > 0].shape[0]
    win_rate = wins / trades_df.shape[0] * 100 if trades_df.shape[0] > 0 else 0
    drawdown = (trades_df['Balance'].cummax() - trades_df['Balance']).max()
    sharpe = trades_df['PnL'].mean() / trades_df['PnL'].std() * np.sqrt(52) if trades_df['PnL'].std() != 0 else 0

    stats = {
        'Total P&L (€)': f"{total_pnl:,.2f}",
        'Return (%)': f"{return_pct:.2f}",
        'Win Rate (%)': f"{win_rate:.2f}",
        'Max Drawdown (€)': f"{drawdown:,.2f}",
        'Sharpe Ratio': f"{sharpe:.2f}"
    }

    return trades_df, stats

# ========== Dash App ==========
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Energy Trading Dashboard"

# ========== Layout ==========
app.layout = dbc.Container([
    html.H1("Energy Trading Dashboard by Finn O'Connor", className="text-center my-3"),

    dcc.Tabs(id="tabs", value='tab0', children=[
        dcc.Tab(label='About', value='tab0'),
        dcc.Tab(label='Market Overview', value='tab1'),
        dcc.Tab(label='Trading Simulation', value='tab2'),
        dcc.Tab(label='Data Explorer', value='tab3'),
    ]),

    html.Div(id='tabs-content')
], fluid=True)

# ========== Tabs Callback ==========
@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):
    if tab == 'tab0':
        return dbc.Container([
            html.H3("About This Dashboard"),
            html.P("This dashboard provides a historical UK energy market simulation and analysis tool. "
                   "It integrates electricity and gas prices, trading volumes, and supply/demand data to "
                   "allow interactive exploration of market dynamics."),
            html.P("The Trading Simulation tab lets you backtest a simple spark spread strategy using real "
                   "historical prices. You can adjust buy/sell thresholds and position sizing to see how "
                   "your strategy would have performed with €100,000 starting capital. "
                   "Based on the initial €100,000 investment, the simulation calculates key performance metrics such as total P&L, return percentage, win rate, max drawdown, and Sharpe ratio. "
                   "The 'Buy Threshold' and 'Sell Threshold' sliders allow you to set the conditions for entering and exiting trades, while the 'Position Size (%)' slider determines how much of your capital is allocated to each trade. "
                   "The thresholds are defined from 0-50 (in € per MWh). If the the spark spread is greater than the buy threshold then a buy trade is executed and if the spark spread is less than the sell threshold then a sell trade is executed. "),
            html.P("The Market Overview tab visualizes historical prices, volumes, generation mix, and demand using the data from 'ofgem.gov.uk'."),
            html.P("The Data Explorer tab provides direct access to the historical datasets I pulled from online with filtering and export."),
            html.P("This dashboard is built using Dash, Plotly, and Bootstrap components for a responsive and accessible design."),
            html.P("As a disclaimer, this is a simulation tool and not financial advice. I also will acknowledge that I used AI to help me build the Plotly dashboard as it is a tool I am still only learning."),
        ], className="my-4")

    elif tab == 'tab1':
        return html.Div([
            html.H3("Electricity Prices"),
            dcc.Graph(figure=go.Figure(data=[go.Scatter(x=elec_prices.iloc[:,0], y=elec_prices.iloc[:,1], mode='lines')])),

            html.H3("Gas Prices"),
            dcc.Graph(figure=go.Figure(data=[go.Scatter(x=gas_prices.iloc[:,0], y=gas_prices.iloc[:,1], mode='lines')])),

            html.H3("Electricity Trading Volumes"),
            dcc.Graph(figure=go.Figure(data=[go.Scatter(x=elec_vol.iloc[:,0], y=elec_vol.iloc[:,1], mode='lines')])),

            html.H3("Gas Trading Volumes"),
            dcc.Graph(figure=go.Figure(data=[go.Scatter(x=gas_vol.iloc[:,0], y=gas_vol.iloc[:,1], mode='lines')])),

            html.H3("Electricity Generation vs Demand"),
            dcc.Graph(figure=go.Figure(data=[go.Scatter(x=elec_gen.iloc[:,0], y=elec_gen.iloc[:,1], mode='lines')])),

            html.H3("Gas Demand vs Supply"),
            dcc.Graph(figure=go.Figure(data=[go.Scatter(x=gas_demand.iloc[:,0], y=gas_demand.iloc[:,1], mode='lines')]))
        ])

    elif tab == 'tab2':
        return html.Div([
            html.Div(id='stats-panel', className='d-flex justify-content-around my-3'),

            dbc.Row([
                dbc.Col([
                    html.Label("Buy Threshold"),
                    dcc.Slider(id='buy-threshold', min=-50, max=50, step=1, value=10, marks=None, tooltip={"placement": "bottom"})
                ]),
                dbc.Col([
                    html.Label("Sell Threshold"),
                    dcc.Slider(id='sell-threshold', min=-50, max=50, step=1, value=-10, marks=None, tooltip={"placement": "bottom"})
                ]),
                dbc.Col([
                    html.Label("Position Size (%)"),
                    dcc.Slider(id='position-size', min=10, max=100, step=5, value=50, marks=None, tooltip={"placement": "bottom"})
                ])
            ], className='my-3'),

            dcc.Graph(id='price-chart'),
            dcc.Graph(id='pnl-chart'),

            html.A("Download Trade History", id="download-link", download="trade_history.csv", href="", target="_blank")
        ])

    elif tab == 'tab3':
        return html.Div([
            html.H3("Data Explorer"),
            html.Label("Select Dataset:"),
            dcc.Dropdown(
                id='dataset-select',
                options=[{'label': k, 'value': k} for k in dataset_dict.keys()],
                value='Electricity Prices',
                clearable=False
            ),
            html.Br(),
            html.Label("Select Date Range:"),
            dcc.DatePickerRange(id='date-range'),
            html.Br(), html.Br(),
            html.A("Download Current Dataset", id="data-download", download="dataset.csv", href="", target="_blank"),
            html.Br(), html.Br(),
            dash_table.DataTable(
                id='data-table',
                page_size=20,
                style_table={'overflowX': 'auto'},
                filter_action="native",
                sort_action="native",
                style_cell={'textAlign': 'left', 'padding': '5px'}
            )
        ])

# ========== Data Explorer Callback ==========
@app.callback(
    [Output('data-table', 'data'),
     Output('data-table', 'columns'),
     Output('data-download', 'href'),
     Output('date-range', 'min_date_allowed'),
     Output('date-range', 'max_date_allowed')],
    [Input('dataset-select', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_data_table(dataset_name, start_date, end_date):
    df = dataset_dict[dataset_name].copy()
    date_col = df.columns[0]

    # Apply date filter if selected
    if start_date and end_date:
        mask = (df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))
        df = df.loc[mask]

    data = df.to_dict('records')
    columns = [{"name": i, "id": i} for i in df.columns]

    # CSV download link
    csv_string = df.to_csv(index=False, encoding='utf-8')
    csv_href = "data:text/csv;charset=utf-8," + csv_string

    return data, columns, csv_href, df[date_col].min(), df[date_col].max()

# ========== Trading Simulation Callback ==========
@app.callback(
    [Output('price-chart', 'figure'),
     Output('pnl-chart', 'figure'),
     Output('stats-panel', 'children'),
     Output('download-link', 'href')],
    [Input('buy-threshold', 'value'),
     Input('sell-threshold', 'value'),
     Input('position-size', 'value')]
)
def update_simulation(buy_thresh, sell_thresh, position_pct):
    trades_df, stats = run_strategy(buy_thresh, sell_thresh, position_pct)

    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=trades_df['Date'], y=trades_df['ElecPrice'], name="Electricity Price"))
    price_fig.add_trace(go.Scatter(x=trades_df['Date'], y=trades_df['GasPrice'], name="Gas Price"))
    price_fig.add_trace(go.Scatter(x=trades_df['Date'], y=trades_df['Spread'], name="Spark Spread", line=dict(dash='dot')))

    pnl_fig = go.Figure()
    pnl_fig.add_trace(go.Scatter(x=trades_df['Date'], y=trades_df['Balance'], name="Account Balance"))

    stats_cards = []
    for key, value in stats.items():
        stats_cards.append(
            dbc.Card(dbc.CardBody([html.H4(value, className="card-title"), html.P(key, className="card-text")]), className="mx-2")
        )

    csv_string = trades_df.to_csv(index=False, encoding='utf-8')
    csv_href = "data:text/csv;charset=utf-8," + csv_string

    return price_fig, pnl_fig, stats_cards, csv_href

# ========== Run ==========
if __name__ == '__main__':
    app.run(debug=True)
