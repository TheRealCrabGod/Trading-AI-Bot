import os
from dotenv import load_dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from alpaca_environment import AlpacaTradingEnvironment
import sys
import logging
import threading
import time
import numpy as np
import webbrowser
import signal
import tkinter as tk
from tkinter import messagebox
import matplotlib
matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('trading.log'),
                        logging.StreamHandler()
                    ])

# Global model reference for saving
global_model = None
global_model_path = os.path.join("models", "grokkkk_alpaca_live.zip")
save_thread_running = False

def signal_handler(sig, frame):
    global global_model, save_thread_running
    logging.info("Ctrl+C detected, saving model and exiting...")
    save_thread_running = False
    if global_model is not None:
        try:
            global_model.save(global_model_path)
            logging.info(f"Model saved to {global_model_path} before exit")
        except Exception as e:
            logging.error(f"Error saving model on exit: {e}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def save_model_thread():
    global global_model, save_thread_running
    while save_thread_running:
        if global_model is not None:
            try:
                global_model.save(global_model_path)
                logging.info(f"Background model save to {global_model_path} at {time.time()}")
            except Exception as e:
                logging.error(f"Error in background model save: {e}")
        time.sleep(300)  # Save every 5 minutes

class SaveOnStepCallback(BaseCallback):
    def __init__(self, save_path, save_interval=100000, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_interval = save_interval
        self.step_count = 0

    def _on_step(self):
        self.step_count += 1
        if self.step_count % self.save_interval == 0:
            try:
                self.model.save(self.save_path)
                logging.info(f"Periodic model save during training to {self.save_path} at step {self.step_count}")
            except Exception as e:
                logging.error(f"Error during periodic training save: {e}")
        return True

class TradingApp:
    def __init__(self, symbol='SPY', window_size=50, initial_balance=10000, interval='1Min'):
        global global_model, save_thread_running
        logging.info("TradingApp.__init__ started")
        self.symbol = symbol
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.interval = interval
        self.running = False
        self.app_running = True
        self.live_mode = False  # Default to historical
        self.fig_lock = threading.Lock()
        self.model = None
        self.step_count = 0  # Continuous step counter
        self.run_number = 0  # Track historical runs

        # Initialize environment (default historical)
        self.env = None
        self.initialize_environment(live_mode=False)

        # Initialize PPO model
        self.model_path = global_model_path
        try:
            self.model = PPO(
                "MlpPolicy",
                self.env,
                verbose=0,
                learning_rate=0.0002,
                gamma=0.99,
                n_steps=5,
                ent_coef=0.01,
                batch_size=5
            )
            global_model = self.model
            if os.path.exists(self.model_path):
                logging.info(f"Loading model from {self.model_path}...")
                self.model = PPO.load(self.model_path, env=self.env)
                global_model = self.model
                logging.info("Model loaded successfully")
            else:
                logging.info("Training a new model...")
                save_thread_running = True
                threading.Thread(target=save_model_thread, daemon=True).start()
                callback = SaveOnStepCallback(save_path=self.model_path)
                try:
                    self.model.learn(total_timesteps=1000000, callback=callback)
                    self.model.save(self.model_path)
                    logging.info(f"Model trained and saved to {self.model_path}")
                except KeyboardInterrupt:
                    logging.info("Training interrupted, saving model...")
                    self.model.save(self.model_path)
                    logging.info(f"Model saved to {self.model_path} on interrupt")
                    save_thread_running = False
                    sys.exit(0)
                save_thread_running = False
        except Exception as e:
            logging.error(f"Error initializing/loading PPO model: {e}")
            save_thread_running = False
            sys.exit(1)

        # Initialize Plotly figure
        logging.info("Creating Plotly figure...")
        self.fig = make_subplots(specs=[[{"secondary_y": True}]])
        self.fig.update_layout(
            title="SPY Trading: Price and Portfolio Profit",
            xaxis_title="Time Step",
            yaxis_title="SPY Price",
            yaxis2_title="Portfolio Profit",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template='plotly_dark',
            annotations=[
                dict(
                    x=1,
                    y=1.1,
                    xref="paper",
                    yref="paper",
                    text="Total Profit: $0.00<br>Sharpe Ratio: 0.00",
                    showarrow=False,
                    font=dict(size=12, color="white"),
                    xanchor="right",
                    yanchor="top"
                )
            ]
        )

        # Use deque without maxlen to store all data
        self.prices_history = deque()
        self.actions_history = deque()
        self.valid_actions_history = deque()
        self.portfolio_history = deque()
        self.time_steps = deque()

        # Initialize with current data
        try:
            self.obs, info = self.env.reset()
            current_price = info.get('current_price', self.env.historical_data[self.env.current_step])
            self.prices_history.append(current_price)
            self.actions_history.append(1)
            self.valid_actions_history.append(1)
            self.portfolio_history.append(info['portfolio_value'])
            self.time_steps.append(self.step_count)
            self.step_count += 1
            logging.info(f"Initial info: {info}")
        except Exception as e:
            logging.error(f"Error resetting environment: {e}")
            sys.exit(1)

        # Initialize traces
        self.price_trace = go.Scatter(x=list(self.time_steps), y=list(self.prices_history), name="SPY Price", line=dict(color="lightblue"))
        self.profit_trace = go.Scatter(x=list(self.time_steps), y=[v - self.initial_balance for v in self.portfolio_history], 
                                     name="Portfolio Profit", line=dict(color="orange"))
        self.buy_trace = go.Scatter(x=[], y=[], mode='markers', name="Buy",
                                   marker=dict(symbol="triangle-up", size=10, color="green"))
        self.sell_trace = go.Scatter(x=[], y=[], mode='markers', name="Sell",
                                    marker=dict(symbol="triangle-down", size=10, color="red"))

        self.fig.add_trace(self.price_trace)
        self.fig.add_trace(self.profit_trace, secondary_y=True)
        self.fig.add_trace(self.buy_trace)
        self.fig.add_trace(self.sell_trace)

        # Save initial plot
        self.plot_filename = "trading_plot_0.html"
        try:
            with self.fig_lock:
                self.fig.write_html(self.plot_filename, auto_open=False)
        except Exception as e:
            logging.error(f"Error saving initial plot: {e}")

        # Initialize Tkinter GUI
        try:
            logging.info("Starting Tkinter GUI...")
            self.root = tk.Tk()
            self.root.title("SPY Trading Bot")
            self.root.geometry("300x250")
            self.start_live_button = tk.Button(self.root, text="Start Live Trading", command=self.start_live_trading)
            self.start_live_button.pack(pady=10)
            self.start_historical_button = tk.Button(self.root, text="Start Historical Trading", command=self.start_historical_trading)
            self.start_historical_button.pack(pady=10)
            self.stop_button = tk.Button(self.root, text="Stop Trading", command=self.stop_trading)
            self.stop_button.pack(pady=10)
            self.graph_button = tk.Button(self.root, text="Show/Update Graph", command=self.show_graph)
            self.graph_button.pack(pady=10)
            self.exit_button = tk.Button(self.root, text="Exit", command=self.exit_app)
            self.exit_button.pack(pady=10)
            self.use_gui = True
            logging.info("Tkinter GUI initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Tkinter GUI: {e}")
            self.use_gui = False
            self.root = None

        logging.info("Initialization complete, entering control mode...")

    def initialize_environment(self, live_mode):
        try:
            logging.info(f"Initializing environment with live_mode={live_mode}...")
            self.env = AlpacaTradingEnvironment(symbol=self.symbol, window_size=self.window_size,
                                               initial_balance=self.initial_balance, interval=self.interval,
                                               live_mode=live_mode)
            logging.info("Environment initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize environment: {e}")
            sys.exit(1)

    def show_graph(self):
        try:
            with self.fig_lock:
                time_steps = list(self.time_steps)
                prices = list(self.prices_history)
                actions = list(self.valid_actions_history)
                portfolio_values = list(self.portfolio_history)
                
                # Validate list lengths
                min_length = min(len(time_steps), len(prices), len(actions), len(portfolio_values))
                if min_length == 0:
                    logging.info("show_graph: no data to plot")
                    messagebox.showwarning("Warning", "No data to plot")
                    return
                if min_length < len(time_steps):
                    logging.warning(f"Truncating data to min length {min_length}: "
                                  f"time_steps={len(time_steps)}, prices={len(prices)}, "
                                  f"actions={len(actions)}, portfolio_values={len(portfolio_values)}")
                    time_steps = time_steps[:min_length]
                    prices = prices[:min_length]
                    actions = actions[:min_length]
                    portfolio_values = portfolio_values[:min_length]

                total_profit = portfolio_values[-1] - self.initial_balance if portfolio_values else 0.0
                returns = [portfolio_values[i] / portfolio_values[i-1] - 1 for i in range(1, len(portfolio_values))]
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252) if returns else 0.0

                # Calculate buy/sell markers
                buy_times = []
                buy_prices = []
                sell_times = []
                sell_prices = []
                for i, action in enumerate(actions):
                    if i < len(time_steps) and i < len(prices):
                        if action == 0:
                            buy_times.append(time_steps[i])
                            buy_prices.append(prices[i])
                        elif action == 2:
                            sell_times.append(time_steps[i])
                            sell_prices.append(prices[i])

                # Downsample if too many points
                if len(time_steps) > 10000:
                    step = max(1, len(time_steps) // 10000)
                    indices = list(range(0, len(time_steps), step))
                    time_steps = [time_steps[i] for i in indices]
                    prices = [prices[i] for i in indices]
                    portfolio_values = [portfolio_values[i] for i in indices]
                    # Filter buy/sell markers to downsampled indices
                    buy_times_ds = [time_steps[i] for i, t in enumerate(time_steps) if t in buy_times]
                    buy_prices_ds = [prices[i] for i, t in enumerate(time_steps) if t in buy_times]
                    sell_times_ds = [time_steps[i] for i, t in enumerate(time_steps) if t in sell_times]
                    sell_prices_ds = [prices[i] for i, t in enumerate(time_steps) if t in sell_times]
                    buy_times = buy_times_ds
                    buy_prices = buy_prices_ds
                    sell_times = sell_times_ds
                    sell_prices = sell_prices_ds

                # Clear existing traces and re-add
                self.fig.data = []
                self.fig.add_trace(go.Scatter(x=time_steps, y=prices, name="SPY Price", line=dict(color="lightblue")))
                self.fig.add_trace(go.Scatter(x=time_steps, y=[v - self.initial_balance for v in portfolio_values], 
                                            name="Portfolio Profit", line=dict(color="orange")), secondary_y=True)
                self.fig.add_trace(go.Scatter(x=buy_times, y=buy_prices, mode='markers', name="Buy",
                                            marker=dict(symbol="triangle-up", size=10, color="green")))
                self.fig.add_trace(go.Scatter(x=sell_times, y=sell_prices, mode='markers', name="Sell",
                                            marker=dict(symbol="triangle-down", size=10, color="red")))

                self.fig.update_layout(
                    annotations=[
                        dict(
                            x=1,
                            y=1.1,
                            xref="paper",
                            yref="paper",
                            text=f"Total Profit: ${total_profit:.2f}<br>Sharpe Ratio: {sharpe_ratio:.2f}",
                            showarrow=False,
                            font=dict(size=12, color="white"),
                            xanchor="right",
                            yanchor="top"
                        )
                    ]
                )

                self.fig.write_html(self.plot_filename, auto_open=False)
                webbrowser.open(f"file://{os.path.abspath(self.plot_filename)}")
                logging.info(f"Graph updated and displayed: {len(time_steps)} points, profit=${total_profit:.2f}, file={self.plot_filename}")
        except Exception as e:
            logging.error(f"Error updating graph: {e}")
            messagebox.showerror("Error", f"Failed to update graph: {e}")

    def start_live_trading(self):
        if not self.running:
            self.live_mode = True
            self.initialize_environment(live_mode=True)
            self.model.set_env(self.env)
            self.running = True
            self.step_count = 0  # Reset step count
            self.plot_filename = "trading_plot_live.html"
            logging.info("Live trading started")
            self.trading_thread = threading.Thread(target=self.trading_loop)
            self.trading_thread.daemon = True
            self.trading_thread.start()

    def start_historical_trading(self):
        if not self.running:
            self.live_mode = False
            self.initialize_environment(live_mode=False)
            self.model.set_env(self.env)
            self.running = True
            self.step_count = 0  # Reset step count
            self.run_number += 1
            self.plot_filename = f"trading_plot_run_{self.run_number}.html"
            # Clear graph data for new run
            self.prices_history.clear()
            self.actions_history.clear()
            self.valid_actions_history.clear()
            self.portfolio_history.clear()
            self.time_steps.clear()
            # Reinitialize with current data
            self.obs, info = self.env.reset()
            current_price = info.get('current_price', self.env.historical_data[self.env.current_step])
            self.prices_history.append(current_price)
            self.actions_history.append(1)
            self.valid_actions_history.append(1)
            self.portfolio_history.append(info['portfolio_value'])
            self.time_steps.append(self.step_count)
            self.step_count += 1
            logging.info(f"Historical trading started, run {self.run_number}, plot file: {self.plot_filename}")
            self.trading_thread = threading.Thread(target=self.trading_loop)
            self.trading_thread.daemon = True
            self.trading_thread.start()

    def stop_trading(self):
        self.running = False
        if self.model is not None:
            try:
                self.model.save(self.model_path)
                logging.info(f"Model saved to {self.model_path} on stop")
            except Exception as e:
                logging.error(f"Error saving model on stop: {e}")
        logging.info("Trading stopped")

    def exit_app(self):
        logging.info("Exiting program")
        self.running = False
        self.app_running = False
        if self.model is not None:
            try:
                self.model.save(self.model_path)
                logging.info(f"Model saved to {self.model_path} on exit")
            except Exception as e:
                logging.error(f"Error saving model on exit: {e}")
        if self.use_gui and self.root:
            self.root.quit()
            self.root.destroy()
        sys.exit(0)

    def cli_loop(self):
        logging.info("CLI started. Type 'start_live', 'start_historical', 'stop', 'graph', or 'exit'")
        while self.app_running:
            try:
                command = input("Enter command (start_live/start_historical/stop/graph/exit): ").strip().lower()
                if command == 'start_live':
                    logging.info("CLI: Starting live trading")
                    self.start_live_trading()
                elif command == 'start_historical':
                    logging.info("CLI: Starting historical trading")
                    self.start_historical_trading()
                elif command == 'stop':
                    logging.info("CLI: Stopping trading")
                    self.stop_trading()
                elif command == 'graph':
                    logging.info("CLI: Showing/updating graph")
                    self.show_graph()
                elif command == 'exit':
                    self.exit_app()
            except KeyboardInterrupt:
                logging.info("Ctrl+C in CLI, saving model and exiting...")
                self.exit_app()
            time.sleep(0.1)

    def trading_loop(self):
        logging.info("TradingApp.trading_loop started")
        last_save_time = time.time()
        while self.running:
            try:
                start_time = time.time()
                action, _ = self.model.predict(self.obs)
                prev_balance = self.env.balance
                prev_shares = self.env.shares_held
                new_obs, reward, done, _, info = self.env.step(action)
                current_price = info['current_price']
                
                # Check if action was executed
                action_executed = info['shares_bought'] > 0 or info['shares_sold'] > 0
                
                # Apply penalty for invalid actions
                if not action_executed and action != 1:
                    reward -= 0.1
                    logging.info(f"Penalty -0.1 applied for invalid action: {action}")

                # Store data
                self.prices_history.append(current_price)
                self.time_steps.append(self.step_count)
                self.actions_history.append(action)
                self.valid_actions_history.append(action if action_executed else 1)
                self.portfolio_history.append(info['portfolio_value'])
                self.obs = new_obs
                self.step_count += 1

                logging.info(f"Step: {self.step_count}, "
                            f"Price: {current_price:.2f}, Action: {action}, Executed: {action_executed}, "
                            f"Shares: {info['shares_held']}, "
                            f"Balance: {info['balance']:.2f}, "
                            f"Portfolio: {info['portfolio_value']:.2f}, Reward: {reward:.2f}")

                # Handle historical mode resets
                if not self.live_mode:
                    balance = info['balance']
                    shares_held = info['shares_held']
                    if balance < current_price and shares_held == 0:
                        logging.info("Insufficient funds and no shares in historical mode. Restarting.")
                        self.model.save(self.model_path)
                        logging.info(f"Model saved to {self.model_path} before restart")
                        # Display graph for this run
                        self.show_graph()
                        self.run_number += 1
                        self.plot_filename = f"trading_plot_run_{self.run_number}.html"
                        # Clear graph data for new run
                        self.prices_history.clear()
                        self.actions_history.clear()
                        self.valid_actions_history.clear()
                        self.portfolio_history.clear()
                        self.time_steps.clear()
                        self.step_count = 0
                        self.obs, info = self.env.reset()
                        self.prices_history.append(info['current_price'])
                        self.time_steps.append(self.step_count)
                        self.actions_history.append(1)
                        self.valid_actions_history.append(1)
                        self.portfolio_history.append(info['portfolio_value'])
                        self.step_count += 1
                        logging.info(f"Historical trading restarted due to insufficient funds, run {self.run_number}, plot file: {self.plot_filename}")
                        logging.info(f"Reset data range: {info.get('data_start_time')} to {info.get('data_end_time')}")
                        continue

                    if done:
                        logging.info("Reached end of historical data. Restarting.")
                        self.model.save(self.model_path)
                        logging.info(f"Model saved to {self.model_path} before restart")
                        # Display graph for this run
                        self.show_graph()
                        self.run_number += 1
                        self.plot_filename = f"trading_plot_run_{self.run_number}.html"
                        # Clear graph data for new run
                        self.prices_history.clear()
                        self.actions_history.clear()
                        self.valid_actions_history.clear()
                        self.portfolio_history.clear()
                        self.time_steps.clear()
                        self.step_count = 0
                        self.obs, info = self.env.reset()
                        self.prices_history.append(info['current_price'])
                        self.time_steps.append(self.step_count)
                        self.actions_history.append(1)
                        self.valid_actions_history.append(1)
                        self.portfolio_history.append(info['portfolio_value'])
                        self.step_count += 1
                        logging.info(f"Historical trading restarted after completing data, run {self.run_number}, plot file: {self.plot_filename}")
                        logging.info(f"Reset data range: {info.get('data_start_time')} to {info.get('data_end_time')}")
                        continue

                # Stop in live mode if insufficient funds and no shares
                if self.live_mode:
                    balance = info['balance']
                    shares_held = info['shares_held']
                    if balance < current_price and shares_held == 0:
                        logging.info("Insufficient funds to buy and no shares to sell in live mode. Stopping trading.")
                        self.show_graph()  # Display final graph for live mode
                        self.stop_trading()
                        break

                current_time = time.time()
                if current_time - last_save_time >= 600:
                    try:
                        self.model.save(self.model_path)
                        logging.info(f"Periodic model save to {self.model_path} at {current_time}")
                        last_save_time = current_time
                    except Exception as e:
                        logging.error(f"Error during periodic model save: {e}")
            except Exception as e:
                logging.error(f"Error in trading loop: {e}")
                self.stop_trading()
                break
        logging.info("TradingApp.trading_loop finished")

if __name__ == '__main__':
    try:
        logging.info("Starting TradingApp...")
        app = TradingApp(symbol='SPY')
        if app.use_gui and app.root:
            app.root.mainloop()
        else:
            app.cli_loop()
        logging.info("TradingApp started successfully")
    except Exception as e:
        logging.error(f"Error creating TradingApp: {e}")
        sys.exit(1)