"""All settings, read from environment variables. See .env.example for the full list with comments."""
import os


def _float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        print(f"⚠️ {name}={os.environ.get(name)!r} is not a number, using default {default}.")
        return float(default)


def _int(name, default):
    return int(round(_float(name, default)))


def _bool(name, default='false'):
    return os.environ.get(name, default).strip().lower() in ('1', 'true', 'yes')


def _str(name, default):
    return os.environ.get(name, default).strip()


# ---------------------------------------------------------------- general
TIMEFRAME = '4h'
CANDLE_SECONDS = 4 * 3600
DRY_RUN = _bool('DRY_RUN', 'true')            # paper trading: simulated fills from 1m candles, no orders sent
SCAN_DELAY_SECONDS = _int('SCAN_DELAY_SECONDS', 60)
TRADER_LOOP_SECONDS = _int('TRADER_LOOP_SECONDS', 60)
HISTORY_BARS = _int('HISTORY_BARS', 500)      # 4h candles fetched per coin (indicators + Kronos context)

# ---------------------------------------------------------------- indicators
BB_PERIOD = _int('BB_PERIOD', 20)
BB_STD = _float('BB_STD', 2.0)
BB_PCTL_WINDOW = _int('BB_PCTL_WINDOW', 100)
ATR_PERIOD = _int('ATR_PERIOD', 14)
RSI_PERIOD = _int('RSI_PERIOD', 14)
VOL_MA_PERIOD = _int('VOL_MA_PERIOD', 20)

# ---------------------------------------------------------------- setups
ENABLE_SQUEEZE = _bool('ENABLE_SQUEEZE', 'true')
SQUEEZE_PCTL = _float('SQUEEZE_PCTL', 0.35)          # band width in the lowest 35% of the last BB_PCTL_WINDOW candles
SQUEEZE_MIN_VOL_RATIO = _float('SQUEEZE_MIN_VOL_RATIO', 0.0)  # 0 = no volume requirement
# 'discount': squeeze longs only at/below the middle line, shorts only at/above it (buy low, sell high)
# 'any'     : Kronos alone picks the side
SQUEEZE_SIDE_FILTER = _str('SQUEEZE_SIDE_FILTER', 'discount')

ENABLE_MEAN_REVERSION = _bool('ENABLE_MEAN_REVERSION', 'true')
MR_PCTB_LOW = _float('MR_PCTB_LOW', 0.10)             # %B at/below this = stretched down -> long candidate
MR_PCTB_HIGH = _float('MR_PCTB_HIGH', 0.90)           # %B at/above this = stretched up -> short candidate
MR_RSI_LOW = _float('MR_RSI_LOW', 40.0)               # extra confirmation for MR longs (RSI <= this); 0 = off
MR_RSI_HIGH = _float('MR_RSI_HIGH', 60.0)             # extra confirmation for MR shorts (RSI >= this); 100 = off

# ---------------------------------------------------------------- Kronos
USE_KRONOS = _bool('USE_KRONOS', 'true')          # false = bootstrap baseline forecaster (paper only, for pipeline tests)
LOG_ALL_FORECASTS = _bool('LOG_ALL_FORECASTS', 'true')  # forecast every coin each scan, not only coins with a setup
# Kronos-mini (4.1M params) + Tokenizer-2k fits a 512 MB instance. On 2 GB+ you can use
# Kronos-small / Kronos-base with Kronos-Tokenizer-base (context 512).
KRONOS_MODEL = _str('KRONOS_MODEL', 'NeoQuasar/Kronos-mini')
KRONOS_TOKENIZER = _str('KRONOS_TOKENIZER', 'NeoQuasar/Kronos-Tokenizer-2k')
KRONOS_MAX_CONTEXT = _int('KRONOS_MAX_CONTEXT', 2048 if 'mini' in KRONOS_MODEL.lower() else 512)
KRONOS_LOOKBACK = _int('KRONOS_LOOKBACK', 400)        # context candles fed to the model
KRONOS_PRED_LEN = _int('KRONOS_PRED_LEN', 6)          # forecast horizon in 4h candles (6 = 24h)
KRONOS_SAMPLES = _int('KRONOS_SAMPLES', 30)           # independent sampled paths per coin
KRONOS_BATCH = _int('KRONOS_BATCH', 10)               # paths computed at once; lower = less peak memory, slower
KRONOS_T = _float('KRONOS_T', 1.0)
KRONOS_TOP_P = _float('KRONOS_TOP_P', 0.9)
KRONOS_SEED = _int('KRONOS_SEED', 0)                  # 0 = random each scan
KRONOS_MIN_PROB = _float('KRONOS_MIN_PROB', 0.60)     # share of paths that must end on our side
KRONOS_THREADS = _int('KRONOS_THREADS', 1)            # torch CPU threads (Render starter has 0.5 CPU)
KRONOS_TIMEOUT = _int('KRONOS_TIMEOUT', 3000)         # seconds before the inference process is killed

# ---------------------------------------------------------------- entry snipe
ENTRY_WINDOW_CANDLES = _int('ENTRY_WINDOW_CANDLES', 2)  # limit order lives this many 4h candles (8h)
SNIPE_QUANTILE = _float('SNIPE_QUANTILE', 0.50)       # long: this quantile of predicted lows in the window (lower = deeper, fewer fills)
SNIPE_MIN_ATR = _float('SNIPE_MIN_ATR', 0.15)         # limit at least this far from the current price
SNIPE_MAX_ATR = _float('SNIPE_MAX_ATR', 2.0)          # ...and at most this far
POST_ONLY = _bool('POST_ONLY', 'true')

# ---------------------------------------------------------------- exits
SL_ATR_MULT = _float('SL_ATR_MULT', 1.0)              # minimum stop distance below the entry
SL_PATH_QUANTILE = _float('SL_PATH_QUANTILE', 0.10)   # stop also beyond this quantile of predicted extremes...
SL_BUFFER_ATR = _float('SL_BUFFER_ATR', 0.25)         # ...plus this buffer
SL_MAX_ATR = _float('SL_MAX_ATR', 3.0)                # never wider than this
TP_QUANTILE = _float('TP_QUANTILE', 0.50)             # take profit at this quantile of predicted horizon closes
MIN_RR = _float('MIN_RR', 1.5)                        # skip trades whose Kronos reward:risk is below this
BREAKEVEN_R = _float('BREAKEVEN_R', 1.0)              # stop to entry after +1R
MAX_HOLD_CANDLES = _int('MAX_HOLD_CANDLES', 0)        # time stop; 0 = KRONOS_PRED_LEN (the forecast horizon)

# ---------------------------------------------------------------- sizing & risk
TRADE_AMOUNT_USDT = _float('TRADE_AMOUNT_USDT', 20.0)  # margin per trade; position = margin x leverage
BASE_LEVERAGE = _int('BASE_LEVERAGE', 3)
MIN_LEVERAGE = _int('MIN_LEVERAGE', 1)
MAX_LEVERAGE = _int('MAX_LEVERAGE', 5)
MIN_SCORE = _float('MIN_SCORE', 55.0)                 # 0-100 quality score of a plan (see signals.score_plan)
MAX_CONCURRENT_POSITIONS = _int('MAX_CONCURRENT_POSITIONS', 4)  # positions + resting entries
COOLDOWN_SECONDS = _int('COOLDOWN_SECONDS', 4 * 3600)
MARGIN_MODE = _str('MARGIN_MODE', 'cross')

BOT = 'KRONOS_SNIPER'
PREFIX = 'krn_'
STATUS_FILE = 'trader_status_kronos.json'
PLANS_FILE = 'plans_kronos.json'
FORECAST_LOG = 'forecast_log_kronos.csv'
