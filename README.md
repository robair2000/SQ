# Kronos Sniper

A KuCoin USDT-futures bot that runs as one Render background worker. It looks for **Bollinger squeezes** and
**mean-reversion stretches** on the 4h chart, lets the **Kronos** candlestick foundation model decide the
direction, and tries to **snipe the entry** with a resting limit order at the low Kronos predicts (longs) or
the high (shorts). It is built to run on Render's **512 MB Starter** instance.

It starts in **paper-trading mode** (`DRY_RUN=true`). Keep it there until `evaluate_forecasts.py` shows that
Kronos's direction calls beat a coin flip on your coins — that has not been demonstrated for 4h crypto.

## How a trade is made

Every 4h candle close (+60 s), for BTC, ETH, SOL, AVAX, LINK and the top-ranked coin:

1. **Setup** — which side is allowed
   - *Squeeze*: Bollinger width in the lowest 35% of the last 100 candles. With `SQUEEZE_SIDE_FILTER=discount`
     longs are only allowed at/below the middle line and shorts at/above it.
   - *Mean reversion*: %B ≤ 0.10 with RSI ≤ 40 → long side; %B ≥ 0.90 with RSI ≥ 60 → short side.
2. **Kronos direction** — Kronos-mini samples 30 independent paths of the next 6 candles (24h).
   P(up) = share of paths that close above today's close.
   Long if P(up) ≥ 0.60, short if ≤ 0.40, otherwise no trade. The side must match an allowed setup.
3. **Snipe entry** — long: limit buy at the median of the predicted lows over the next 2 candles,
   kept between 0.15 and 2 ATR under the price. Short: mirrored on the predicted highs. Post-only (maker fee).
4. **Stop** — at least 1 ATR beyond the entry and beyond the 10% worst predicted extreme (+0.25 ATR),
   never wider than 3 ATR.
5. **Target** — the median predicted close at the 24h horizon. Trades under 1.5 R:R are skipped.
6. **Score & leverage** — 0–100 from Kronos confidence, R:R, setup strength and Kronos's own fill
   probability; minimum 55. Leverage 3x scaled to 2–4x by confidence (bounds 1–5x).

The trader then:
- leaves the limit order for 2 candles (8h); cancels it if it expires, if price reaches the target first
  (missed) or goes through the stop first (invalidated)
- on a fill places the stop-loss and take-profit on the exchange (stop first; if it cannot be placed, the
  position is closed)
- moves the stop to entry at +1R, closes at the 24h forecast horizon (time stop), and keeps a software
  SL/TP check as backup
- respects `MAX_CONCURRENT_POSITIONS` (positions + resting entries), a 4h cooldown per coin and the optional
  `MAX_DAILY_LOSS_USDT`

**Paper mode** sends nothing to KuCoin. Fills, stops and targets are simulated from 1-minute candles
(when a candle touches both stop and target, the stop counts), with maker/taker fee estimates.
Results go to `trade_journal_kronossniper.csv` and the running total to `trader_status_kronos.json`.

## How it fits in 512 MB

- **Kronos-mini** (4.1M parameters, ~16 MB of weights) with its Tokenizer-2k instead of Kronos-small.
- **torch never loads in the always-on process.** Once per scan the worker writes the candles of all coins
  to a temp file and starts `kronos_infer.py` as a child process. It loads torch and the model, computes
  the paths 10 at a time (`KRONOS_BATCH`), writes them back and exits, which hands all of that memory back.
  Between scans only the worker (ccxt + pandas) is resident.
- **No second Python for the coin ranking**: `getcoins` runs inside the worker on the shared exchange.
- **Small native footprints**: 1 torch thread, `OMP_NUM_THREADS=1`, `MALLOC_ARENA_MAX=2`.
- **Memory is logged** after every scan: worker size, the Kronos process peak, and the container total and peak.

If Render ever reports an out-of-memory restart, the log says so and names the knobs: lower `KRONOS_BATCH`
(5), then `KRONOS_LOOKBACK` (300) or `KRONOS_SAMPLES` (20). Each scan takes a few minutes on the Starter's
0.5 CPU; that is fine for a 4h strategy.

## Files

| File | What it does |
|---|---|
| `worker.py` | Entry point: scanner thread + trader loop |
| `scanner.py` | 4h scan: data, setups, forecast, plans, forecast log |
| `signals.py` | Indicators, setups, Kronos path summary, entry/stop/target/score |
| `kronos_engine.py` | Runs Kronos in a child process and returns every sampled path; bootstrap baseline; weight download |
| `kronos_infer.py` | The child process: loads torch + Kronos, forecasts all coins, exits |
| `mem.py` | Memory readings for the logs |
| `trader.py` | Limit entries, fills, protection, breakeven, time stop, paper simulation |
| `config.py` / `.env.example` | Every setting with its default |
| `selftest.py` | One real Kronos forecast on live data, no orders |
| `evaluate_forecasts.py` | Scores logged forecasts against what actually happened |
| `kronos_model/` | Vendored Kronos model code (MIT, see its LICENSE; two small changes listed in its `__init__.py`) |
| `getcoins.py`, `asset_utils.py` | Coin ranking and trading universe |
| `order_utils.py`, `risk_utils.py`, `github_utils.py`, `market_data.py` | Exchange, journal, GitHub and data helpers |

## Deploy on Render

1. Push this folder to a new GitHub repo.
2. Make a GitHub fine-grained token with **Contents: read and write** on that repo.
3. Render → **New → Blueprint** → pick the repo (`render.yaml`), or **New → Background Worker** with the
   build and start commands from `render.yaml`. The **Starter** plan (512 MB) is enough.
4. Set `KUCOIN_API_KEY`, `KUCOIN_API_SECRET`, `KUCOIN_API_PASSWORD`, `GITHUB_TOKEN` and `GITHUB_REPO`.
   The KuCoin key only needs futures trading permission — no withdrawals.
5. After the first deploy open the service **Shell** and run `python selftest.py`. It should print a
   forecast per coin, `sanity: OK`, and a memory line whose container peak is comfortably under 512 MB.
6. Let it paper trade. After a week or two run `python evaluate_forecasts.py`.

The build installs CPU-only torch and downloads the Kronos weights into the build, so restarts don't
re-download them.

## Going live

Only switch `DRY_RUN=false` when:
- `evaluate_forecasts.py` shows a confident-call hit rate clearly above 50% over a few hundred matured
  forecasts, and a Brier score below 0.25, and
- the paper journal is net positive after fees over a meaningful number of trades.

When you switch, start with a small `TRADE_AMOUNT_USDT`. Paper positions are not carried into live mode.

## Tuning

| Want | Change |
|---|---|
| Fewer, stronger trades | `KRONOS_MIN_PROB=0.65`, `MIN_SCORE=65`, `MIN_RR=2` |
| Deeper snipes (better price, fewer fills) | `SNIPE_QUANTILE=0.3` |
| More fills | `SNIPE_QUANTILE=0.7`, `ENTRY_WINDOW_CANDLES=3` |
| Only one setup type | `ENABLE_SQUEEZE=false` or `ENABLE_MEAN_REVERSION=false` |
| Let Kronos pick squeeze sides freely | `SQUEEZE_SIDE_FILTER=any` |
| Bigger model (needs the 2 GB Standard plan) | `KRONOS_MODEL=NeoQuasar/Kronos-small`, `KRONOS_TOKENIZER=NeoQuasar/Kronos-Tokenizer-base`, `KRONOS_MAX_CONTEXT=512` |
| Less memory | `KRONOS_BATCH=5`, `KRONOS_LOOKBACK=300` |
| More coins | `SCAN_TOP_N=3` (each coin adds roughly half a minute of CPU per scan) |

## Run locally

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env            # fill in keys, keep DRY_RUN=true
set -a; source .env; set +a
python selftest.py
python worker.py
```

## Credits

Kronos: *Kronos: A Foundation Model for the Language of Financial Markets* (Shi et al.),
https://github.com/shiyu-coder/Kronos — model code in `kronos_model/` is used under its MIT license; weights
are downloaded from Hugging Face (`NeoQuasar/Kronos-mini`, `NeoQuasar/Kronos-Tokenizer-2k`).

This is experimental software that trades with leverage. Nothing here is financial advice; you can lose
the margin you put in.
