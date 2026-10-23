"""One trader step in its own process (run by the supervisor every TRADER_LOOP_SECONDS).

Loads the trader state from trader_status_kronos.json, picks up plans it has not handled yet from
plans_kronos.json, places / manages / expires entries and positions, saves the state and exits.
"""
import json
import os
import sys

import config as C


def main():
    if not C.DRY_RUN and not os.getenv('KUCOIN_API_KEY'):
        print("❌ [TRADER] DRY_RUN=false but KUCOIN_API_KEY is empty -> not trading.")
        return 1
    import ccxt
    from trader import Trader

    ex = ccxt.kucoinfutures({
        'apiKey': os.getenv('KUCOIN_API_KEY', ''),
        'secret': os.getenv('KUCOIN_API_SECRET', ''),
        'password': os.getenv('KUCOIN_API_PASSWORD', ''),
        'enableRateLimit': True,
    })
    tr = Trader(ex)
    tr.local_always = True
    tr.load_status(prefer_local=True, quiet=True)

    plans = []
    if os.path.exists(C.PLANS_FILE):
        try:
            with open(C.PLANS_FILE, encoding='utf-8') as f:
                plans = json.load(f).get('plans', [])
        except Exception as e:
            print(f"⚠️ [TRADER] {C.PLANS_FILE} unreadable: {e}")
    fresh = tr.new_plans(plans)
    if fresh:
        print(f"📨 [TRADER] {len(fresh)} new plan(s) from the scanner")
        tr._dirty = True
    tr.step()
    return 0


if __name__ == '__main__':
    sys.exit(main())
