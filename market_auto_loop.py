import argparse, datetime as dt, json, sys, time, urllib.request

API = "http://localhost:8000"
POLL_SECONDS = 30
DEFAULT_INTERVAL = 900

try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
except Exception:
    pass

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"


def _now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _get(path, timeout=10):
    req = urllib.request.Request(API + path)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _post(path, body={}, timeout=120):
    data = json.dumps(body).encode()
    req = urllib.request.Request(API + path, data=data,
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def check_server():
    try:
        _get("/health", timeout=4)
        return True
    except Exception:
        return False


def get_market_status():
    try:
        return _get("/status", timeout=8)
    except Exception:
        return None


def run_agent(dry_run_override):
    body = {}
    if dry_run_override is not None:
        body["dry_run"] = dry_run_override
    try:
        return _post("/run", body, timeout=120)
    except Exception as e:
        print(f"{RED}[{_now()}] Agent cycle FAILED: {e}{RESET}")
        return None


def run_all_agent(dry_run_override):
    body = {}
    if dry_run_override is not None:
        body["dry_run"] = dry_run_override
    try:
        return _post("/run-all", body, timeout=600)
    except Exception as e:
        print(f"{RED}[{_now()}] Multi-symbol cycle FAILED: {e}{RESET}")
        return None


def fmt_next(iso):
    try:
        ts = dt.datetime.fromisoformat(iso).astimezone()
        return ts.strftime("%H:%M %Z")
    except Exception:
        return iso


def print_single_result(result, interval):
    direction = result.get("direction", "flat").upper().replace("_", " ")
    edge      = result.get("composite_edge") or 0
    regime    = result.get("regime_label", "?")
    kelly     = result.get("kelly_fraction") or 0
    budget    = result.get("risk_budget_dollars") or 0
    orders    = result.get("planned_orders", [])
    spot      = result.get("spot") or 0
    fvol      = (result.get("forecast_vol") or 0) * 100
    iv        = (result.get("live_atm_iv") or 0) * 100
    drawdown  = (result.get("drawdown_pct") or 0) * 100

    ec = GREEN if edge > 0.2 else RED if edge < -0.2 else YELLOW
    dc = GREEN if "LONG" in direction else RED if "SHORT" in direction else DIM

    print(f"  {CYAN}Symbol  :{RESET} {result.get('symbol')}   spot ${spot:.2f}")
    print(f"  {CYAN}Regime  :{RESET} {regime}   Forecast {fvol:.1f}%   Live IV {iv:.1f}%")
    print(f"  {CYAN}Edge    :{RESET} {ec}{edge:+.4f}{RESET}  =>  {dc}{BOLD}{direction}{RESET}")
    print(f"  {CYAN}Kelly   :{RESET} {kelly*100:+.2f}%   Risk budget ${budget:,.0f}   Drawdown {drawdown:.2f}%")

    if orders:
        for o in orders:
            tag = f"{GREEN}SUBMITTED{RESET}" if o.get("submitted") else f"{YELLOW}DRY-RUN (not submitted){RESET}"
            print(f"  {CYAN}Order   :{RESET} [{tag}] {o.get('description', '')}")
    else:
        print(f"  {CYAN}Orders  :{RESET} {DIM}none this cycle{RESET}")

    notes = result.get("notes") or []
    for n in notes[:2]:
        print(f"  {YELLOW}NOTE: {n[:110]}{RESET}")

    print(f"  {DIM}Next cycle in {interval//60} min | Ctrl+C to stop{RESET}")
    print()


def print_multi_result(data, interval):
    summary = data.get("summary", {})
    results = data.get("results", [])
    errors  = data.get("errors", [])

    print(f"  {CYAN}Scanned :{RESET} {BOLD}{summary.get('total_symbols', 0)}{RESET} symbols  "
          f"({GREEN}{summary.get('succeeded', 0)} ok{RESET}"
          f"{f', {RED}{summary.get(chr(34)+\"failed\"+chr(34), 0)} failed{RESET}' if summary.get('failed') else ''})")

    # Summary table
    print(f"\n  {'Symbol':<8} {'Direction':<12} {'Edge':>8} {'Kelly':>8} {'Orders':>7} {'Exits':>6}")
    print(f"  {'─'*8} {'─'*12} {'─'*8} {'─'*8} {'─'*7} {'─'*6}")

    for r in results:
        sym       = r.get("symbol", "?")
        direction = r.get("direction", "flat").upper().replace("_", " ")
        edge      = r.get("composite_edge") or 0
        kelly     = r.get("kelly_fraction") or 0
        n_orders  = len(r.get("planned_orders", []))
        n_exits   = len(r.get("exit_decisions", []))

        dc = GREEN if "LONG" in direction else RED if "SHORT" in direction else DIM
        ec = GREEN if edge > 0.2 else RED if edge < -0.2 else YELLOW

        print(f"  {BOLD}{sym:<8}{RESET} {dc}{direction:<12}{RESET} "
              f"{ec}{edge:>+8.4f}{RESET} {kelly*100:>+7.2f}% {n_orders:>7} {n_exits:>6}")

    for e in errors:
        print(f"  {RED}{e.get('symbol','?'):<8} ERROR: {e.get('error','')[:60]}{RESET}")

    long_v  = summary.get('long_vol', 0)
    short_v = summary.get('short_vol', 0)
    flat_v  = summary.get('flat', 0)
    exits   = summary.get('total_exits', 0)
    orders  = summary.get('total_orders', 0)

    print(f"\n  {GREEN}▲ LONG {long_v}{RESET}  {RED}▼ SHORT {short_v}{RESET}  "
          f"{DIM}— FLAT {flat_v}{RESET}  Exits: {exits}  Orders: {orders}")
    print(f"  {DIM}Next cycle in {interval//60} min | Ctrl+C to stop{RESET}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Auto-loop agent on market open")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help="Seconds between agent cycles (default 900=15min)")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=False,
                        help="Force dry_run=true (plans only, no submit)")
    parser.add_argument("--live", dest="live", action="store_true",
                        help="Force dry_run=false (submits real paper orders)")
    parser.add_argument("--single", dest="single", action="store_true",
                        help="Single-symbol mode (uses SYMBOL from .env instead of WATCHLIST)")
    args = parser.parse_args()

    dry_run_override = None
    if args.dry_run:
        dry_run_override = True
    elif args.live:
        dry_run_override = False

    scan_mode = "SINGLE-SYMBOL" if args.single else "MULTI-SYMBOL (WATCHLIST)"

    print()
    print(f"{BOLD}{CYAN}=== VOL/AGENT - Market-Open Auto-Loop ==={RESET}")
    print(f"  API      : {API}")
    print(f"  Interval : every {args.interval//60} min")
    print(f"  Scan     : {scan_mode}")
    mode_str = "DRY-RUN override" if dry_run_override else "LIVE PAPER override" if dry_run_override is False else "uses .env DRY_RUN setting"
    print(f"  Mode     : {mode_str}")
    print(f"  Stop     : Ctrl+C")
    print()

    cycle_count = 0
    last_run_ts = 0.0

    while True:
        try:
            if not check_server():
                print(f"{RED}[{_now()}] API server not reachable at {API}{RESET}")
                print(f"{YELLOW}  => Start it with:  python cli.py serve --port 8000{RESET}")
                time.sleep(POLL_SECONDS)
                continue

            status = get_market_status()
            if status is None:
                print(f"{YELLOW}[{_now()}] Could not read market status - retrying...{RESET}")
                time.sleep(POLL_SECONDS)
                continue

            market_open = status.get("market_open", False)
            next_open   = status.get("next_open", "")
            next_close  = status.get("next_close", "")
            acct        = status.get("account", {})
            equity      = acct.get("equity", "?")
            mode_label  = "DRY RUN" if status.get("dry_run") else "LIVE PAPER"

            if not market_open:
                nof = fmt_next(next_open) if next_open else "unknown"
                print(f"{DIM}[{_now()}] Market CLOSED - next open {nof} - checking in {POLL_SECONDS}s...      {RESET}", end="\r")
                time.sleep(POLL_SECONDS)
                continue

            ncf = fmt_next(next_close) if next_close else "unknown"
            now_ts = time.time()
            elapsed = now_ts - last_run_ts
            time_to_next = max(0.0, args.interval - elapsed)

            if elapsed < args.interval and last_run_ts > 0:
                mins, secs = divmod(int(time_to_next), 60)
                print(f"{GREEN}[{_now()}] OPEN - closes {ncf} - cycle #{cycle_count} done - "
                      f"next in {mins:02d}:{secs:02d} - ${equity} - {mode_label}{RESET}", end="\r")
                time.sleep(min(POLL_SECONDS, time_to_next))
                continue

            cycle_count += 1
            print(f"\n{BOLD}{GREEN}[{_now()}] >> CYCLE #{cycle_count}  "
                  f"(market open - closes {ncf} - equity ${equity}){RESET}")

            if args.single:
                result = run_agent(dry_run_override)
                last_run_ts = time.time()
                if result:
                    print_single_result(result, args.interval)
                else:
                    print(f"{RED}  No result returned - will retry next interval{RESET}\n")
            else:
                data = run_all_agent(dry_run_override)
                last_run_ts = time.time()
                if data:
                    print_multi_result(data, args.interval)
                else:
                    print(f"{RED}  No result returned - will retry next interval{RESET}\n")

        except KeyboardInterrupt:
            print(f"\n\n{BOLD}{YELLOW}[{_now()}] Stopped by user. {cycle_count} cycles completed.{RESET}\n")
            sys.exit(0)


if __name__ == "__main__":
    main()
