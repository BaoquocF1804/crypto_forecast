import ccxt

exchange = ccxt.binance()
exchange.load_markets()

print("Checking for XAUT...")
found = False
for symbol in exchange.symbols:
    if 'XAUT' in symbol or 'GOLD' in symbol:
        print(f"Found: {symbol}")
        found = True

if not found:
    print("XAUT not found on Binance.")
    print("Checking for PAXG (Paxos Gold)...")
    for symbol in exchange.symbols:
        if 'PAXG' in symbol:
            print(f"Found: {symbol}")
