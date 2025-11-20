import ccxt

try:
    exchange = ccxt.okx()
    exchange.load_markets()

    print("Checking for XAUT/USDT on OKX...")
    if 'XAUT/USDT' in exchange.symbols:
        print("Found: XAUT/USDT")
    else:
        print("XAUT/USDT not found on OKX.")
        print("Checking for similar symbols...")
        for symbol in exchange.symbols:
            if 'XAUT' in symbol:
                print(f"Found similar: {symbol}")

except Exception as e:
    print(f"Error: {e}")
