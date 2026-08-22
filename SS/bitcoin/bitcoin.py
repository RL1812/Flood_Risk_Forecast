import requests
import sys
try:
    if len(sys.argv) == 1:
        print("Missing command-line argument")
        sys.exit(1)
    try:
        coin_num = float(sys.argv[1])
    except ValueError:
        print("Command-line argument is not a number")
        sys.exit(1)
    r = requests.get("https://api.coindesk.com/v1/bpi/currentprice.json")
    # print(r.json())
    # print(type(sys.argv[1]))
    result = float(sys.argv[1]) * float(r.json()["bpi"]["USD"]["rate_float"])
    print(f"${result:,.4f}")
except requests.RequestException:
    sys.exit()
