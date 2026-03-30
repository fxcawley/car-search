import requests, json, re, os, time, csv
from bs4 import BeautifulSoup
from datetime import datetime

DATA_DIR = os.path.join("C:", os.sep, "Users", "lcawley", "bridge", "data")
H = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36","Accept":"text/html"}

all_vehicles = {}

def fetch_page(url):
    resp = requests.get(url, headers=H, timeout=30)
    if resp.status_code != 200: return []
    soup = BeautifulSoup(resp.text, "html.parser")
    nd = soup.select_one("#__NEXT_DATA__")
    if not (nd and nd.string): return []
    data = json.loads(nd.string)
    return data.get("props", {}).get("pageProps", {}).get("vehicles", [])

def add_vehicles(vehs, label=""):
    nc = 0
    for v in vehs:
        vin = v.get("vin", "")
        if vin and vin not in all_vehicles:
            all_vehicles[vin] = v
            nc += 1
    if nc > 0: print("  " + label + ": " + str(nc) + " new (total: " + str(len(all_vehicles)) + ")")
    return nc

print("Fetching pages...")
base = "https://www.autolist.com/bmw-3+series-2011"
for pg in range(1, 30):
    url = base + ("?page=" + str(pg) if pg > 1 else "")
    vehs = fetch_page(url)
    if not vehs: break
    nc = add_vehicles(vehs, "Page " + str(pg))
    if nc == 0: break
    time.sleep(1.5)

print("Trying sort options...")
for sort in ["price_asc","price_desc","miles_asc","miles_desc","newest","best_deal"]:
    for pg in range(1, 8):
        url = base + "?sort=" + sort + ("&page=" + str(pg) if pg > 1 else "")
        vehs = fetch_page(url)
        if vehs: add_vehicles(vehs, sort + " p" + str(pg))
        time.sleep(0.8)

print("Trying locations...")
for loc in ["New+York-NY","Los+Angeles-CA","Chicago-IL","Houston-TX","Phoenix-AZ","Philadelphia-PA","Dallas-TX","Miami-FL","Atlanta-GA","Denver-CO","Seattle-WA","Detroit-MI","Minneapolis-MN","Tampa-FL","Portland-OR","Charlotte-NC","Nashville-TN","Indianapolis-IN","San+Francisco-CA","Boston-MA"]:
    url = base + "?location=" + loc
    vehs = fetch_page(url)
    if vehs: add_vehicles(vehs, loc)
    time.sleep(1)


print("Total unique all trims:", len(all_vehicles))
bmw328 = {vin: v for vin, v in all_vehicles.items() if "328i" in (v.get("trim", "") or "")}
print("328i/328i xDrive:", len(bmw328))

trims = {}
for v in all_vehicles.values():
    t = v.get("trim", "unknown")
    trims[t] = trims.get(t, 0) + 1
print("Trims:", trims)

listings = []
for vin, v in bmw328.items():
    price = v.get("price", 0)
    listings.append({
        "title": str(v.get("year",2011))+" BMW "+str(v.get("model","3 Series"))+" "+str(v.get("trim","328i")),
        "price": float(price) if price and price > 0 else None,
        "mileage": float(v["mileage"]) if v.get("mileage") else None,
        "location": v.get("location", ""),
        "state": v.get("state", ""),
        "seller_type": "dealer",
        "accident_history": "unknown",
        "source": "autolist",
        "url": "https://www.autolist.com" + v.get("vdpUrl", ""),
        "vin": vin,
        "body_type": v.get("bodyType", ""),
        "transmission": v.get("transmission", ""),
        "driveline": v.get("driveline", ""),
        "ext_color": v.get("normalizedColorExterior", ""),
        "dealer_name": v.get("dealerName", ""),
        "imv_expected_price": v.get("imvExpectedPrice"),
        "imv_deal_rating": v.get("imvLocalizedDealRating", ""),
    })

listings.sort(key=lambda x: x.get("price") or 999999)

raw_path = os.path.join(DATA_DIR, "market_listings_raw.json")
with open(raw_path, "w") as f:
    json.dump({"scrape_date": datetime.now().isoformat(), "search_params": {"make":"BMW","model":"328i","year":2011,"generation":"E90"}, "source":"autolist.com", "total_unique_all_trims": len(all_vehicles), "total_328i": len(listings), "listings": listings}, f, indent=2, default=str)
print("Saved:", raw_path)

csv_path = os.path.join(DATA_DIR, "market_listings.csv")
flds = ["title","price","mileage","location","state","seller_type","accident_history","source","url","vin","body_type","transmission","driveline","ext_color","dealer_name","imv_expected_price","imv_deal_rating"]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=flds, extrasaction="ignore")
    writer.writeheader()
    for lst in listings: writer.writerow(lst)
print("Saved:", csv_path)

print()
print("=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
prices = [l["price"] for l in listings if l.get("price")]
mileages = [l["mileage"] for l in listings if l.get("mileage")]
D = chr(36)
print("Total 328i listings:", len(listings))
print("With asking price:", len(prices))
print("With mileage:", len(mileages))
if prices:
    sp = sorted(prices)
    print("Price Min:", D + "{:,.0f}".format(min(sp)))
    print("Price Max:", D + "{:,.0f}".format(max(sp)))
    print("Price Mean:", D + "{:,.0f}".format(sum(sp)/len(sp)))
    print("Price Median:", D + "{:,.0f}".format(sp[len(sp)//2]))
if mileages:
    sm = sorted(mileages)
    print("Mileage Min:", "{:,.0f}".format(min(sm)), "mi")
    print("Mileage Max:", "{:,.0f}".format(max(sm)), "mi")
    print("Mileage Mean:", "{:,.0f}".format(sum(sm)/len(sm)), "mi")
    print("Mileage Median:", "{:,.0f}".format(sm[len(sm)//2]), "mi")

states = {}
for l in listings:
    s = l.get("state", "?")
    states[s] = states.get(s, 0) + 1
print("By state:", dict(sorted(states.items(), key=lambda x:-x[1])))

print("LISTINGS (sorted by price):")
for i, l in enumerate(listings):
    p_s = D + "{:,.0f}".format(l["price"]) if l.get("price") else "(no price)"
    m_s = "{:,.0f} mi".format(l["mileage"]) if l.get("mileage") else "N/A"
    imv_s = ""
    if l.get("imv_expected_price"): imv_s = " [IMV:" + D + "{:,.0f}]".format(l["imv_expected_price"])
    dl = l.get("imv_deal_rating", "")
    print(" ", str(i+1)+".", l["title"], "|", p_s+imv_s, "|", m_s, "|", l.get("location","?"), "|", dl)

print("Files:")
for fn in sorted(os.listdir(DATA_DIR)):
    fp = os.path.join(DATA_DIR, fn)
    if os.path.isfile(fp): print(" ", fn, "(" + "{:,}".format(os.path.getsize(fp)) + " bytes)")
print("Done:", datetime.now())
