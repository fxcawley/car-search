import requests, json, re, os, time, csv
from bs4 import BeautifulSoup

DATA_DIR = os.path.join('C:', os.sep, 'Users', 'lcawley', 'bridge', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
H = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36','Accept':'text/html'}

all_v = []
page = 1
print('Fetching AutoList pages for 2011 BMW 3 Series...')
while page <= 25:
    try:
        url = 'https://www.autolist.com/bmw-3+series-2011'
        if page > 1:
            url += '?page=' + str(page)
        resp = requests.get(url, headers=H, timeout=30)
        if resp.status_code != 200:
            print('  Page', page, ': HTTP', resp.status_code, '- stopping')
            break
        soup = BeautifulSoup(resp.text, 'html.parser')
        nd = soup.select_one('#__NEXT_DATA__')
        if not (nd and nd.string):
            print('  Page', page, ': No data - stopping')
            break
        data = json.loads(nd.string)
        vehs = data.get('props', {}).get('pageProps', {}).get('vehicles', [])
        if not vehs:
            print('  Page', page, ': 0 vehicles - stopping')
            break
        all_v.extend(vehs)
        print('  Page', page, ':', len(vehs), 'vehicles (total:', len(all_v), ')')
        page += 1
        time.sleep(1)
    except Exception as e:
        print('  Error on page', page, ':', e)
        break

print()
print('Total fetched:', len(all_v))
bmw328 = [v for v in all_v if '328i' in (v.get('trim', '') or '')]
print('328i variants:', len(bmw328))

trims = {}
for v in all_v:
    t = v.get('trim', 'unknown')
    trims[t] = trims.get(t, 0) + 1
print('Trim breakdown:', trims)

listings = []
for v in bmw328:
    price = v.get('price', 0)
    mileage = v.get('mileage')
    yr = v.get('year', 2011)
    mdl = v.get('model', '3 Series')
    trm = v.get('trim', '328i')
    loc = v.get('location', '')
    st = v.get('state', '')
    vin_val = v.get('vin', '')
    imv = v.get('imvExpectedPrice')
    deal = v.get('imvLocalizedDealRating', '')
    dn = v.get('dealerName', '')
    bt = v.get('bodyType', '')
    dl = v.get('driveline', '')
    tm = v.get('transmission', '')
    ec = v.get('normalizedColorExterior', '')
    vdp = v.get('vdpUrl', '')
    
    listing = {
        'title': str(yr) + ' BMW ' + str(mdl) + ' ' + str(trm),
        'price': float(price) if price and price > 0 else None,
        'mileage': float(mileage) if mileage else None,
        'location': loc,
        'state': st,
        'seller_type': 'dealer',
        'accident_history': 'unknown',
        'source': 'autolist',
        'url': 'https://www.autolist.com' + vdp,
        'vin': vin_val,
        'body_type': bt,
        'transmission': tm,
        'driveline': dl,
        'ext_color': ec,
        'dealer_name': dn,
        'imv_expected_price': imv,
        'imv_deal_rating': deal,
    }
    listings.append(listing)

print('Total listings:', len(listings))
with_price = [l for l in listings if l.get('price')]
print('With asking price:', len(with_price))

# Save raw JSON
from datetime import datetime
raw_path = os.path.join(DATA_DIR, 'market_listings_raw.json')
with open(raw_path, 'w') as f:
    json.dump({
        'scrape_date': datetime.now().isoformat(),
        'search_params': {'make': 'BMW', 'model': '328i', 'year': 2011, 'generation': 'E90'},
        'source': 'autolist.com',
        'total_listings': len(listings),
        'listings_with_price': len(with_price),
        'listings': listings,
    }, f, indent=2, default=str)
print('Saved:', raw_path)

# Save CSV
csv_path = os.path.join(DATA_DIR, 'market_listings.csv')
flds = ['title','price','mileage','location','state','seller_type','accident_history','source','url','vin','body_type','transmission','driveline','ext_color','dealer_name','imv_expected_price','imv_deal_rating']
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=flds, extrasaction='ignore')
    writer.writeheader()
    for lst in listings:
        writer.writerow(lst)
print('Saved:', csv_path)

# Statistics
print()
print('=' * 60)
print('SUMMARY STATISTICS')
print('=' * 60)
prices = [l['price'] for l in listings if l.get('price')]
mileages = [l['mileage'] for l in listings if l.get('mileage')]
print('Total 328i listings:', len(listings))
print('With asking price:', len(prices))
print('With mileage data:', len(mileages))

if prices:
    sp = sorted(prices)
    print('Price range: $' + '{:,.0f}'.format(min(sp)) + ' - $' + '{:,.0f}'.format(max(sp)))
    print('Mean price: $' + '{:,.0f}'.format(sum(sp)/len(sp)))
    print('Median price: $' + '{:,.0f}'.format(sp[len(sp)//2]))
    
    # Price distribution
    brackets = [(0,5000),(5000,7500),(7500,10000),(10000,12500),(12500,15000),(15000,20000),(20000,50000)]
    print()
    print('Price distribution:')
    for lo, hi in brackets:
        cnt = len([p for p in sp if lo <= p < hi])
        if cnt > 0:
            bar = '#' * cnt
            print('  $' + '{:>6,}'.format(lo) + '-$' + '{:>6,}'.format(hi) + ': ' + str(cnt) + ' ' + bar)

if mileages:
    sm = sorted(mileages)
    print()
    print('Mileage range:', '{:,.0f}'.format(min(sm)), '-', '{:,.0f}'.format(max(sm)), 'mi')
    print('Mean mileage:', '{:,.0f}'.format(sum(sm)/len(sm)), 'mi')
    print('Median mileage:', '{:,.0f}'.format(sm[len(sm)//2]), 'mi')

print()
print('All listings:')
for i, l in enumerate(listings):
    p_str = '$' + '{:,.0f}'.format(l['price']) if l.get('price') else '(no price)'
    m_str = '{:,.0f} mi'.format(l['mileage']) if l.get('mileage') else 'N/A'
    imv_str = ''
    if l.get('imv_expected_price'):
        imv_str = ' IMV:$' + '{:,.0f}'.format(l['imv_expected_price'])
    deal_str = l.get('imv_deal_rating', '')
    title = l.get('title', '?')
    loc = l.get('location', '?')
    print('  ' + str(i+1) + '. ' + title + ' | ' + p_str + ' | ' + m_str + ' | ' + loc + imv_str + ' ' + deal_str)

# Files summary
print()
print('Files in', DATA_DIR + ':')
for fn in sorted(os.listdir(DATA_DIR)):
    fp = os.path.join(DATA_DIR, fn)
    if os.path.isfile(fp):
        print('  ' + fn + ' (' + '{:,}'.format(os.path.getsize(fp)) + ' bytes)')

print()
print('Completed:', datetime.now())
