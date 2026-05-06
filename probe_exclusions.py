"""Toma 3 páginas de mercados y muestra desglose de exclusiones."""
import sys, json, requests, time
sys.path.insert(0, ".")
from src.data.download import passes_filters, parse_dt, GAMMA_API_BASE
from collections import Counter

excluded = Counter()
candidates = 0
total = 0

for offset in [20000, 21000, 25000]:  # páginas donde vimos el salto de candidatos
    r = requests.get(f"{GAMMA_API_BASE}/markets", params={
        "closed": "true", "limit": 100, "offset": offset,
        "order": "closedTime", "ascending": "false",
    }, timeout=15)
    page = r.json()
    for m in page:
        total += 1
        ok, reason = passes_filters(m)
        if ok:
            candidates += 1
            # Mostrar algunos ejemplos de candidatos
            if candidates <= 3:
                sd = m.get("startDate","") or m.get("createdAt","")
                ct = m.get("closedTime","")
                try:
                    dur = (parse_dt(ct) - parse_dt(sd)).days
                except:
                    dur = "?"
                print(f"CANDIDATO: {m.get('question','')[:65]}")
                print(f"  start={sd[:10]} closed={ct[:10]} dur={dur}d vol={m.get('volumeNum','?')}")
                op = json.loads(m.get("outcomePrices", '["0"]'))
                outcome_str = "YES" if op[0] == "1" else "NO"
                print(f"  outcome={outcome_str}")
                print()
        else:
            excluded[reason] += 1
    time.sleep(0.5)

print(f"Total procesados: {total} | Candidatos: {candidates} | Tasa: {candidates/total*100:.1f}%")
print("\nExclusiones:")
for reason, count in excluded.most_common():
    pct = count / total * 100
    print(f"  {reason:30s}: {count:4d} ({pct:.1f}%)")
