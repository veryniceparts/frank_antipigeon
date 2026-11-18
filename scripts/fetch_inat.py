#!/usr/bin/env python3
"""Download CC-licensed iNaturalist photos: pigeons as positives, look-alike birds as hard negatives.

  python scripts/fetch_inat.py --out data/inat --per-taxon 600

Writes <out>/pigeon/*.jpg and <out>/not_pigeon/*.jpg plus ATTRIBUTION.csv (license + observer per
photo; CC-BY requires attribution if you redistribute). Photos are whole observations, not tight
motion crops, so they complement but do not replace crops from your own camera (harvest.py).
Uses the public API (~1 request/s, be polite). Requires network access to api.inaturalist.org and
inaturalist-open-data.s3.amazonaws.com.
"""
import argparse
import csv
import json
import os
import time
import urllib.parse
import urllib.request

POS = ["Columba livia", "Columba palumbus"]  # wood pigeon counts as pigeon, as in the teacher
# hard negatives: doves, other city birds, birds of similar size/colour
NEG = ["Streptopelia decaocto", "Streptopelia senegalensis", "Zenaida macroura",
       "Corvus corone", "Corvus brachyrhynchos", "Corvus monedula", "Pica pica", "Larus argentatus",
       "Chroicocephalus ridibundus", "Passer domesticus", "Sturnus vulgaris", "Turdus merula",
       "Accipiter nisus", "Falco tinnunculus", "Psittacula krameri", "Myiopsitta monachus"]
API = "https://api.inaturalist.org/v1/observations"
UA = {"User-Agent": "pigeonkit-dataset-fetcher"}
LICENSES = {"cc0", "cc-by", "cc-by-sa"}


def download(url, fn, timeout=30):
    tmp = fn + ".part"  # atomic: a failed transfer never leaves a truncated file under the real name
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r, open(tmp, "wb") as fh:
            fh.write(r.read())
        os.replace(tmp, fn)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def fetch(taxon, n, out_dir, size, att):
    os.makedirs(out_dir, exist_ok=True)
    got, page = 0, 1
    while got < n:
        q = urllib.parse.urlencode({
            "taxon_name": taxon, "quality_grade": "research", "photos": "true",
            "photo_license": "cc0,cc-by,cc-by-sa", "per_page": 200, "page": page, "order_by": "votes"})
        with urllib.request.urlopen(urllib.request.Request(f"{API}?{q}", headers=UA), timeout=30) as r:
            res = json.load(r)["results"]
        if not res:
            break
        for obs in res:
            if got >= n:
                break
            # the API filter matches the observation; check each photo's own license
            ph = next((p for p in obs["photos"] if (p.get("license_code") or "").lower() in LICENSES), None)
            if ph is None:
                continue
            url = ph["url"].replace("/square.", f"/{size}.")
            fn = os.path.join(out_dir, f"inat_{taxon.replace(' ', '_')}_{ph['id']}.jpg")
            if not os.path.exists(fn):
                try:
                    download(url, fn)
                except Exception as e:  # noqa: BLE001
                    print("skip", url, e)
                    continue
            att.writerow([os.path.basename(fn), taxon, ph.get("license_code"), ph.get("attribution"), url])
            got += 1
        page += 1
        time.sleep(1.0)
    print(f"{taxon}: {got}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/inat")
    ap.add_argument("--per-taxon", type=int, default=500)
    ap.add_argument("--neg-per-taxon", type=int, default=150)
    ap.add_argument("--size", default="small", choices=["small", "medium"], help="small=240px, medium=500px")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "ATTRIBUTION.csv"), "a", newline="") as fh:
        att = csv.writer(fh)
        for t in POS:
            fetch(t, a.per_taxon, os.path.join(a.out, "pigeon"), a.size, att)
        for t in NEG:
            fetch(t, a.neg_per_taxon, os.path.join(a.out, "not_pigeon"), a.size, att)


if __name__ == "__main__":
    main()
