"""
NatureDex AI — Dataset Downloader (v2: auto-discovery)
======================================================
Builds a training dataset of North Carolina wildlife from research-grade,
CC-licensed iNaturalist photos.

WHAT CHANGED FROM v1:
  - No more hand-typed species list. The script asks iNaturalist for the
    MOST-OBSERVED species in North Carolina and uses their real taxon IDs,
    common names, and scientific names. (This fixes the duplicate-taxon-id
    bugs that hand-typed lists are prone to.)
  - The NC place ID is looked up at runtime, so it's always correct.
  - An image floor (MIN_IMAGES) keeps only well-supported species, which is
    what keeps accuracy high as the species count grows.

The output format is IDENTICAL to v1, so train_model.py works unchanged:
    dataset/
    ├── train/<species_slug>/*.jpg   (80%)
    ├── val/<species_slug>/*.jpg     (20%)
    ├── label_map.json
    └── dataset_log.csv

HOW TO TUNE (the two knobs that trade species-count vs accuracy):
  - MAX_SPECIES   : how many of the top NC species to consider. Higher = more
                    species (and a harder classification problem).
  - MIN_IMAGES    : minimum photos a species needs to be included. Higher =
                    fewer but better-trained classes (higher accuracy).

Usage:
    pip install requests Pillow tqdm
    python download_dataset.py
"""

import os
import csv
import json
import time
import random
import urllib.request
import urllib.parse
from pathlib import Path
from io import BytesIO

try:
    from PIL import Image
    from tqdm import tqdm
except ImportError:
    print("Install dependencies first:  pip install Pillow tqdm requests")
    raise

# ── Configuration ──────────────────────────────────────────────────────────────

STATE_NAME    = "North Carolina"   # looked up at runtime -> correct place_id

MAX_SPECIES        = 200    # consider up to this many top-observed NC species
IMAGES_PER_SPECIES = 200    # target images per species (gets what's available)
MIN_IMAGES         = 80     # ACCURACY FLOOR: skip species with fewer than this
IMG_SIZE      = 224         # resize all images to this (matches EfficientNet input)
VAL_SPLIT     = 0.2         # 20% validation, 80% training
RATE_LIMIT    = 0.7         # seconds between API requests (stay under ~60/min)

# Only real wildlife groups (no humans, no unknown taxa)
ICONIC_TAXA = [
    "Aves", "Mammalia", "Reptilia", "Amphibia",
    "Insecta", "Arachnida", "Plantae", "Fungi",
    "Mollusca", "Actinopterygii",
]

OUTPUT_DIR    = Path("dataset")
LOG_FILE      = OUTPUT_DIR / "dataset_log.csv"
USER_AGENT    = "NatureDexAI-DatasetDownloader/2.0 (student project)"
API           = "https://api.inaturalist.org/v1"


# ── API Helpers ────────────────────────────────────────────────────────────────

def api_get(url: str, params: dict) -> dict:
    """GET request to the iNaturalist v1 API (rate-limited by caller)."""
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{query}",
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def resolve_place_id(state_name: str) -> int:
    """
    Look up the iNaturalist place_id for a US state by name, so we never
    hardcode the wrong number. Prefers a state-level (admin_level 10) match.
    """
    data = api_get(f"{API}/places/autocomplete", {"q": state_name})
    results = data.get("results", [])
    if not results:
        raise SystemExit(f"Could not find a place named '{state_name}' on iNaturalist.")

    # Prefer an exact, state-level match
    for r in results:
        if r.get("name", "").lower() == state_name.lower() and r.get("admin_level") == 10:
            return r["id"]
    # Fall back to the first exact-name match, then the top result
    for r in results:
        if r.get("name", "").lower() == state_name.lower():
            return r["id"]
    return results[0]["id"]


def get_top_species(place_id: int, limit: int) -> list[dict]:
    """
    Ask iNaturalist for the most-observed species in this place.
    Returns dicts: {common_name, scientific_name, taxon_id, iconic_group}.
    """
    species = []
    page = 1
    per_page = 100
    while len(species) < limit:
        data = api_get(f"{API}/observations/species_counts", {
            "place_id":      place_id,
            "quality_grade": "research",
            "photos":        "true",
            "iconic_taxa":   ",".join(ICONIC_TAXA),
            "per_page":      per_page,
            "page":          page,
        })
        results = data.get("results", [])
        if not results:
            break
        for row in results:
            taxon = row.get("taxon") or {}
            if taxon.get("rank") != "species":   # skip genus/family-level IDs
                continue
            species.append({
                "common_name":     taxon.get("preferred_common_name")
                                   or taxon.get("name", "unknown"),
                "scientific_name": taxon.get("name", "unknown"),
                "taxon_id":        taxon.get("id"),
                "iconic_group":    taxon.get("iconic_taxon_name", ""),
                "observations":    row.get("count", 0),
            })
            if len(species) >= limit:
                break
        print(f"  discovered {len(species)} species...", flush=True)
        page += 1
        time.sleep(RATE_LIMIT)
    return species


def fetch_observation_photos(taxon_id: int, place_id: int, max_results: int) -> list[dict]:
    """Fetch up to max_results research-grade, CC-licensed photo URLs for a taxon."""
    photos = []
    page = 1
    per_page = min(200, max_results)
    while len(photos) < max_results:
        try:
            data = api_get(f"{API}/observations", {
                "taxon_id":      taxon_id,
                "place_id":      place_id,
                "quality_grade": "research",
                "photos":        "true",
                "photo_license": "cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa,cc0",
                "per_page":      per_page,
                "page":          page,
                "order":         "votes",
                "order_by":      "votes",
            })
        except Exception as e:
            print(f"    API error (page {page}): {e}")
            break

        results = data.get("results", [])
        if not results:
            break
        for obs in results:
            obs_photos = obs.get("photos", [])
            if not obs_photos:
                continue
            url = obs_photos[0].get("url", "")
            if not url:
                continue
            url = url.replace("/square.", "/medium.")   # 75px -> 500px
            photos.append({
                "photo_url": url,
                "obs_id":    obs.get("id"),
                "license":   obs_photos[0].get("license_code", "unknown"),
            })
            if len(photos) >= max_results:
                break
        if len(results) < per_page:
            break
        page += 1
        time.sleep(RATE_LIMIT)
    return photos


def download_image(url: str, dest: Path) -> bool:
    """Download, center-crop to square, resize, and save one image."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as resp:
            img_bytes = resp.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        short = min(w, h)
        left, top = (w - short) // 2, (h - short) // 2
        img = img.crop((left, top, left + short, top + short))
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        img.save(dest, "JPEG", quality=92)
        return True
    except Exception:
        return False


# ── Main ───────────────────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_").replace("'", "")


def main():
    print("=" * 60)
    print("NatureDex AI — Dataset Downloader v2 (auto-discovery)")
    print("=" * 60)

    # 1) Resolve the correct NC place_id
    print(f"\nResolving iNaturalist place_id for '{STATE_NAME}'...")
    place_id = resolve_place_id(STATE_NAME)
    print(f"  -> place_id = {place_id}")
    print("  (sanity check: open "
          f"https://www.inaturalist.org/observations?place_id={place_id} "
          "— it should show NC)")

    # 2) Discover the top species
    print(f"\nDiscovering up to {MAX_SPECIES} most-observed NC species...")
    candidates = get_top_species(place_id, MAX_SPECIES)
    print(f"  Found {len(candidates)} candidate species.")
    print(f"  Keeping only those with >= {MIN_IMAGES} usable images.\n")

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "train").mkdir(exist_ok=True)
    (OUTPUT_DIR / "val").mkdir(exist_ok=True)

    log_rows = []
    label_map = {}
    skipped = []
    downloaded_total = 0

    for i, sp in enumerate(candidates):
        common, scientific = sp["common_name"], sp["scientific_name"]
        taxon_id, group = sp["taxon_id"], sp["iconic_group"]
        slug = slugify(common)
        print(f"\n[{i+1}/{len(candidates)}] {common} ({scientific}) — "
              f"{sp['observations']} obs")

        time.sleep(RATE_LIMIT)
        photos = fetch_observation_photos(taxon_id, place_id, IMAGES_PER_SPECIES)

        if len(photos) < MIN_IMAGES:
            print(f"  skip — only {len(photos)} images (need {MIN_IMAGES})")
            skipped.append(common)
            continue

        print(f"  {len(photos)} photos — downloading...")
        random.shuffle(photos)
        n_val = max(1, int(len(photos) * VAL_SPLIT))
        splits = [("val", photos[:n_val]), ("train", photos[n_val:])]

        label_map[slug] = {
            "common_name":     common,
            "scientific_name": scientific,
            "taxon_id":        taxon_id,
            "iconic_group":    group,
        }

        for split, split_photos in splits:
            split_dir = OUTPUT_DIR / split / slug
            split_dir.mkdir(parents=True, exist_ok=True)
            ok = 0
            for j, photo in enumerate(tqdm(split_photos, desc=f"  {split}", leave=False)):
                dest = split_dir / f"{j:04d}.jpg"
                if dest.exists():
                    ok += 1
                    continue
                if download_image(photo["photo_url"], dest):
                    ok += 1
                    log_rows.append({
                        "split": split, "species_slug": slug,
                        "common_name": common, "scientific": scientific,
                        "taxon_id": taxon_id, "obs_id": photo["obs_id"],
                        "license": photo["license"], "url": photo["photo_url"],
                        "file": str(dest),
                    })
                time.sleep(0.1)
            downloaded_total += ok
            print(f"    {split}: {ok}/{len(split_photos)}")

    # Save metadata (same format as v1)
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "split", "species_slug", "common_name", "scientific",
            "taxon_id", "obs_id", "license", "url", "file"])
        writer.writeheader()
        writer.writerows(log_rows)
    with open(OUTPUT_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Species included:        {len(label_map)}")
    print(f"  Total images downloaded: {downloaded_total}")
    if skipped:
        print(f"  Skipped (too few images): {len(skipped)}")
    print(f"  Label map:               {OUTPUT_DIR / 'label_map.json'}")
    print("\nNext: copy the dataset/ folder to your Windows PC and run train_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()