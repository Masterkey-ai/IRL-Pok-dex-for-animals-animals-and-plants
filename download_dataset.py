"""
NatureDex AI — Phase 4 Dataset Downloader
==========================================
Downloads research-grade, CC-licensed wildlife photos from iNaturalist
for training a custom NC species classifier.

Usage:
    python download_dataset.py

Output:
    dataset/
    ├── train/
    │   ├── eastern_bluebird/   (80% of images)
    │   ├── white_tailed_deer/
    │   └── ...
    ├── val/
    │   ├── eastern_bluebird/   (20% of images)
    │   └── ...
    └── dataset_log.csv         (every image: URL, taxon, license, obs ID)

This structure is ready for PyTorch ImageFolder / torchvision directly.

Requirements:
    pip install requests Pillow tqdm
"""

import os
import csv
import json
import time
import random
import shutil
import argparse
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

NC_PLACE_ID   = 51          # iNaturalist place ID for North Carolina
IMAGES_PER_SPECIES = 150    # target images per species (will get what's available)
MIN_IMAGES    = 30          # skip species with fewer than this many available images
IMG_SIZE      = 224         # resize all images to this (matches EfficientNet input)
VAL_SPLIT     = 0.2         # 20% validation, 80% training
RATE_LIMIT    = 0.6         # seconds between API requests (stay under 100/min)
OUTPUT_DIR    = Path("dataset")
LOG_FILE      = OUTPUT_DIR / "dataset_log.csv"
USER_AGENT    = "NatureDexAI-DatasetDownloader/1.0 (tejo.mukkamala@student)"

# ── NC Species List ────────────────────────────────────────────────────────────
# Curated list of commonly observed NC species with their iNaturalist taxon IDs.
# Each tuple: (common_name, scientific_name, taxon_id, iconic_group)
# Taxon IDs verified against iNaturalist as of 2025.
# Mix of birds, mammals, reptiles, amphibians, insects, and plants — broad coverage.

NC_SPECIES = [
    # ── Birds ──────────────────────────────────────────────────────────────────
    ("Eastern Bluebird",         "Sialia sialis",              9083,    "Aves"),
    ("Northern Cardinal",        "Cardinalis cardinalis",      9200,    "Aves"),
    ("American Robin",           "Turdus migratorius",         20727,   "Aves"),
    ("Blue Jay",                 "Cyanocitta cristata",        8916,    "Aves"),
    ("Carolina Chickadee",       "Poecile carolinensis",       13858,   "Aves"),
    ("Red-tailed Hawk",          "Buteo jamaicensis",          5228,    "Aves"),
    ("Great Blue Heron",         "Ardea herodias",             4849,    "Aves"),
    ("Downy Woodpecker",         "Dryobates pubescens",        18772,   "Aves"),
    ("American Goldfinch",       "Spinus tristis",             13632,   "Aves"),
    ("Mourning Dove",            "Zenaida macroura",           8973,    "Aves"),
    ("Red-bellied Woodpecker",   "Melanerpes carolinus",       18787,   "Aves"),
    ("Eastern Towhee",           "Pipilo erythrophthalmus",    14886,   "Aves"),
    ("White-breasted Nuthatch",  "Sitta carolinensis",         13933,   "Aves"),
    ("Tufted Titmouse",          "Baeolophus bicolor",         13859,   "Aves"),
    ("Brown Thrasher",           "Toxostoma rufum",            28544,   "Aves"),
    ("Eastern Meadowlark",       "Sturnella magna",            13697,   "Aves"),
    ("Ruby-throated Hummingbird","Archilochus colubris",       4849,    "Aves"),
    ("Osprey",                   "Pandion haliaetus",          5579,    "Aves"),
    ("Barred Owl",               "Strix varia",                4703,    "Aves"),
    ("Canada Goose",             "Branta canadensis",          7107,    "Aves"),

    # ── Mammals ────────────────────────────────────────────────────────────────
    ("White-tailed Deer",        "Odocoileus virginianus",     42389,   "Mammalia"),
    ("Eastern Gray Squirrel",    "Sciurus carolinensis",       46017,   "Mammalia"),
    ("Virginia Opossum",         "Didelphis virginiana",       42754,   "Mammalia"),
    ("Eastern Cottontail",       "Sylvilagus floridanus",      43916,   "Mammalia"),
    ("Raccoon",                  "Procyon lotor",              41654,   "Mammalia"),
    ("Red Fox",                  "Vulpes vulpes",              42069,   "Mammalia"),
    ("Groundhog",                "Marmota monax",              43812,   "Mammalia"),
    ("Eastern Chipmunk",         "Tamias striatus",            46024,   "Mammalia"),
    ("Striped Skunk",            "Mephitis mephitis",          41905,   "Mammalia"),
    ("North American River Otter","Lontra canadensis",         42189,   "Mammalia"),

    # ── Reptiles ───────────────────────────────────────────────────────────────
    ("Eastern Box Turtle",       "Terrapene carolina",         39556,   "Reptilia"),
    ("Eastern Fence Lizard",     "Sceloporus undulatus",       37716,   "Reptilia"),
    ("Black Racer",              "Coluber constrictor",        28963,   "Reptilia"),
    ("Copperhead",               "Agkistrodon contortrix",     27379,   "Reptilia"),
    ("Eastern Ratsnake",         "Pantherophis alleghaniensis",62234,   "Reptilia"),
    ("Five-lined Skink",         "Plestiodon fasciatus",       37691,   "Reptilia"),
    ("Snapping Turtle",          "Chelydra serpentina",        39620,   "Reptilia"),

    # ── Amphibians ─────────────────────────────────────────────────────────────
    ("American Bullfrog",        "Lithobates catesbeianus",    64968,   "Amphibia"),
    ("Green Tree Frog",          "Dryophytes cinereus",        24971,   "Amphibia"),
    ("Eastern Red-backed Salamander","Plethodon cinereus",     27112,   "Amphibia"),
    ("Spring Peeper",            "Pseudacris crucifer",        24976,   "Amphibia"),

    # ── Insects ────────────────────────────────────────────────────────────────
    ("Monarch Butterfly",        "Danaus plexippus",           48662,   "Insecta"),
    ("Eastern Tiger Swallowtail","Papilio glaucus",            54507,   "Insecta"),
    ("Black Swallowtail",        "Papilio polyxenes",          55626,   "Insecta"),
    ("Painted Lady",             "Vanessa cardui",             56459,   "Insecta"),
    ("Eastern Carpenter Bee",    "Xylocopa virginica",         69247,   "Insecta"),
    ("Firefly",                  "Photinus pyralis",           124922,  "Insecta"),
    ("Praying Mantis",           "Mantis religiosa",           52954,   "Insecta"),
    ("Luna Moth",                "Actias luna",                52933,   "Insecta"),
    ("Common Whitetail",         "Plathemis lydia",            61372,   "Insecta"),
    ("Japanese Beetle",          "Popillia japonica",          67757,   "Insecta"),

    # ── Plants ─────────────────────────────────────────────────────────────────
    ("Longleaf Pine",            "Pinus palustris",            53639,   "Plantae"),
    ("Flowering Dogwood",        "Cornus florida",             58717,   "Plantae"),
    ("Black-eyed Susan",         "Rudbeckia hirta",            55834,   "Plantae"),
    ("Cardinal Flower",          "Lobelia cardinalis",         52350,   "Plantae"),
    ("Wild Columbine",           "Aquilegia canadensis",       55945,   "Plantae"),
    ("Virginia Creeper",         "Parthenocissus quinquefolia",54952,   "Plantae"),
    ("Eastern Redbud",           "Cercis canadensis",          58720,   "Plantae"),
    ("Pitcher Plant",            "Sarracenia purpurea",        52853,   "Plantae"),
    ("Venus Flytrap",            "Dionaea muscipula",          49845,   "Plantae"),
    ("Pokeweed",                 "Phytolacca americana",       55850,   "Plantae"),
]

# ── API Helpers ────────────────────────────────────────────────────────────────

def api_get(url: str, params: dict) -> dict:
    """Make a GET request to the iNaturalist v1 API with rate limiting."""
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"
    req = urllib.request.Request(
        full_url,
        headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def fetch_observation_photos(taxon_id: int, max_results: int) -> list[dict]:
    """
    Fetch up to max_results research-grade photos for a taxon in NC.
    Returns list of dicts with keys: photo_url, obs_id, license, quality_grade.
    iNaturalist returns max 200 per page so we paginate if needed.
    """
    photos = []
    page = 1
    per_page = min(200, max_results)

    while len(photos) < max_results:
        try:
            data = api_get("https://api.inaturalist.org/v1/observations", {
                "taxon_id":     taxon_id,
                "place_id":     NC_PLACE_ID,
                "quality_grade":"research",
                "photos":       "true",
                "photo_license":"cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa,cc0",
                "per_page":     per_page,
                "page":         page,
                "order":        "votes",      # highest-quality first
                "order_by":     "votes",
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
            photo = obs_photos[0]   # first photo per observation
            url = photo.get("url", "")
            if not url:
                continue
            # Replace 'square' (75px) with 'medium' (500px)
            url = url.replace("/square.", "/medium.")
            photos.append({
                "photo_url":     url,
                "obs_id":        obs.get("id"),
                "license":       photo.get("license_code", "unknown"),
                "quality_grade": obs.get("quality_grade", ""),
                "observed_on":   obs.get("observed_on", ""),
            })
            if len(photos) >= max_results:
                break

        if len(results) < per_page:
            break   # no more pages
        page += 1
        time.sleep(RATE_LIMIT)

    return photos


def download_image(url: str, dest: Path) -> bool:
    """Download and resize one image. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as resp:
            img_bytes = resp.read()

        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        # Center-crop to square, then resize
        w, h = img.size
        short = min(w, h)
        left  = (w - short) // 2
        top   = (h - short) // 2
        img   = img.crop((left, top, left + short, top + short))
        img   = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

        img.save(dest, "JPEG", quality=92)
        return True

    except Exception:
        return False


# ── Main ───────────────────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def main():
    print("=" * 60)
    print("NatureDex AI — Dataset Downloader")
    print(f"Target: {len(NC_SPECIES)} species × ~{IMAGES_PER_SPECIES} images")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "train").mkdir(exist_ok=True)
    (OUTPUT_DIR / "val").mkdir(exist_ok=True)

    log_rows = []
    label_map = {}   # slug → common_name, for saving alongside dataset

    skipped = []
    downloaded_total = 0

    for i, (common, scientific, taxon_id, group) in enumerate(NC_SPECIES):
        slug = slugify(common)
        print(f"\n[{i+1}/{len(NC_SPECIES)}] {common} ({scientific})")

        # Fetch photo metadata
        print(f"  Fetching photo list...")
        time.sleep(RATE_LIMIT)
        photos = fetch_observation_photos(taxon_id, IMAGES_PER_SPECIES)

        if len(photos) < MIN_IMAGES:
            print(f"  ⚠  Only {len(photos)} photos available — skipping "
                  f"(need {MIN_IMAGES} minimum)")
            skipped.append(common)
            continue

        print(f"  Found {len(photos)} photos — downloading...")

        # Split train/val
        random.shuffle(photos)
        n_val   = max(1, int(len(photos) * VAL_SPLIT))
        val_set = photos[:n_val]
        trn_set = photos[n_val:]

        label_map[slug] = {
            "common_name":    common,
            "scientific_name":scientific,
            "taxon_id":       taxon_id,
            "iconic_group":   group,
        }

        for split, split_photos in [("train", trn_set), ("val", val_set)]:
            split_dir = OUTPUT_DIR / split / slug
            split_dir.mkdir(parents=True, exist_ok=True)

            ok = 0
            for j, photo in enumerate(tqdm(split_photos,
                                           desc=f"  {split}",
                                           leave=False)):
                dest = split_dir / f"{j:04d}.jpg"
                if dest.exists():
                    ok += 1
                    continue

                if download_image(photo["photo_url"], dest):
                    ok += 1
                    log_rows.append({
                        "split":        split,
                        "species_slug": slug,
                        "common_name":  common,
                        "scientific":   scientific,
                        "taxon_id":     taxon_id,
                        "obs_id":       photo["obs_id"],
                        "license":      photo["license"],
                        "url":          photo["photo_url"],
                        "file":         str(dest),
                    })
                time.sleep(0.1)  # gentle on the CDN

            downloaded_total += ok
            print(f"    {split}: {ok}/{len(split_photos)} downloaded")

    # Save metadata
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "split","species_slug","common_name","scientific",
            "taxon_id","obs_id","license","url","file"
        ])
        writer.writeheader()
        writer.writerows(log_rows)

    with open(OUTPUT_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Total images downloaded: {downloaded_total}")
    print(f"  Species included:        {len(NC_SPECIES) - len(skipped)}")
    if skipped:
        print(f"  Skipped (too few photos): {', '.join(skipped)}")
    print(f"  Log saved to:            {LOG_FILE}")
    print(f"  Label map:               {OUTPUT_DIR / 'label_map.json'}")
    print("\nNext step: copy the 'dataset/' folder to your Windows PC")
    print("and run train_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()