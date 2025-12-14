import os
import csv
import requests
import hashlib
from pathlib import Path
from serpapi import GoogleSearch

CLASSES = ["stop_sign", "yield_sign", "speed_limit_sign", "apple", "banana", "orange"]
DATA_DIR = Path("data/raw")


def get_api_key():
    key = os.environ.get("SERPAPI_KEY")
    if not key:
        raise ValueError("SERPAPI_KEY environment variable not set")
    return key


def search_images(query, api_key, max_results=350):
    all_results = []
    page = 0
    
    while len(all_results) < max_results:
        params = {
            "engine": "google_images",
            "q": query,
            "api_key": api_key,
            "ijn": str(page),  # Image JSON page number for pagination
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        images = results.get("images_results", [])
        
        if not images:
            break
        
        all_results.extend(images)
        print(f"    Page {page}: fetched {len(images)} results (total: {len(all_results)})")
        page += 1
        
        if page >= 5:  # safety limit
            break
    
    return all_results[:max_results]


def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()
        
        content_type = response.headers.get("content-type", "")
        if "image" not in content_type:
            return False
        
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    [FAILED] {url[:60]}... - {e}")
        return False


def get_file_extension(url):
    url_lower = url.lower()
    if ".png" in url_lower:
        return ".png"
    elif ".gif" in url_lower:
        return ".gif"
    elif ".webp" in url_lower:
        return ".webp"
    return ".jpg"


def scrape_class_images(class_name, api_key, max_images, seen_urls):
    class_dir = DATA_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = class_dir / f"{class_name}_sources.csv"
    csv_rows = []
    
    if csv_path.exists():  # resume from previous run
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen_urls.add(row["image_url"])
                csv_rows.append(row)
        print(f"  Loaded {len(csv_rows)} existing entries from CSV")
    
    query = class_name.replace("_", " ")
    print(f"  Searching Google Images for '{query}'...")
    results = search_images(query, api_key, max_results=max_images + 100)
    print(f"  Total results fetched: {len(results)}")
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for result in results:
        if downloaded >= max_images:
            break
        
        image_url = result.get("original")
        source = result.get("link", "")
        
        if not image_url:
            continue
        
        if image_url in seen_urls:  # skip duplicates
            skipped += 1
            continue
        
        url_hash = hashlib.md5(image_url.encode()).hexdigest()[:12]
        ext = get_file_extension(image_url)
        filename = f"{class_name}_{url_hash}{ext}"
        save_path = class_dir / filename
        
        print(f"  [{downloaded + 1}/{max_images}] Downloading {filename}...", end=" ")
        if download_image(image_url, save_path):
            seen_urls.add(image_url)
            csv_rows.append({
                "filename": filename,
                "image_url": image_url,
                "source": source
            })
            downloaded += 1
            print("OK")
        else:
            failed += 1
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "image_url", "source"])
        writer.writeheader()
        writer.writerows(csv_rows)
    
    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "total": len(csv_rows)
    }


def scrape_images(max_images_per_class=350):
    api_key = get_api_key()
    seen_urls = set()
    results_summary = {}
    
    print("=" * 60)
    print("SerpAPI Image Scraper")
    print(f"Classes: {', '.join(CLASSES)}")
    print(f"Max images per class: {max_images_per_class}")
    print("=" * 60)
    
    for class_name in CLASSES:
        print(f"\n[{class_name}]")
        stats = scrape_class_images(class_name, api_key, max_images_per_class, seen_urls)
        results_summary[class_name] = stats
        print(f"  Done: {stats['downloaded']} downloaded, {stats['skipped']} skipped, {stats['failed']} failed")
        print(f"  CSV saved: data/raw/{class_name}/{class_name}_sources.csv")
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Class':<20} {'Downloaded':<12} {'Skipped':<10} {'Failed':<10} {'Total':<10}")
    print("-" * 60)
    
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    total_images = 0
    
    for class_name, stats in results_summary.items():
        print(f"{class_name:<20} {stats['downloaded']:<12} {stats['skipped']:<10} {stats['failed']:<10} {stats['total']:<10}")
        total_downloaded += stats['downloaded']
        total_skipped += stats['skipped']
        total_failed += stats['failed']
        total_images += stats['total']
    
    print("-" * 60)
    print(f"{'TOTAL':<20} {total_downloaded:<12} {total_skipped:<10} {total_failed:<10} {total_images:<10}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape images using SerpAPI")
    parser.add_argument("--max-images-per-class", type=int, default=350,
                        help="Maximum images to download per class (default: 350)")
    args = parser.parse_args()
    
    scrape_images(max_images_per_class=args.max_images_per_class)
