#!/usr/bin/env python3
"""
Automatic Image Downloader for Training Data
Downloads images from multiple sources and organizes them automatically
"""

import os
import sys
import requests
import time
import json
from pathlib import Path
from urllib.parse import quote_plus
import random

# Configuration
BASE_DIR = Path("test data")
TIMEOUT = 15
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Search queries for each category
UI_GOOD_QUERIES = [
    "touchdesigner node graph screenshot",
    "comfyui workflow interface",
    "resolume vj software interface",
    "cinema 4d viewport screenshot",
    "unreal engine blueprint nodes",
    "visual studio code programming",
    "python code editor screenshot",
    "blender node editor",
    "max msp patcher",
    "ableton live arrangement view",
    "obs studio interface",
    "davinci resolve timeline",
    "after effects timeline",
    "figma design interface",
    "unity editor screenshot"
]

UI_BAD_QUERIES = [
    "cluttered desktop screenshot",
    "messy computer screen",
    "blurry software interface",
    "zoomed out desktop",
]

ART_GOOD_QUERIES = [
    "generative art abstract",
    "motion graphics render",
    "vj visuals fullscreen",
    "procedural art output",
    "creative coding visualization",
    "abstract 3d render",
    "digital art colorful",
    "shader art",
    "fractal art",
    "algorithmic art",
    "audio reactive visuals",
    "particle system art",
    "glitch art aesthetic",
    "cyberpunk neon art",
    "geometric abstract art"
]

ART_BAD_QUERIES = [
    "low quality render",
    "unfinished 3d model",
    "pixelated digital art",
]

def download_image(url, save_path, timeout=TIMEOUT):
    """Download an image from URL"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Verify it's actually an image by checking file size
        if os.path.getsize(save_path) < 1000:  # Less than 1KB, probably not a real image
            os.remove(save_path)
            return False

        return True
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

def search_unsplash(query, per_page=30):
    """Search Unsplash for images (no API key required for basic search)"""
    try:
        # Unsplash public search endpoint
        url = f"https://unsplash.com/napi/search/photos?query={quote_plus(query)}&per_page={per_page}"
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            images = []
            for item in data.get('results', []):
                images.append({
                    'url': item['urls']['regular'],
                    'thumb': item['urls']['small'],
                })
            return images
    except Exception as e:
        print(f"  WARNING: Unsplash search failed: {e}")
    return []

def search_duckduckgo_images(query, max_results=30):
    """Search DuckDuckGo for images (no API key required)"""
    try:
        url = "https://duckduckgo.com/"
        params = {'q': query, 'iax': 'images', 'ia': 'images'}

        # Get search page
        response = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)

        # Get vqd token from page
        import re
        vqd_match = re.search(r'vqd=([\d-]+)&', response.text)
        if not vqd_match:
            return []

        vqd = vqd_match.group(1)

        # Get actual image results
        params = {
            'l': 'us-en',
            'o': 'json',
            'q': query,
            'vqd': vqd,
            'f': ',,,',
            'p': '1',
            'v7exp': 'a'
        }

        response = requests.get('https://duckduckgo.com/i.js', params=params, headers=HEADERS, timeout=TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            images = []
            for item in data.get('results', [])[:max_results]:
                images.append({
                    'url': item.get('image'),
                    'thumb': item.get('thumbnail'),
                })
            return images
    except Exception as e:
        print(f"  WARNING: DuckDuckGo search failed: {e}")
    return []

def search_pexels(query, api_key, per_page=30):
    """Search Pexels for images (requires free API key)"""
    if not api_key:
        return []

    try:
        url = f"https://api.pexels.com/v1/search?query={quote_plus(query)}&per_page={per_page}"
        headers = {**HEADERS, 'Authorization': api_key}
        response = requests.get(url, headers=headers, timeout=TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            images = []
            for item in data.get('photos', []):
                images.append({
                    'url': item['src']['large'],
                    'thumb': item['src']['small'],
                })
            return images
    except Exception as e:
        print(f"  WARNING: Pexels search failed: {e}")
    return []

def download_category(queries, target_folder, target_count, category_name, pexels_api_key=None):
    """Download images for a category"""

    print(f"\n{'='*60}")
    print(f"Downloading {category_name}")
    print(f"   Target: {target_count} images")
    print(f"   Folder: {target_folder}")
    print(f"{'='*60}\n")

    target_folder.mkdir(parents=True, exist_ok=True)

    downloaded_count = 0
    query_index = 0

    while downloaded_count < target_count and query_index < len(queries):
        query = queries[query_index]
        print(f"\nSearching: '{query}'")

        # Try different search methods
        all_images = []

        # Try Unsplash first
        print("   -> Trying Unsplash...")
        unsplash_results = search_unsplash(query, per_page=20)
        all_images.extend(unsplash_results)
        print(f"   OK: Found {len(unsplash_results)} from Unsplash")

        # Try Pexels if API key provided
        if pexels_api_key:
            print("   -> Trying Pexels...")
            pexels_results = search_pexels(query, pexels_api_key, per_page=20)
            all_images.extend(pexels_results)
            print(f"   OK: Found {len(pexels_results)} from Pexels")

        # Try DuckDuckGo
        print("   -> Trying DuckDuckGo...")
        ddg_results = search_duckduckgo_images(query, max_results=20)
        all_images.extend(ddg_results)
        print(f"   OK: Found {len(ddg_results)} from DuckDuckGo")

        # Shuffle to get variety
        random.shuffle(all_images)

        # Download images from this query
        images_from_query = 0
        max_per_query = min(8, target_count - downloaded_count)  # Max 8 per query

        for img in all_images:
            if images_from_query >= max_per_query:
                break

            if downloaded_count >= target_count:
                break

            url = img.get('url')
            if not url:
                continue

            # Generate filename
            filename = f"{category_name.lower().replace(' ', '_')}_{downloaded_count + 1:03d}.jpg"
            save_path = target_folder / filename

            print(f"   Downloading {downloaded_count + 1}/{target_count}...", end=' ')

            if download_image(url, save_path):
                print(f"OK: {filename}")
                downloaded_count += 1
                images_from_query += 1
                time.sleep(0.5)  # Be polite to servers
            else:
                print(f"FAILED")

        query_index += 1
        time.sleep(1)  # Pause between queries

    print(f"\nDownloaded {downloaded_count}/{target_count} images for {category_name}")
    return downloaded_count

def main():
    """Main function"""

    print("""
============================================================
     Automatic Training Image Downloader
     Dual-Model Training Data Collection
============================================================
    """)

    # Check for Pexels API key (optional)
    pexels_api_key = os.environ.get('PEXELS_API_KEY')
    if not pexels_api_key:
        print("INFO: No Pexels API key found (optional)")
        print("   To use Pexels, get free key at: https://www.pexels.com/api/")
        print("   Then set: export PEXELS_API_KEY='your-key'\n")
    else:
        print("OK: Pexels API key detected\n")

    print("This script will download ~200 training images:")
    print("  - 75 UI good examples")
    print("  - 25 UI bad examples")
    print("  - 75 Art good examples")
    print("  - 25 Art bad examples")
    print("\nThis will take approximately 15-30 minutes.\n")

    input("Press Enter to start downloading...")

    start_time = time.time()

    # Download UI Good Images
    download_category(
        UI_GOOD_QUERIES,
        BASE_DIR / 'ui_training_data' / 'good',
        75,
        "UI Good",
        pexels_api_key
    )

    # Download UI Bad Images
    download_category(
        UI_BAD_QUERIES,
        BASE_DIR / 'ui_training_data' / 'bad',
        25,
        "UI Bad",
        pexels_api_key
    )

    # Download Art Good Images
    download_category(
        ART_GOOD_QUERIES,
        BASE_DIR / 'art_training_data' / 'good',
        75,
        "Art Good",
        pexels_api_key
    )

    # Download Art Bad Images
    download_category(
        ART_BAD_QUERIES,
        BASE_DIR / 'art_training_data' / 'bad',
        25,
        "Art Bad",
        pexels_api_key
    )

    elapsed_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE!")
    print(f"{'='*60}")
    print(f"Time taken: {elapsed_time/60:.1f} minutes")
    print(f"Images saved to: {BASE_DIR.absolute()}")
    print(f"\nNext steps:")
    print(f"   1. Review images and remove any that don't fit")
    print(f"   2. Add more images manually if needed")
    print(f"   3. Train your models:")
    print(f'      python train_custom_model.py --training-data "./test data/ui_training_data/" --output ui_model.pth')
    print(f'      python train_custom_model.py --training-data "./test data/art_training_data/" --output art_model.pth')

if __name__ == "__main__":
    main()
