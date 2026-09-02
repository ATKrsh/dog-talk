import os
import urllib.request
import zipfile
import sys

def download_progress(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = f"\rDownloading dataset: {percent:5.1f}% [{read_so_far / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB]"
        sys.stdout.write(s)
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rDownloading dataset: {read_so_far / 1024 / 1024:.1f} MB")
        sys.stdout.flush()

def main():
    dest_dir = "C:\\Dump\\datasets"
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "dog-pose.zip")
    extract_path = os.path.join(dest_dir, "dog-pose")

    url = "https://ultralytics.com/assets/dog-pose.zip"
    
    print(f"Downloading {url} to {zip_path}...")
    urllib.request.urlretrieve(url, zip_path, download_progress)
    print("\nExtraction started...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
        
    print(f"Extraction completed to {extract_path}")
    
    # Remove zip to free space
    try:
        os.remove(zip_path)
    except:
        pass

if __name__ == "__main__":
    main()
