import os
import sys
import subprocess
from multiprocessing import Pool, cpu_count
try:
    from tqdm import tqdm
except ImportError:
    print("Error: This script requires 'tqdm'. Please run: pip install tqdm")
    sys.exit(1)

def convert_single_file(eps_path):
    """
    Worker function to convert a single EPS file.
    Returns (True, eps_path) on success, or (False, eps_path) on failure.
    """
    try:
        # Run epstopdf silently
        subprocess.run(
            ["epstopdf", eps_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return (True, eps_path)
    except subprocess.CalledProcessError:
        return (False, eps_path)

def main():
    # 1. Determine target directory
    target_folder = sys.argv[1] if len(sys.argv) > 1 else "."
    
    if not os.path.isdir(target_folder):
        print(f"Error: Directory '{target_folder}' not found.")
        sys.exit(1)

    # 2. Collect all .eps files first (fast scan)
    eps_files = []
    print(f"Scanning '{target_folder}' for .eps files...", end=" ", flush=True)
    
    for dirpath, _, filenames in os.walk(target_folder):
        for filename in filenames:
            if filename.lower().endswith(".eps"):
                eps_files.append(os.path.join(dirpath, filename))
    
    print(f"Found {len(eps_files)} files.\n")

    if not eps_files:
        print("No EPS files found to convert.")
        sys.exit(0)

    # 3. Setup Multiprocessing
    # Leave 1 core free so your computer doesn't freeze completely
    num_workers = max(1, cpu_count() - 1)
    
    print(f"Starting conversion with {num_workers} worker processes...")

    # 4. Run conversion with Progress Bar
    # pool.imap_unordered is ideal here because we don't care about the order 
    # they finish, just that they all get done.
    success_count = 0
    fail_list = []

    with Pool(processes=num_workers) as pool:
        # tqdm wraps the iterator to create the bar
        # unit="plot" changes the speed text to "X plots/s"
        for success, path in tqdm(pool.imap_unordered(convert_single_file, eps_files), 
                                  total=len(eps_files), 
                                  unit="plot"):
            if success:
                success_count += 1
            else:
                fail_list.append(path)

    # 5. Final Summary
    print(f"\nDone! Successfully converted {success_count}/{len(eps_files)} files.")
    
    if fail_list:
        print(f"\n[Warning] {len(fail_list)} files failed to convert:")
        for f in fail_list:
            print(f"  - {f}")

if __name__ == "__main__":
    main()
