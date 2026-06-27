from pathlib import Path
import numpy as np

DATA_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data/processed_raw_data/hic_matrix'


# 1. Define the directory containing your .txt files
directory = Path(DATA_DIR)

# 2. Loop through all .txt files in the directory
for txt_path in directory.glob('*.txt'):
    try:
        # Load the data from the text file
        # (Change delimiter=',' if your files are comma-separated)
        data = np.loadtxt(txt_path)

        # Define the new path with the .npy extension
        npy_path = txt_path.with_suffix('.npy')

        # Save the data in NumPy binary format
        np.save(npy_path, data)
        print(f"Converted: {txt_path.name} -> {npy_path.name}")

    except Exception as e:
        print(f"Failed to convert {txt_path.name}: {e}")
