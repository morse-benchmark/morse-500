import os
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
from PIL import Image
import numpy as np

def plot_knot_images(base_directory, output_filename="all_knots_2.pdf"):
    """
    Plot all knot images with one folder per row in a PDF file.
    Each folder's name is shown only for the first image in that folder.
    Handles matplotlib's limitation of 200 subplots per figure.
    
    Parameters:
    base_directory (str): The base directory containing all knot folders
    output_filename (str): Name of the output PDF file
    """
    # Get all knot directories
    knot_folders = []
    for knot_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, knot_name)
        if os.path.isdir(folder_path):
            knot_folders.append((knot_name, folder_path))
    
    # Sort folders alphabetically
    knot_folders.sort()
    
    # Create a list to hold all knots and their images
    all_knots = []
    max_images_per_knot = 0
    
    # Process each knot folder
    for knot_name, image_folder in knot_folders:
        print(f"Processing knot: {knot_name}")
        
        # Get all image files in the folder
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Extract image numbers from filenames (assuming they have R1, R2, etc.)
        numbered_files = []
        for img_file in image_files:
            # Look for R followed by a number in the filename
            match = re.search(r'R(\d+)', img_file)
            if match:
                number = int(match.group(1))
                numbered_files.append((number, img_file))
        
        # Sort by the extracted numbers
        numbered_files.sort()
        
        # Create a list of images for this knot
        knot_images = []
        for _, img_file in numbered_files:
            img_path = os.path.join(image_folder, img_file)
            try:
                img = Image.open(img_path)
                knot_images.append((img_path, img))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        if knot_images:
            all_knots.append((knot_name, knot_images))
            max_images_per_knot = max(max_images_per_knot, len(knot_images))
    
    total_knots = len(all_knots)
    print(f"Total knot types: {total_knots}")
    print(f"Maximum images per knot: {max_images_per_knot}")
    
    # Create PDF to save all images
    with PdfPages(output_filename) as pdf:
        # Calculate maximum number of columns and rows per figure
        # Matplotlib has a limit of 200 subplots per figure
        MAX_SUBPLOTS = 200
        
        # Let's try to fit as many images in a row as possible
        # but limit it to a reasonable number (e.g., 20)
        max_cols = min(20, max_images_per_knot)
        
        # Break up knots into segments for each figure
        for knot_idx in range(0, total_knots, 1):
            knot_name, knot_images = all_knots[knot_idx]
            num_images = len(knot_images)
            
            # We need to break this knot's images into chunks if there are too many
            # to fit in a single row (max_cols)
            for chunk_start in range(0, num_images, max_cols):
                chunk_end = min(chunk_start + max_cols, num_images)
                chunk_images = knot_images[chunk_start:chunk_end]
                chunk_size = len(chunk_images)
                
                # Create a new figure for this row
                fig_width = min(20, chunk_size * 1.5)  # Limit width to 20 inches
                fig_height = 2  # One row gets 2 inches
                
                fig = plt.figure(figsize=(fig_width, fig_height))
                
                # Plot the images for this chunk
                for col_idx, (img_path, img) in enumerate(chunk_images):
                    # Create subplot (1 row, chunk_size columns, position)
                    ax = fig.add_subplot(1, chunk_size, col_idx + 1)
                    
                    # Display image
                    ax.imshow(np.array(img))
                    
                    # Add title only for the first image in the first chunk
                    if col_idx == 0 and chunk_start == 0:
                        ax.set_title(knot_name, fontsize=10, loc='left')
                    
                    # Remove axes ticks for cleaner look
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                # Add text annotation showing which knot this is and which chunk
                if chunk_start > 0:
                    plt.figtext(0.5, 0.01, f"{knot_name} (continued)", 
                              ha='center', fontsize=8)
                
                # Adjust layout and save to PDF
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
                
                print(f"Saved page for knot {knot_idx+1}/{total_knots}: {knot_name} "
                      f"(images {chunk_start+1}-{chunk_end}/{num_images})")
    
    print(f"All images have been saved to {output_filename}")

# Example usage
if __name__ == "__main__":
    # Base directory containing all knot folders
    base_directory = "AnimatedKnots"
    
    # Plot all images
    plot_knot_images(base_directory)