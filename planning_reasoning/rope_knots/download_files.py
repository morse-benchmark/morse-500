# https://www.animatedknots.com/complete-knot-list

import os
import requests
import tempfile
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import urljoin
import csv
import pandas as pd
import os
import subprocess
import time
from tqdm import tqdm 

def create_folder(folder_name):
    """Create a folder if it doesn't exist."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")

def download_image(url, folder_path, headers):
    """Download an image and save it to the specified folder."""
    try:
        response = requests.get(url, stream=True, headers=headers)
        if response.status_code == 200:
            # Extract filename from URL
            filename = url.split('/')[-1]
            file_path = os.path.join(folder_path, filename)
            
            # Save the image
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            
            print(f"Downloaded: {filename}")
            return True
        else:
            print(f"Failed to download {url}, status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def get_random_user_agent():
    """Return a random user agent to avoid detection."""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
    ]
    return random.choice(user_agents)

def scrape_knots():
    """Scrape knot information and download images."""
    base_url = "https://www.animatedknots.com"
    knot_list_url = f"{base_url}/complete-knot-list"
    
    # Create main directory for all knots
    main_dir = "AnimatedKnots"
    create_folder(main_dir)
    
    # Set up headers to mimic a browser
    headers = {
        'User-Agent': get_random_user_agent(),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
    }
    
    # Fetch the knot list page
    try:
        print(f"Fetching the knot list page with headers...")
        response = requests.get(knot_list_url, headers=headers)
        print(f"Status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Failed to fetch the knot list page. Status code: {response.status_code}")
            print("Trying an alternative approach...")
            
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find all anchor tags with the specified class
        knot_links = soup.find_all('a', class_='w-grid-item-anchor')
        
        print(f"Found {len(knot_links)} knot links")
        
        # Extract the URLs and names
        knot_data = []
        for link in knot_links:
            href = link.get('href')
            # Try to extract the knot name from the aria-label or from the URL
            name = link.get('aria-label')
            if not name:
                # Extract name from URL
                name = href.split('/')[-1].replace('-', ' ').title()
            
            knot_data.append({
                'name': name,
                'url': href
            })
            print(f"Found knot: {name} - {href}")
        
        # Save the data to a CSV file
        with open('knot_links.csv', 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['name', 'url']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for data in knot_data:
                writer.writerow(data)
        
        print(f"Saved {len(knot_data)} knot links to knot_links.csv")
        
        print("Scraping complete!")
        return knot_data
        
    except Exception as e:
        print(f"Error accessing the knot list page: {str(e)}")
        print("Falling back to alternative approach...")


def get_knot_urls_and_images(knot_list_url):
    # Set up headers to mimic a browser
    headers = {
        'User-Agent': get_random_user_agent(),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
    }
    
    # Fetch the knot list page
    try:
        print(f"Fetching the knot list page with headers...")
        response = requests.get(knot_list_url, headers=headers)
        print(f"Status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Failed to fetch the knot list page. Status code: {response.status_code}")
            print("Trying an alternative approach...")
            
        knot_soup = BeautifulSoup(response.text, 'html.parser')
        breakpoint()

        knot_img = knot_soup.find('img', id='Knot')


        # Find all anchor tags with the specified class
        # image_link = soup.find_all('a', class_='w-grid-item-anchor')
    except Exception as knot_err:
        print(f"  Error processing knot {knot_list_url}: {str(knot_err)}")


def download_knot_images(base_url, folder_name):
    """
    Download a sequence of knot images by incrementing R1, R2, etc. until a 404 is encountered.
    
    Args:
        base_url: URL of the first image (containing R1)
        folder_name: Name of the folder to save the images
    """
    # Extract the base part of the URL before the R1, R2, etc.
    if "R1" not in base_url:
        print("Error: URL must contain 'R1' to serve as the base URL.")
        return
    
    # Split the URL at R1 to get the prefix and suffix
    prefix = base_url.split("R1")[0]
    suffix = base_url.split("R1")[1]
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    
    # Set up headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.animatedknots.com/'
    }
    
    image_count = 0
    image_num = 1
    
    print(f"Starting to download images from: {base_url}")
    
    while True:
        # Construct the URL for this step
        current_url = f"{prefix}R{image_num}{suffix}"
        
        # Get the filename from the URL
        filename = os.path.basename(current_url)
        save_path = os.path.join(folder_name, filename)
        
        print(f"Trying: {current_url}")
        
        try:
            # Download the image
            response = requests.get(current_url, headers=headers)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Save the image
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"Downloaded: {filename}")
                image_count += 1
                
                # Add a small delay to be respectful to the server
                time.sleep(0.5)
            else:
                print(f"Received status code {response.status_code} for {current_url}")
                # Break the loop if we get a 404 (not found)
                if response.status_code == 404:
                    print(f"Reached the end of the sequence at R{image_num-1}")
                    break
        
        except Exception as e:
            print(f"Error downloading {current_url}: {str(e)}")
            break
        
        # Move to the next image
        image_num += 1
    
    print(f"Download complete! Downloaded {image_count} images to {folder_name}")


def get_html_with_wget_and_parse(url):
    """
    Use wget to download HTML from a URL, then parse it with BeautifulSoup.
    
    Args:
        url: The URL to download and parse
        
    Returns:
        BeautifulSoup object of the parsed HTML
    """
    # Create a temporary file to store the HTML
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
        temp_path = temp_file.name
    
    try:
        # Use wget to download the HTML to the temporary file
        wget_cmd = ['wget', '-q', '-O', temp_path, url]
        subprocess.run(wget_cmd, check=True)
        
        # Read the HTML from the temporary file
        with open(temp_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        return soup
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {url}: {str(e)}")
        return None
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def download_knot_images(base_url, folder_name):
    """
    Download a sequence of knot images using wget by incrementing R1, R2, etc.
    Continues until wget fails to download an image (404 error).
    
    Args:
        base_url: URL of the first image (containing R1)
        folder_name: Name of the folder to save the images
    """
    # Extract the base part of the URL before the R1, R2, etc.
    if "R" not in base_url:
        print("Error: URL must contain 'R' to serve as the base URL.")
        return
    
    # Split the URL at R to get the prefix and suffix
    prefix = base_url.split("R")[0]
    suffix = base_url.split(".")[-1]
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    
    image_count = 0
    image_num = 1
    
    print(f"Starting to download images from: {base_url}")
    
    while True:
        # Construct the URL for this step
        current_url = f"{prefix}R{image_num}.{suffix}"
        
        # Get the filename from the URL
        filename = os.path.basename(current_url)
        save_path = os.path.join(folder_name, filename)
        
        print(f"Trying: {current_url}")
        
        try:
            # Use wget to download the image
            # -q for quiet, --spider to just check if the file exists
            # First check if the file exists
            check_cmd = ['wget', '-q', '--spider', current_url]
            check_result = subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if check_result.returncode == 0:
                # File exists, download it
                download_cmd = ['wget', '-q', '-O', save_path, current_url]
                subprocess.run(download_cmd, check=True)
                
                print(f"Downloaded: {filename}")
                image_count += 1
                
                # Add a small delay to be respectful to the server
                time.sleep(0.5)
            else:
                print(f"File not found: {current_url}")
                print(f"Reached the end of the sequence at R{image_num-1}")
                break
        
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {current_url}: {str(e)}")
            break
        
        # Move to the next image
        image_num += 1
    
    print(f"Download complete! Downloaded {image_count} images to {folder_name}")


if __name__ == "__main__":
    # knot_data = scrape_knots()
    # with open('knot_links.csv', 'r') as file:
    #     csv_reader = csv.reader(file)

    data = pd.read_csv('knot_links.csv')
    for idx, row in tqdm(data.iterrows()):
        # if idx<=2: continue
        name = row["name"]
        url = row["url"]
        print(f"Index: {idx}, {name}, {url}")
        knot_name = url.split('/')[-1].replace('-','_')
        folder_name = f"AnimatedKnots/{knot_name}"

        soup = get_html_with_wget_and_parse(url)
        knot_img = soup.find('img', id='Knot')
        img_url = knot_img.get('src').strip()
        print(img_url)
        download_knot_images(img_url, folder_name)
        # note: some folders contain extra images after ending
        # Adjustable Grip Hitch
        # Crown Sinnet
        # Figure 8 Flake
        # Highwayman’s Hitch
        # Mooring Hitch
        # Ring Hitch
        # Slip Knot
        # Tumble Hitch
