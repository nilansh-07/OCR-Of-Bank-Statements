# Image Scraper and Downloader Project

## Overview
This project includes two Python scripts:
1. **Bing Image Downloader**: Fetches financial-related images from Bing using the `bing_image_downloader` library.
2. **Cloudinary Downloader**: Downloads images stored on the Cloudinary platform while preserving the folder hierarchy.

---

## Key Features

### Bing Image Downloader
- Searches and downloads images for predefined financial terms (e.g., bank statements, invoices, salary slips).
- Saves images into a structured directory.
- Allows customization of search terms and download limits.

### Cloudinary Downloader
- Fetches images from Cloudinary based on a specified prefix or downloads all available images.
- Maintains the Cloudinary folder hierarchy locally.
- Handles large-scale downloads and assigns unique filenames to images.

---

## How It Works

### Bing Image Downloader
1. Uses the `bing_image_downloader` library to fetch and download images based on search terms.
2. Downloads images to the `Financial Related Data` directory, with each search term creating a subfolder.
3. Logs errors for any failed downloads.

### Cloudinary Downloader
1. Connects to the Cloudinary API using configured credentials.
2. Fetches image metadata, including folder hierarchy and file format.
3. Downloads images locally while preserving the folder structure.

---

## Prerequisites
- Python 3.6 or higher
- Install required libraries:
  ```bash
  pip install -r requirements.txt
