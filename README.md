# Traffic-Data-Collection
🚦 Neon Traffic Analyzer

An automated, cloud-native traffic congestion monitoring system powered by Computer Vision and Serverless PostgreSQL.

📖 Overview

The Neon Traffic Analyzer is an end-to-end data pipeline designed to monitor, record, and visualize urban traffic congestion in real-time.

Unlike traditional API-based solutions that can be expensive, this project utilizes a computer vision approach. It autonomously captures visual traffic data, processes it to extract geospatial congestion metrics, and stores the results in a serverless cloud database for historical analysis.

🏗️ Architecture

The system operates on a 15-minute automated cron schedule:

Data Acquisition (GitHub Actions): A cloud runner launches a headless Chrome browser (Selenium) to capture high-resolution snapshots of the target region from Google Maps Traffic layer.

Computer Vision Processing (OpenCV): * The image is processed to isolate specific color ranges (HSV) representing heavy traffic (Red/Dark Red).

Contours are extracted and filtered to remove noise (labels, icons).

Geospatial Mapping (OSMnx): * Pixel coordinates of traffic jams are mapped to real-world Latitude/Longitude coordinates.

These coordinates are matched to the nearest road segments using the OpenStreetMap (OSM) graph network.

Storage (Neon DB): The structured data (Road Name, Lat/Lon, Segment Length, Timestamp) is pushed to a Neon Serverless PostgreSQL database.

Visualization (Flask + Leaflet): A local dashboard fetches the data via a REST API to provide interactive maps and historical insights.

🛠️ Tech Stack

Automation: GitHub Actions (Cron Scheduler)

Core Script: Python 3.10

Computer Vision: OpenCV, NumPy

Geospatial Analysis: OSMnx, NetworkX, Geopy

Browser Automation: Selenium WebDriver

Database: Neon (Serverless PostgreSQL)

Frontend: Flask (Backend), HTML5, TailwindCSS, Leaflet.js, Chart.js

🚀 Setup & Installation

Prerequisites

Python 3.10+

A Neon Database Connection String

1. Clone the Repository

git clone [https://github.com/yourusername/Traffic-Data-Collection.git](https://github.com/yourusername/Traffic-Data-Collection.git)
cd Traffic-Data-Collection


2. Install Dependencies

pip install -r requirements_local.txt


3. Environment Configuration

Create a .env file in the root directory and add your Neon connection string:

NEON_DB_URL=postgres://neondb_owner:YOUR_PASSWORD@ep-xyz.aws.neon.tech/neondb?sslmode=require


4. Run the Dashboard

python app.py


Access the dashboard at http://127.0.0.1:5000.

📊 Dashboard Features

Live Traffic View: Interactive map showing currently congested segments.

Time Slider: Filter historical traffic data by hour of the day.

Historical Analysis: Charts showing peak congestion hours and "Top 5 Worst Roads."

Interactive Insights: Click any road on the map to view its specific congestion history.

Dark Mode UI: Optimized for low-light monitoring environments.

📂 Project Structure

├── .github/workflows/   # CI/CD Automation scripts
├── templates/           # Frontend HTML Dashboard
├── app.py               # Flask Server (API)
├── main.py              # Core Data Collection Script
├── map.osm.zip          # Compressed Road Network Graph
└── requirements.txt     # Dependencies


Developed for OELP Semester 5 Project under Prof. B.K. Bhavathrathan, Indian Institute of Technology, Palakkad.
