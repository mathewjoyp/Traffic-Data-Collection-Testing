import cv2
import numpy as np
import osmnx as ox
import psycopg2 
import time
import geopy.distance
import os
import zipfile
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from pathlib import Path

# --- MULTI-REGION CONFIGURATION ---
REGIONS = [
    {
        "id": "region_1", # Unique ID for Database
        "bbox": (28.64397, 28.62054, 77.23663, 77.20281), # New Delhi (Current)
        "zip_file": "map.osm.zip",
        "osm_file": "map.osm"
    },
    {
        "id": "region_2", # CHANGE THIS ID
        "bbox": (10.79174, 10.75540, 76.68835, 76.62260), # <--- UPDATE COORDINATES HERE
        "zip_file": "map2.osm.zip", # <--- UPDATE FILENAME HERE
        "osm_file": "map2.osm" # Assuming the file inside the zip is map2.osm
    }
]

# Shared Settings
SCREENSHOT_FILENAME = "temp_screenshot.png"
OUTPUT_IMAGE_FILENAME = "temp_traffic.png"

# --- HELPER FUNCTIONS ---

def get_db_connection():
    try:
        db_url = os.getenv('NEON_DB_URL')
        if not db_url: return None
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as e:
        print(f"DB Connection Error: {e}")
        return None

def map_pixels_to_geo(pixel_coords, img_shape, bbox):
    img_height, img_width, _ = img_shape
    north, south, east, west = bbox
    geo_coords = []
    for x, y in pixel_coords:
        lon = west + (x / img_width) * (east - west)
        lat = north - (y / img_height) * (north - south)
        geo_coords.append((lat, lon))
    return geo_coords

def get_road_network(osm_path, bbox):
    try:
        graph = ox.graph_from_xml(osm_path)
        # We rely on cropping to ensure graph matches bbox
        graph = ox.truncate.truncate_graph_bbox(graph, bbox=bbox, truncate_by_edge=True)
        graph_proj = ox.project_graph(graph)
        return graph, graph_proj
    except Exception as e:
        print(f"Map Load Error: {e}")
        return None, None

def take_screenshot(bbox, filename):
    north, south, east, west = bbox
    center_lat = (north + south) / 2
    center_lon = (east + west) / 2
    zoom = np.interp(max(north - south, east - west), [0.01, 0.1, 1], [15, 12, 8])
    
    url = f"https://www.google.com/maps/@{center_lat},{center_lon},{zoom:.2f}z/data=!5m1!1e1"
    
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)
        time.sleep(20) 
        driver.save_screenshot(filename)
        driver.quit()
        return True
    except Exception:
        return False

def process_image(filename, out_name):
    img = cv2.imread(filename)
    if img is None: return None, None
    img = img[100:-50, 50:-50] # Crop UI
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Red Traffic Mask
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    
    # Remove Noise
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=15, minRadius=5, maxRadius=30)
    noise = np.zeros_like(mask)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]: cv2.circle(noise, (i[0], i[1]), i[2]+2, 255, -1)
    
    final_mask = cv2.subtract(mask, noise)
    result = np.zeros_like(img)
    result[final_mask > 0] = (0, 0, 255)
    return result, img.shape

def analyze_and_store(processed_img, img_shape, graph, graph_proj, bbox, timestamp, region_id):
    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return 0

    conn = get_db_connection()
    if not conn: return 0
    cursor = conn.cursor()
    
    count = 0
    processed_edges = set()

    for cnt in contours:
        if cv2.contourArea(cnt) < 50: continue
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        
        cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        lat, lon = map_pixels_to_geo([(cX, cY)], img_shape, bbox)[0]
        
        u, v, _ = ox.distance.nearest_edges(graph, X=[lon], Y=[lat], return_dist=False)[0]
        if (u, v) in processed_edges: continue
        processed_edges.add((u, v))

        try:
            edge = graph_proj.get_edge_data(u, v)[0]
            name = edge.get('name', 'Unnamed')
            if isinstance(name, list): name = name[0]
            
            sql = """INSERT INTO traffic_data 
                     (street_name, segment_length_meters, start_lat, start_lon, end_lat, end_lon, capture_timestamp, region_id) 
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
            cursor.execute(sql, (name, edge.get('length', 0), graph.nodes[u]['y'], graph.nodes[u]['x'], graph.nodes[v]['y'], graph.nodes[v]['x'], timestamp, region_id))
            count += 1
        except Exception as e:
            print(f"Insert Error: {e}")

    conn.commit()
    conn.close()
    return count

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- 🚦 Starting Multi-Region Job ---")
    script_dir = Path(__file__).parent
    
    for region in REGIONS:
        rid = region['id']
        print(f"\n>>> Processing Region: {rid}")
        
        # 1. Unzip Map
        zip_path = script_dir / region['zip_file']
        osm_path = script_dir / region['osm_file']
        
        if not zip_path.exists():
            print(f"Skipping {rid}: Zip file {region['zip_file']} not found.")
            continue
            
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(script_dir)
            
        # 2. Load Graph
        graph, graph_proj = get_road_network(osm_path, region['bbox'])
        if not graph: continue
        
        # 3. Capture & Process
        if take_screenshot(region['bbox'], SCREENSHOT_FILENAME):
            proc_img, shape = process_image(SCREENSHOT_FILENAME, OUTPUT_IMAGE_FILENAME)
            if proc_img is not None:
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                saved = analyze_and_store(proc_img, shape, graph, graph_proj, region['bbox'], ts, rid)
                print(f"✅ {rid}: Saved {saved} records.")
                
        # Cleanup extracted map to save space for next iteration
        if osm_path.exists(): os.remove(osm_path)

    print("\n--- All Regions Complete ---")
