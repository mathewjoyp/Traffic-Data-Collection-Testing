import cv2
import numpy as np
import osmnx as ox
import psycopg2 
import time
import math
import os
import zipfile
import json
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from pathlib import Path

# --- MULTI-REGION CONFIGURATION ---
REGIONS = [
    {
        "id": "region_1", 
        "bbox": (28.64397, 28.62054, 77.23663, 77.20281), # New Delhi
        "zip_file": "map.osm.zip",
        "osm_file": "map.osm"
    },
    {
        "id": "region_2", 
        "bbox": (10.79174, 10.75540, 76.68835, 76.62260), # Palakkad
        "zip_file": "map2.osm.zip",
        "osm_file": "map2.osm"
    }
]

# Shared Settings
SCREENSHOT_FILENAME = "temp_screenshot.png"
OUTPUT_IMAGE_FILENAME = "temp_traffic.png"
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
CROP = (100, 50, 50, 50) # Top, Bottom, Left, Right

# --- GOOGLE MAPS MATH HELPERS ---
class GoogleMapsCalculator:
    """Calculates real-world coordinates from pixel coordinates using Web Mercator."""
    TILE_SIZE = 256

    @staticmethod
    def lat_lon_to_point(lat, lon, zoom):
        """Converts lat/lon to world pixel coordinates."""
        _lat = np.clip(lat, -85.05112878, 85.05112878)
        x = (lon + 180) / 360
        y = (1 - np.log(np.tan(_lat * np.pi / 180) + 1 / np.cos(_lat * np.pi / 180)) / np.pi) / 2
        scale = 2 ** zoom
        return x * GoogleMapsCalculator.TILE_SIZE * scale, y * GoogleMapsCalculator.TILE_SIZE * scale

    @staticmethod
    def point_to_lat_lon(x, y, zoom):
        """Converts world pixel coordinates back to lat/lon."""
        scale = 2 ** zoom
        lon = (x / (GoogleMapsCalculator.TILE_SIZE * scale)) * 360 - 180
        y_norm = 0.5 - (y / (GoogleMapsCalculator.TILE_SIZE * scale))
        lat = 90 - 360 * np.arctan(np.exp(-y_norm * 2 * np.pi)) / np.pi
        return lat, lon

    @staticmethod
    def get_actual_image_bounds(center_lat, center_lon, zoom, window_w, window_h, crop):
        """Calculates the geographic bbox of the CROPPED image."""
        center_x, center_y = GoogleMapsCalculator.lat_lon_to_point(center_lat, center_lon, zoom)

        # Screen Edges in World Pixels
        screen_left_x = center_x - (window_w / 2)
        screen_top_y = center_y - (window_h / 2)

        # Apply Crop Offsets
        img_start_x = screen_left_x + crop[2] 
        img_start_y = screen_top_y + crop[0]  
        img_end_x = screen_left_x + (window_w - crop[3]) 
        img_end_y = screen_top_y + (window_h - crop[1])

        # Convert back to Lat/Lon
        north, west = GoogleMapsCalculator.point_to_lat_lon(img_start_x, img_start_y, zoom)
        south, east = GoogleMapsCalculator.point_to_lat_lon(img_end_x, img_end_y, zoom)

        return (north, south, east, west)

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

def map_pixels_to_geo(pixel_coords, img_shape, actual_bbox):
    img_height, img_width, _ = img_shape
    north, south, east, west = actual_bbox
    
    geo_coords = []
    for x, y in pixel_coords:
        lon = west + (x / img_width) * (east - west)
        lat = north - (y / img_height) * (north - south)
        geo_coords.append((lat, lon))
    return geo_coords

def is_point_in_bbox(lat, lon, bbox):
    # bbox = (North, South, East, West)
    return (bbox[1] <= lat <= bbox[0]) and (bbox[3] <= lon <= bbox[2])

def get_road_network(osm_path, bbox):
    try:
        graph = ox.graph_from_xml(osm_path)
        graph = ox.truncate.truncate_graph_bbox(graph, bbox=bbox, truncate_by_edge=True)
        graph_proj = ox.project_graph(graph)
        return graph, graph_proj
    except Exception as e:
        print(f"Map Load Error: {e}")
        return None, None

def calculate_map_params(bbox):
    north, south, east, west = bbox
    center_lat = (north + south) / 2
    center_lon = (east + west) / 2
    zoom = np.interp(max(north - south, east - west), [0.01, 0.1, 1], [15, 12, 8])
    return center_lat, center_lon, zoom

def take_screenshot(center_lat, center_lon, zoom, filename):
    url = f"https://www.google.com/maps/@{center_lat},{center_lon},{zoom:.2f}z/data=!5m1!1e1"
    
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument(f"--window-size={WINDOW_WIDTH},{WINDOW_HEIGHT}")
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

def process_image(filename):
    img = cv2.imread(filename)
    if img is None: return None, None
    img = img[CROP[0]:-CROP[1], CROP[2]:-CROP[3]] # Apply global crop
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Red Traffic Mask
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=15, minRadius=5, maxRadius=30)
    noise = np.zeros_like(mask)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]: cv2.circle(noise, (i[0], i[1]), i[2]+2, 255, -1)
    
    final_mask = cv2.subtract(mask, noise)
    result = np.zeros_like(img)
    result[final_mask > 0] = (0, 0, 255)
    return result, img.shape

def analyze_and_store(processed_img, img_shape, graph, graph_proj, actual_bbox, user_bbox, timestamp, region_id):
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
        
        # 1. Map pixel to real lat/lon using ACTUAL image bounds
        lat, lon = map_pixels_to_geo([(cX, cY)], img_shape, actual_bbox)[0]
        
        try:
            # 2. Find nearest road
            u, v, _ = ox.distance.nearest_edges(graph, X=[lon], Y=[lat], return_dist=False)[0]
            if (u, v) in processed_edges: continue
            
            # 3. STRICT CHECK: Are both ends of the road inside the USER BBOX?
            u_node = graph.nodes[u]
            v_node = graph.nodes[v]
            
            if not (is_point_in_bbox(u_node['y'], u_node['x'], user_bbox) and 
                    is_point_in_bbox(v_node['y'], v_node['x'], user_bbox)):
                continue

            processed_edges.add((u, v))
            edge = graph_proj.get_edge_data(u, v)[0]
            name = edge.get('name', 'Unnamed')
            if isinstance(name, list): name = name[0]
            
            # --- NEW: EXTRACT ROAD GEOMETRY ---
            if 'geometry' in edge:
                coords = list(edge['geometry'].coords)
                # Convert to [[lat, lon], [lat, lon]]
                geometry_list = [[y, x] for x, y in coords]
            else:
                geometry_list = [[u_node['y'], u_node['x']], [v_node['y'], v_node['x']]]
            
            geometry_json = json.dumps(geometry_list)
            # ----------------------------------

            sql = """INSERT INTO traffic_data 
                     (street_name, segment_length_meters, start_lat, start_lon, end_lat, end_lon, capture_timestamp, region_id, geometry) 
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            cursor.execute(sql, (name, edge.get('length', 0), u_node['y'], u_node['x'], v_node['y'], v_node['x'], timestamp, region_id, geometry_json))
            count += 1
        except Exception as e:
            print(f"Mapping Error: {e}")

    conn.commit()
    conn.close()
    return count

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- 🚦 Starting Precise Multi-Region Job ---")
    script_dir = Path(__file__).parent
    
    for region in REGIONS:
        rid = region['id']
        print(f"\n>>> Processing Region: {rid}")
        
        # 1. Unzip Map
        zip_path = script_dir / region['zip_file']
        osm_path = script_dir / region['osm_file']
        
        if not zip_path.exists():
            print(f"Skipping {rid}: Zip not found.")
            continue
            
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(script_dir)
            
        # 2. Load Graph
        graph, graph_proj = get_road_network(osm_path, region['bbox'])
        if not graph: continue
        
        # 3. Calculate Map Parameters
        center_lat, center_lon, zoom = calculate_map_params(region['bbox'])
        
        # 4. Capture
        if take_screenshot(center_lat, center_lon, zoom, SCREENSHOT_FILENAME):
            proc_img, shape = process_image(SCREENSHOT_FILENAME)
            if proc_img is not None:
                # 5. CALCULATE ACTUAL IMAGE BOUNDS
                actual_bbox = GoogleMapsCalculator.get_actual_image_bounds(
                    center_lat, center_lon, zoom, 
                    WINDOW_WIDTH, WINDOW_HEIGHT, CROP
                )
                
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # 6. Analyze with Strict Boundary Check
                saved = analyze_and_store(proc_img, shape, graph, graph_proj, actual_bbox, region['bbox'], ts, rid)
                print(f"✅ {rid}: Saved {saved} records.")
                
        # Cleanup
        if osm_path.exists(): os.remove(osm_path)

    print("\n--- All Regions Complete ---")
