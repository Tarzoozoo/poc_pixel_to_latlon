import math
from typing import Tuple
import numpy as np

def pixel_to_latlon(
    center_lat: float,
    center_lon: float,
    altitude_m: float,
    fov_width_deg: float,
    fov_height_deg: float,
    image_width_px: int,
    image_height_px: int,
    pixel_x: int,
    pixel_y: int
) -> Tuple[float, float]:
    """
    แปลงพิกเซลบนภาพความร้อน ไปยังพิกัดละติจูด ลองจิจูด
    """
    # รัศมีโลก (เมตร)
    R = 6378137

    # แปลง FOV เป็น radians
    fov_w_rad = math.radians(fov_width_deg)
    fov_h_rad = math.radians(fov_height_deg)

    # คำนวณความกว้างและความสูงของภาพบนพื้นดิน (เมตร)
    ground_width = 2 * altitude_m * math.tan(fov_w_rad / 2)
    ground_height = 2 * altitude_m * math.tan(fov_h_rad / 2)
    print (f"Ground area: {ground_width * ground_height:.2f} m²")

    # คำนวณขนาดเมตรต่อพิกเซล
    meter_per_px_x = ground_width / image_width_px
    meter_per_px_y = ground_height / image_height_px

    # คำนวณ offset จากระยะ x, y จากจุดศูนย์กลาง (เมตร)
    dx = (pixel_x - image_width_px / 2) * meter_per_px_x
    dy = (pixel_y - image_height_px / 2) * meter_per_px_y

    # แปลงเป็น lat/lon
    delta_lat = (dy / R) * (180 / math.pi)
    delta_lon = (dx / (R * math.cos(math.radians(center_lat)))) * (180 / math.pi)

    lat = center_lat + delta_lat
    lon = center_lon + delta_lon

    return lat, lon

def detected_color_locations(
    center_lat: float,
    center_lon: float,
    altitude_m: float,
    fov_width_deg: float,
    fov_height_deg: float,
    image_width_px: int,
    image_height_px: int,
    detected_pixels: np.ndarray
) -> list:
    """
    Find the latitude and longitude of detected colors based on pixel coordinates.
    """
    locations = []
    for pixel_y, pixel_x in detected_pixels:
        print (len(locations))
        lat, lon = pixel_to_latlon(
            center_lat, center_lon, altitude_m, fov_width_deg, fov_height_deg,
            image_width_px, image_height_px, pixel_x, pixel_y
        )
        locations.append((lat, lon))
    return locations

# Example usage
lat, lon = pixel_to_latlon(
    center_lat=13.7563,
    center_lon=100.5018,
    altitude_m=100,
    fov_width_deg=40.0,
    fov_height_deg=32.0,
    image_width_px=640,
    image_height_px=512,
    pixel_x=639,         # พิกเซลซ้ายบน
    pixel_y=0
)

print(f"พิกัดของพิกเซล (0, 0): {lat:.6f}, {lon:.6f}")
