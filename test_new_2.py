import numpy as np
import math
from pyproj import Transformer

def get_rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    """
    สร้าง rotation matrix สำหรับ gimbal
    yaw: หมุนรอบแกน Z (ซ้าย-ขวา)
    pitch: หมุนรอบแกน Y (ขึ้น-ลง) 
    roll: หมุนรอบแกน X (เอียงซ้าย-ขวา)
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    
    # Rotation matrices
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll),  math.cos(roll)]
    ])
    
    # การคูณ matrix ตามลำดับ: Rz * Ry * Rx
    return Rz @ Ry @ Rx

def pixel_to_latlon_with_gimbal(
    drone_lat, drone_lon, drone_alt,
    gimbal_yaw, gimbal_pitch, gimbal_roll,
    image_width, image_height,
    pixel_x, pixel_y,
    # image_width=640, image_height=512,
    hfov_deg=40.0, vfov_deg=32.0,
    ground_z=0
):
    """
    แปลงพิกัด pixel เป็น lat/lon โดยคำนึงถึงการหมุนของ gimbal
    """
    # จุดกึ่งกลางของภาพ
    cx = image_width / 2
    cy = image_height / 2
    
    # แปลงพิกัด pixel เป็นมุม (relative to center)
    angle_x = (pixel_x - cx) / cx * (hfov_deg / 2)
    angle_y = -(pixel_y - cy) / cy * (vfov_deg / 2)  # เครื่องหมายลบเพราะ Y axis กลับกัน
    
    print(f"Pixel angles: X={angle_x:.2f}°, Y={angle_y:.2f}°")
    
    # สร้าง ray vector ใน camera coordinate system
    # กล้องมองไปทิศ +Z, X ไปขวา, Y ไปขึ้น
    ray_x = math.tan(math.radians(angle_x))
    ray_y = math.tan(math.radians(angle_y))
    ray_z = 1.0
    
    ray_camera = np.array([ray_x, ray_y, ray_z])
    ray_camera = ray_camera / np.linalg.norm(ray_camera)  # normalize
    
    print(f"Ray in camera frame: {ray_camera}")
    
    # แปลง ray จาก camera frame เป็น world frame ด้วย gimbal rotation
    R = get_rotation_matrix(gimbal_yaw, gimbal_pitch, gimbal_roll)
    ray_world = R @ ray_camera
    
    print(f"Ray in world frame: {ray_world}")
    print(f"Ray Z component: {ray_world[2]} (should be negative if pointing down)")
    
    # แปลงพิกัด drone จาก WGS84 เป็น UTM
    transformer = Transformer.from_crs("epsg:4326", "epsg:32647")  # UTM Zone 47N for Thailand
    x_drone, y_drone = transformer.transform(drone_lat, drone_lon)
    z_drone = drone_alt
    
    drone_pos = np.array([x_drone, y_drone, z_drone])
    print(f"Drone position (UTM): {drone_pos}")
    
    # คำนวณจุดตัดระหว่าง ray กับพื้นดิน (z = ground_z)
    if abs(ray_world[2]) < 1e-10:
        raise ValueError("Ray is parallel to ground - cannot find intersection")
    
    t = (ground_z - z_drone) / ray_world[2]
    
    if t < 0:
        print("Warning: Ray is pointing upward or intersection is behind camera")
    
    print(f"Ray parameter t: {t}")
    
    # คำนวณจุดบนพื้นดิน
    ground_point = drone_pos + t * ray_world
    print(f"Ground intersection (UTM): {ground_point}")
    
    # แปลงกลับเป็น lat/lon
    transformer_back = Transformer.from_crs("epsg:32647", "epsg:4326")
    lat_target, lon_target = transformer_back.transform(ground_point[0], ground_point[1])
    
    return lat_target, lon_target

# ทดสอบ
if __name__ == "__main__":
    print("=== Testing pixel to lat/lon conversion ===")
    
    # พิกัดโดรน (กรุงเทพฯ)
    drone_lat = 13.7563
    drone_lon = 100.5018
    drone_alt = 90  # เมตร
    
    # การตั้งค่า gimbal
    gimbal_yaw = 0      # หันหน้าไปทิศเหนือ (0°)
    gimbal_pitch = -89.9  # ชี้ลงตรง (-90°)
    gimbal_roll = 0     # ไม่เอียง
    
    # พิกัด pixel ที่ต้องการแปลง (จุดกึ่งกลางภาพ)
    pixel_x = 320
    pixel_y = 256
    
    try:
        lat, lon = pixel_to_latlon_with_gimbal(
            drone_lat=drone_lat,
            drone_lon=drone_lon,
            drone_alt=drone_alt,
            gimbal_yaw=gimbal_yaw,
            gimbal_pitch=gimbal_pitch,
            gimbal_roll=gimbal_roll,
            image_width=640,
            image_height=512,
            pixel_x=pixel_x,
            pixel_y=pixel_y
        )
        
        print(f"\n=== Result ===")
        print(f"Drone position: {drone_lat:.6f}, {drone_lon:.6f}")
        print(f"Target pixel ({pixel_x}, {pixel_y}) maps to:")
        print(f"Latitude: {lat:.6f}")
        print(f"Longitude: {lon:.6f}")
        
        # คำนวณระยะห่างจากโดรน
        # from geopy.distance import geodesic
        # distance = geodesic((drone_lat, drone_lon), (lat, lon)).meters
        # print(f"Distance from drone: {distance:.2f} meters")
        
    except Exception as e:
        print(f"Error: {e}")