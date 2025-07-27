import numpy as np
import math

def create_rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    """
    สร้าง rotation matrix สำหรับการหมุนในระบบ NED
    ลำดับการหมุน: Yaw (Z) -> Pitch (Y) -> Roll (X)
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    
    # Rotation matrices for each axis
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                   [math.sin(yaw), math.cos(yaw), 0],
                   [0, 0, 1]])
    
    Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                   [0, 1, 0],
                   [-math.sin(pitch), 0, math.cos(pitch)]])
    
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(roll), -math.sin(roll)],
                   [0, math.sin(roll), math.cos(roll)]])
    
    # Combined rotation matrix (Rz * Ry * Rx)
    R = Rz @ Ry @ Rx
    return R

def pixel_to_latlon(pixel_x, pixel_y, drone_lat, drone_lon, drone_alt,
                   drone_yaw=0, drone_pitch=0, drone_roll=0,
                   gimbal_yaw=0, gimbal_pitch=-90, gimbal_roll=0,
                   debug=False):
    """
    แปลงพิกัด pixel เป็น lat/lon (แก้ไขแล้ว)
    
    Parameters:
    - pixel_x, pixel_y: พิกัดพิกเซล (0,0 = มุมซ้ายบน)
    - drone_lat, drone_lon: ตำแหน่ง GPS ของโดรน
    - drone_alt: ความสูงของโดรน (เมตร)
    - drone_yaw, drone_pitch, drone_roll: ท่าทางของโดรน (องศา)
    - gimbal_yaw, gimbal_pitch, gimbal_roll: ท่าทางของ gimbal (องศา)
    """
    
    # K40T Camera specifications
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 512
    FOV_HORIZONTAL = 40  # degrees
    FOV_VERTICAL = 32    # degrees
    
    # คำนวณขนาดพิกเซลในมุม
    pixel_size_x = FOV_HORIZONTAL / IMAGE_WIDTH   # degrees per pixel
    pixel_size_y = FOV_VERTICAL / IMAGE_HEIGHT     # degrees per pixel
    
    center_x = IMAGE_WIDTH / 2
    center_y = IMAGE_HEIGHT / 2
    
    # คำนวณมุมจากจุดกลางของภาพ
    # สำหรับกล้อง: X+ = right, Y+ = down, Z+ = forward
    x_angle = (pixel_x - center_x) * pixel_size_x  # Right/Left angle
    y_angle = (pixel_y - center_y) * pixel_size_y  # Down/Up angle
    
    if debug:
        print(f"DEBUG: Pixel ({pixel_x}, {pixel_y}) -> Camera angles: X={x_angle:.2f}°, Y={y_angle:.2f}°")
    
    # สร้าง ray vector ในระบบพิกัดกล้อง (Camera frame)
    # Camera coordinate: X+ = right, Y+ = down, Z+ = forward
    x_rad = math.radians(x_angle)
    y_rad = math.radians(y_angle)
    print(f"x_rad: {x_rad}, y_rad: {y_rad}")

    ray_camera = np.array([
        math.sin(x_rad),        # X component (right)
        math.sin(y_rad),        # Y component (down)  
        math.cos(x_rad) * math.cos(y_rad)  # Z component (forward)
    ])
    
    if debug:
        print(f"DEBUG: Ray camera: [{ray_camera[0]:.3f}, {ray_camera[1]:.3f}, {ray_camera[2]:.3f}]")
    
    # Transform จาก Camera frame ไป Body frame
    # สำหรับ gimbal ที่ติดตั้งแบบ standard:
    # Camera forward (Z+) ตรงกับ Body forward (X+) เมื่อ gimbal อยู่ในตำแหน่ง neutral
    ray_body_cam = np.array([
        ray_camera[2],   # Camera Z (forward) -> Body X (forward)
        ray_camera[0],   # Camera X (right) -> Body Y (right)
        ray_camera[1]    # Camera Y (down) -> Body Z (down)
    ])
    
    # จากนั้นใช้ gimbal rotation
    R_gimbal = create_rotation_matrix(gimbal_yaw, gimbal_pitch, gimbal_roll)
    ray_body = R_gimbal @ ray_body_cam
    
    if debug:
        print(f"DEBUG: Ray body (after camera transform): [{ray_body_cam[0]:.3f}, {ray_body_cam[1]:.3f}, {ray_body_cam[2]:.3f}]")
        print(f"DEBUG: Ray body (after gimbal): [{ray_body[0]:.3f}, {ray_body[1]:.3f}, {ray_body[2]:.3f}]")
    
    # Transform จาก Body frame ไป World frame (NED)
    R_drone = create_rotation_matrix(drone_yaw, drone_pitch, drone_roll)
    ray_world = R_drone @ ray_body
    
    if debug:
        print(f"DEBUG: Ray world: [{ray_world[0]:.3f}, {ray_world[1]:.3f}, {ray_world[2]:.3f}]")
    
    # คำนวณจุดตัดกับพื้นดิน
    # ray_world[2] = Down component, ต้องเป็นบวกเพื่อชี้ลงสู่พื้นดิน
    if ray_world[2] <= 0:
        if debug:
            print(f"DEBUG: Ray Z component = {ray_world[2]:.3f} (not pointing down)")
        return None  # Ray ไม่ชี้ลงสู่พื้นดิน
    
    # คำนวณระยะทางไปยังพื้นดิน
    t = drone_alt / ray_world[2]
    
    # คำนวณตำแหน่งบนพื้นดิน (NED coordinates)
    ground_north = ray_world[0] * t
    ground_east = ray_world[1] * t
    
    # แปลงเป็น lat/lon
    # 1 degree latitude ≈ 111,320 meters
    # 1 degree longitude ≈ 111,320 * cos(latitude) meters
    lat_per_meter = 1.0 / 111320.0
    lon_per_meter = 1.0 / (111320.0 * math.cos(math.radians(drone_lat)))
    
    target_lat = drone_lat + (ground_north * lat_per_meter)
    target_lon = drone_lon + (ground_east * lon_per_meter)
    
    # คำนวณระยะทางแนวนอน
    horizontal_distance = math.sqrt(ground_north**2 + ground_east**2)
    
    # Ground Sample Distance
    pixel_distance = math.sqrt((pixel_x - center_x)**2 + (pixel_y - center_y)**2)
    gsd = horizontal_distance / pixel_distance if pixel_distance > 0 else 0
    
    if debug:
        print("=== Debug Information ===")
        print(f"Drone position: ({drone_lat:.6f}, {drone_lon:.6f}) at {drone_alt}m")
        print(f"Drone attitude: heading={drone_yaw}°, pitch={drone_pitch}°, roll={drone_roll}°")
        print(f"Gimbal attitude: yaw={gimbal_yaw}°, pitch={gimbal_pitch}°, roll={gimbal_roll}°")
        print(f"Pixel ({pixel_x}, {pixel_y}) -> Camera angles: ({x_angle:.2f}°, {y_angle:.2f}°)")
        print(f"Ray direction (world NED): [{ray_world[0]:.3f}, {ray_world[1]:.3f}, {ray_world[2]:.3f}]")
        print(f"Ground position (NED): North={ground_north:.2f}m, East={ground_east:.2f}m")
        print(f"Target: ({target_lat:.8f}, {target_lon:.8f})")
        print(f"Horizontal distance: {horizontal_distance:.2f}m")
        print(f"Ground sample distance: {gsd:.3f} m/pixel")
        print("=" * 30)
    
    return {
        'lat': target_lat,
        'lon': target_lon,
        'horizontal_distance': horizontal_distance,
        'gsd': gsd,
        'ray_world': ray_world
    }

# ========== ทดสอบระบบ ==========
if __name__ == "__main__":
    print("=== K40T Thermal Camera Pixel to Lat/Lon Converter (FIXED) ===\n")
    
    # ข้อมูลโดรนตัวอย่าง
    drone_lat = 13.756300
    drone_lon = 100.501800
    drone_alt = 100  # meters
    
    # ท่าทางโดรน (heading north, level)
    drone_yaw = 0    # degrees (north)
    drone_pitch = 0  # degrees (level)
    drone_roll = 0   # degrees (level)
    
    # ท่าทาง gimbal (pointing straight down)
    gimbal_yaw = 0     # degrees
    gimbal_pitch = -90 # degrees (pointing down) - ใช้ -90 สำหรับการชี้ลง
    gimbal_roll = 0    # degrees
    
    # ทดสอบมุมทั้ง 4 มุมและจุดกลาง
    test_points = [
        (0, 0, "มุมซ้ายบน (Top-Left)"),
        (639, 0, "มุมขวาบน (Top-Right)"),
        (0, 511, "มุมซ้ายล่าง (Bottom-Left)"),
        (639, 511, "มุมขวาล่าง (Bottom-Right)"),
        (320, 256, "จุดกลาง (Center)")
    ]
    
    print("=== Corner Points Test ===")
    results = {}
    
    for pixel_x, pixel_y, description in test_points:
        result = pixel_to_latlon(
            pixel_x, pixel_y,
            drone_lat, drone_lon, drone_alt,
            drone_yaw, drone_pitch, drone_roll,
            gimbal_yaw, gimbal_pitch, gimbal_roll,
            debug=True
        )
        
        if result:
            results[description] = result
            print(f"{description}: ({result['lat']:.8f}, {result['lon']:.8f})")
        else:
            print(f"{description}: ไม่สามารถคำนวณได้ (ray ไม่ชี้ลงสู่พื้นดิน)")
        print()
    
    # ตรวจสอบทิศทางเทียบกับจุดกลาง
    if "จุดกลาง (Center)" in results:
        center = results["จุดกลาง (Center)"]
        center_lat = center['lat']
        center_lon = center['lon']
        
        print(f"\n=== ตรวจสอบทิศทางเทียบกับจุดกลาง ===")
        
        for desc, result in results.items():
            if desc != "จุดกลาง (Center)":
                delta_lat = result['lat'] - center_lat
                delta_lon = result['lon'] - center_lon
                
                # กำหนดทิศทาง
                ns_dir = "เหนือ" if delta_lat > 0 else "ใต้"
                ew_dir = "ตะวันออก" if delta_lon > 0 else "ตะวันตก"
                
                print(f"{desc}: {ns_dir}-{ew_dir} (Δlat={delta_lat:.6f}, Δlon={delta_lon:.6f})")
    
    print(f"\n=== Verification ===")
    if "จุดกลาง (Center)" in results:
        center = results["จุดกลาง (Center)"]
        
        # ตรวจสอบว่าจุดกลางอยู่ตรงโดรนหรือไม่
        print(f"จุดกลางของกล้อง: ({center['lat']:.8f}, {center['lon']:.8f})")
        print(f"ตำแหน่งโดรน: ({drone_lat:.8f}, {drone_lon:.8f})")
        
        if abs(center['lat'] - drone_lat) < 0.0001 and abs(center['lon'] - drone_lon) < 0.0001:
            print("✓ จุดกลางอยู่ตรงใต้โดรน (ถูกต้อง)")
        else:
            print("✗ จุดกลางไม่ได้อยู่ตรงใต้โดรน")
        
        # ตรวจสอบทิศทางที่ถูกต้อง
        checks = [
            ("มุมซ้ายบน (Top-Left)", "เหนือ-ตะวันตก"),
            ("มุมขวาบน (Top-Right)", "เหนือ-ตะวันออก"),
            ("มุมซ้ายล่าง (Bottom-Left)", "ใต้-ตะวันตก"),
            ("มุมขวาล่าง (Bottom-Right)", "ใต้-ตะวันออก")
        ]
        
        for desc, expected_dir in checks:
            if desc in results:
                result = results[desc]
                delta_lat = result['lat'] - center['lat']
                delta_lon = result['lon'] - center['lon']
                
                ns_dir = "เหนือ" if delta_lat > 0 else "ใต้"
                ew_dir = "ตะวันออก" if delta_lon > 0 else "ตะวันตก"
                actual_dir = f"{ns_dir}-{ew_dir}"
                
                if actual_dir == expected_dir:
                    print(f"✓ {desc.split('(')[0].strip()}อยู่ทิศ{expected_dir} เมื่อเทียบกับจุดกลาง")
                else:
                    print(f"✗ {desc.split('(')[0].strip()}ควรอยู่ทิศ{expected_dir} แต่ได้ {actual_dir}")