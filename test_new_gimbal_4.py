import numpy as np
import math
from pyproj import Transformer


def get_rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    """
    สร้าง rotation matrix จากมุม yaw, pitch, roll
    สำหรับ gimbal coordinate system
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    # Rotation matrices (standard aerospace convention)
    Rz_yaw = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1],
        ]
    )

    Ry_pitch = np.array(
        [
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)],
        ]
    )

    Rx_roll = np.array(
        [
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)],
        ]
    )

    # Combined rotation: Rz * Ry * Rx
    return Rz_yaw @ Ry_pitch @ Rx_roll


def pixel_to_latlon_k40t_thermal(
    drone_lat,
    drone_lon,
    drone_altitude_m,
    drone_heading_deg,
    drone_pitch_deg,
    drone_roll_deg,
    hfov_deg,
    vfov_deg,
    gimbal_yaw_deg,
    gimbal_pitch_deg,
    gimbal_roll_deg,
    image_width,
    image_height,
    pixel_x,
    pixel_y,
    ground_elevation_m=0,
    utm_zone="32647",  # Thailand UTM Zone 47N
):
    """
    แปลงพิกัด pixel บนภาพ thermal จาก K40T gimbal เป็น lat/lon

    Parameters:
    - drone_lat, drone_lon: พิกัดโดรน (WGS84)
    - drone_altitude_m: ความสูงโดรน (เหนือ MSL)
    - gimbal_yaw_deg: มุม yaw ของ gimbal (0 = หันหน้าเหนือ)
    - gimbal_pitch_deg: มุม pitch ของ gimbal (-90 = ชี้ตรงลง)
    - gimbal_roll_deg: มุม roll ของ gimbal (0 = level)
    - pixel_x, pixel_y: พิกัด pixel ที่ต้องการ
    - image_width, image_height: ขนาดภาพ
    - hfov_deg, vfov_deg: Field of View
    - ground_elevation_m: ความสูงของพื้นดิน (เหนือ MSL)
    - utm_zone: UTM zone (เช่น "32647" สำหรับไทย)
    """

    # 1. แปลง pixel เป็นมุมใน camera frame
    cx = image_width / 2.0
    cy = image_height / 2.0

    # มุมเบี่ยงเบนจากจุดกลาง (radians)
    angle_x_rad = math.atan((pixel_x - cx) / cx * math.tan(math.radians(hfov_deg / 2)))
    # angle_y_rad = math.atan((pixel_y - cy) / cy * math.tan(math.radians(vfov_deg / 2)))   # ของเดิม
    angle_y_rad = math.atan(-(pixel_y - cy) / cy * math.tan(math.radians(vfov_deg / 2)))    
    # coordinate system ของภาพ (ชี้ลง) กับ coordinate system ของโลกจริง (ชี้ขึ้น) มีทิศทางตรงกันข้าม

    # 2. สร้าง ray vector ใน camera coordinate system
    # Camera frame: +X = right, +Y = down, +Z = forward                 ของเดิม
    # Camera frame: +X = right, +Y = up (แก้ไขจาก down), +Z = forward    ของใหม่
    ray_camera = np.array([math.tan(angle_x_rad), math.tan(angle_y_rad), 1.0])
    ray_camera = ray_camera / np.linalg.norm(ray_camera)

    # 3. แปลง ray จาก camera frame เป็น world frame (2 ขั้นตอน)
    # 3.1 จาก camera frame เป็น drone body frame
    R_gimbal = get_rotation_matrix(gimbal_yaw_deg, gimbal_pitch_deg, gimbal_roll_deg)
    ray_body = R_gimbal @ ray_camera
    # 3.2 จาก drone body frame เป็น world frame (NED: North-East-Down)
    R_drone = get_rotation_matrix(drone_heading_deg, drone_pitch_deg, drone_roll_deg)
    ray_world = R_drone @ ray_body

    # 4. แปลงพิกัดโดรนเป็น UTM
    transformer_to_utm = Transformer.from_crs("epsg:4326", f"epsg:{utm_zone}")
    drone_x_utm, drone_y_utm = transformer_to_utm.transform(drone_lat, drone_lon)

    # ตำแหน่งโดรนใน UTM coordinates
    drone_pos_utm = np.array([drone_x_utm, drone_y_utm, drone_altitude_m])

    # 5. คำนวณจุดตัดระหว่าง ray กับพื้นดิน
    print(f"ray_world: {ray_world}, ray_world[2]: {ray_world[2]}")
    print(f"abs(ray_world[2]): {abs(ray_world[2])}")
    if abs(ray_world[2]) < 1e-10:
        raise ValueError("Ray is parallel to ground plane - cannot find intersection")

    # ระยะทางจากโดรนไปยังจุดตัดกับพื้นดิน
    t = (ground_elevation_m - drone_altitude_m) / ray_world[2]

    if t >= 0:
        print(f"Warning: Ray points upward (t={t:.2f}) - unusual for downward gimbal")

    # จุดตัดใน UTM coordinates
    target_utm = drone_pos_utm + t * ray_world

    # 6. แปลงกลับเป็น lat/lon
    transformer_to_wgs84 = Transformer.from_crs(f"epsg:{utm_zone}", "epsg:4326")
    target_lat, target_lon = transformer_to_wgs84.transform(
        target_utm[0], target_utm[1]
    )

    # แสดงข้อมูลเพิ่มเติมสำหรับการ debug
    distance_m = np.linalg.norm(target_utm[:2] - drone_pos_utm[:2])
    height_diff = drone_altitude_m - ground_elevation_m
    print(f"Drone UTM: ({drone_x_utm:.2f}, {drone_y_utm:.2f}, {drone_altitude_m:.2f})")
    print(f"Target UTM: ({target_utm[0]:.2f}, {target_utm[1]:.2f}, {ground_elevation_m:.2f})")
    print(f"Horizontal distance: {distance_m:.2f} m")
    print(f"Ray direction (world): [{ray_world[0]:.3f}, {ray_world[1]:.3f}, {ray_world[2]:.3f}]")
    print(f"=== Debug Information ===")
    print(f"Drone position: ({drone_lat:.6f}, {drone_lon:.6f}) at {drone_altitude_m}m")
    print(f"Drone attitude: heading={drone_heading_deg}°, pitch={drone_pitch_deg}°, roll={drone_roll_deg}°")
    print(f"Gimbal attitude: yaw={gimbal_yaw_deg}°, pitch={gimbal_pitch_deg}°, roll={gimbal_roll_deg}°")
    print(f"Pixel ({pixel_x}, {pixel_y}) -> Angles: ({math.degrees(angle_x_rad):.2f}°, {math.degrees(angle_y_rad):.2f}°)")
    print(f"Ray direction (world NED): [{ray_world[0]:.3f}, {ray_world[1]:.3f}, {ray_world[2]:.3f}]")
    print(f"Target: ({target_lat:.8f}, {target_lon:.8f})")
    print(f"Horizontal distance: {distance_m:.2f}m, Height: {height_diff:.2f}m")
    print(f"Ground sample distance: {distance_m * math.tan(math.radians(hfov_deg/2)) / (image_width/2):.3f} m/pixel")
    print("=" * 30)
    return target_lat, target_lon


# === การใช้งาน ===
if __name__ == "__main__":
    # ตัวอย่างการใช้งาน
    # โดรนบินอยู่เหนือกรุงเทพฯ ความสูง 100 เมตร
    # กล้อง thermal ชี้ตรงลงด้วย active stabilization

    lat, lon = pixel_to_latlon_k40t_thermal(
        drone_lat=13.7563,  # พิกัดโดรน (กรุงเทพฯ)
        drone_lon=100.5018,
        drone_altitude_m=90,  # ความสูง 100 เมตร
        drone_heading_deg=0,        # โดรนหันหน้าไปทิศตะวันออกเฉียงเหนือ
        drone_pitch_deg=0,           # โดรนบินเรียบ (ไม่เอียงหน้า-หลัง)
        drone_roll_deg=0,            # โดรนบินเรียบ (ไม่เอียงซ้าย-ขวา)
        hfov_deg=40.0,  # FOV กว้าง 40 องศา
        vfov_deg=32.0,  # FOV สูง 30 องศา
        gimbal_yaw_deg=0,  # หันหน้าเหนือ
        gimbal_pitch_deg=-89.9,  # ชี้ตรงลง (active stabilization)
        gimbal_roll_deg=0,  # level
        image_width=640,  # ขนาดภาพ 640x512
        image_height=512,
        pixel_x=320,  # จุดกลางภาพ X
        pixel_y=256,  # จุดกลางภาพ Y
        ground_elevation_m=0,  # ระดับน้ำทะเล
        utm_zone="32647",  # UTM zone สำหรับประเทศไทย
    )
    print(f"\nพิกัดของ pixel (320,256): drone_lat={13.7563}, drone_lon={100.5018}")
    print(f"\nพิกัดของ pixel (320,256): lat={lat:.8f}, lon={lon:.8f}")

    # ทดสอบกับจุดมุมภาพ
    print("\n=== ทดสอบจุดมุมภาพ ===")
    corners = [
        (0, 0, "มุมซ้ายบน"),
        (639, 0, "มุมขวาบน"),
        (0, 511, "มุมซ้ายล่าง"),
        (639, 511, "มุมขวาล่าง"),
    ]

    for px, py, desc in corners:
        lat_corner, lon_corner = pixel_to_latlon_k40t_thermal(
            drone_lat=13.7563,  # พิกัดโดรน (กรุงเทพฯ)
            drone_lon=100.5018,
            drone_altitude_m=90,  # ความสูง 100 เมตร
            drone_heading_deg=0,        # โดรนหันหน้าไปทิศตะวันออกเฉียงเหนือ
            drone_pitch_deg=0,           # โดรนบินเรียบ (ไม่เอียงหน้า-หลัง)
            drone_roll_deg=0,            # โดรนบินเรียบ (ไม่เอียงซ้าย-ขวา)
            hfov_deg=40.0,  # FOV กว้าง 40 องศา
            vfov_deg=32.0,  # FOV สูง 30 องศา
            gimbal_yaw_deg=0,  # หันหน้าเหนือ
            gimbal_pitch_deg=-89.9,  # ชี้ตรงลง (active stabilization)
            gimbal_roll_deg=0,  # level
            image_width=640,  # ขนาดภาพ 640x512
            image_height=512,
            pixel_x=px,  # จุดกลางภาพ X
            pixel_y=py,  # จุดกลางภาพ Y
            ground_elevation_m=0,  # ระดับน้ำทะเล
            utm_zone="32647",  # UTM zone สำหรับประเทศไทย
        )
        print(f"{desc} ({px},{py}): lat={lat_corner:.8f}, lon={lon_corner:.8f}")
        print("---")
