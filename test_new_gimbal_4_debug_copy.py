import numpy as np
import math
from pyproj import Transformer

def create_rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                   [math.sin(yaw),  math.cos(yaw), 0],
                   [0, 0, 1]])
    Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                   [0, 1, 0],
                   [-math.sin(pitch), 0, math.cos(pitch)]])
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(roll), -math.sin(roll)],
                   [0, math.sin(roll), math.cos(roll)]])
    return Rz @ Ry @ Rx

def pixel_to_latlon_utm(
    pixel_x, pixel_y,
    drone_lat, drone_lon, drone_alt,
    drone_yaw=0, drone_pitch=0, drone_roll=0,
    gimbal_yaw=0, gimbal_pitch=-90, gimbal_roll=0,
    image_width=640, image_height=512,
    fov_horizontal=40, fov_vertical=32,
    ground_elevation=0,
    utm_zone="32647",
    debug=False
):
    # 1. Calculate center and angle per pixel
    center_x = image_width / 2
    center_y = image_height / 2
    pixel_size_x = fov_horizontal / image_width   # deg/pixel
    pixel_size_y = fov_vertical / image_height    # deg/pixel

    x_angle = (pixel_x - center_x) * pixel_size_x
    y_angle = (pixel_y - center_y) * pixel_size_y

    x_rad = math.radians(x_angle)
    y_rad = math.radians(y_angle)

    # 2. Create ray in camera frame
    ray_camera = np.array([
        math.sin(x_rad),         # X: right
        math.sin(y_rad),         # Y: down
        math.cos(x_rad) * math.cos(y_rad)  # Z: forward
    ])
    ray_camera = ray_camera / np.linalg.norm(ray_camera)

    # 3. Map to body frame (K40T: Z-forward camera to X-forward drone, X-right camera to Y-right drone, Y-down camera to Z-down drone)
    ray_body_cam = np.array([ray_camera[2], ray_camera[0], ray_camera[1]])

    # 4. Rotate by gimbal
    R_gimbal = create_rotation_matrix(gimbal_yaw, gimbal_pitch, gimbal_roll)
    ray_body = R_gimbal @ ray_body_cam

    # 5. Rotate by drone
    R_drone = create_rotation_matrix(drone_yaw, drone_pitch, drone_roll)
    ray_world = R_drone @ ray_body

    # 6. Drone position in UTM
    transformer_to_utm = Transformer.from_crs("epsg:4326", f"epsg:{utm_zone}")
    drone_x_utm, drone_y_utm = transformer_to_utm.transform(drone_lat, drone_lon)
    drone_pos_utm = np.array([drone_x_utm, drone_y_utm, drone_alt])

    # 7. Find intersection with ground
    if abs(ray_world[2]) < 1e-10:
        raise ValueError("Ray is parallel to ground plane - cannot find intersection")
    t = (ground_elevation - drone_alt) / ray_world[2]
    target_utm = drone_pos_utm + t * ray_world

    # 8. Back to lat/lon
    transformer_to_wgs84 = Transformer.from_crs(f"epsg:{utm_zone}", "epsg:4326")
    target_lat, target_lon = transformer_to_wgs84.transform(target_utm[0], target_utm[1])

    if debug:
        print("=== Debug UTM Ray ===")
        print(f"Drone UTM: ({drone_x_utm:.3f}, {drone_y_utm:.3f}, {drone_alt:.3f})")
        print(f"Target UTM: ({target_utm[0]:.3f}, {target_utm[1]:.3f}, {ground_elevation:.3f})")
        print(f"Ray world: [{ray_world[0]:.4f}, {ray_world[1]:.4f}, {ray_world[2]:.4f}]")
        print(f"Horizontal dist: {np.linalg.norm(target_utm[:2] - drone_pos_utm[:2]):.2f} m")
        print(f"Pixel ({pixel_x},{pixel_y}) → lat/lon: ({target_lat:.8f},{target_lon:.8f})")
        print("====================")

    return target_lat, target_lon

# ========== ตัวอย่างใช้งาน ==========

if __name__ == "__main__":
    drone_lat = 13.7563
    drone_lon = 100.5018
    drone_alt = 100
    drone_yaw = 0
    drone_pitch = 0
    drone_roll = 0
    gimbal_yaw = 0
    gimbal_pitch = -90
    gimbal_roll = 0

    print("\n=== Center ===")
    lat, lon = pixel_to_latlon_utm(
        320, 256,
        drone_lat, drone_lon, drone_alt,
        drone_yaw, drone_pitch, drone_roll,
        gimbal_yaw, gimbal_pitch, gimbal_roll,
        debug=True
    )
    print(f"Center: lat={lat:.8f}, lon={lon:.8f}")

    print("\n=== Corner ===")
    for px, py, name in [
        (0, 0, "Top-Left"),
        (639, 0, "Top-Right"),
        (0, 511, "Bottom-Left"),
        (639, 511, "Bottom-Right")
    ]:
        lat, lon = pixel_to_latlon_utm(
            px, py,
            drone_lat, drone_lon, drone_alt,
            drone_yaw, drone_pitch, drone_roll,
            gimbal_yaw, gimbal_pitch, gimbal_roll,
            debug=True
        )
        print(f"{name}: lat={lat:.8f}, lon={lon:.8f}")

