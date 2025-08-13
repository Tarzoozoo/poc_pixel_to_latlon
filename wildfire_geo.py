from wildfire_cal import img2geo
from wildfire_box_detect import detect_red_rectangles, detect_red_rectangles_advanced
import json

with open("config.json", "r") as f:
    config = json.load(f)

# Accessing camera parameters
IMAGE_WIDTH = config["image"]["width_px"]
IMAGE_HEIGHT = config["image"]["height_px"]
FOV_WIDTH = config["image"]["horizontal_fov_deg"]
FOV_HEIGHT = config["image"]["vertical_fov_deg"]
gps = config["telemetry"]["center_gps"]
CAM_LAT = gps["lat_deg"]
CAM_LNG = gps["lon_deg"]
ALTITUDE = gps["alt_m"]

# Accessing drone pose
drone_attitude = config["telemetry"]["drone_attitude_deg"]
DRONE_YAW = drone_attitude["yaw"]
DRONE_PITCH = drone_attitude["pitch"]
# Accessing camera pose
cam_attitude = config["telemetry"]["camera_mount_deg"]
CAM_YAW = cam_attitude["yaw"]
CAM_PITCH = cam_attitude["pitch"]
# IMAGE_WIDTH  = 1280
# IMAGE_HEIGHT = 1024
# FOV_WIDTH    = 22.9
# FOV_HEIGHT   = 18.4
# ALTITUDE     = 1000
# TILT_ANGLE   = 0
# HEADING      = 0
# CAM_LAT      = 13.756300
# CAM_LNG      = 100.501800

TOTAL_PITCH = DRONE_PITCH + CAM_PITCH
TOTAL_HEADING = DRONE_YAW + CAM_YAW
imggeo = img2geo(
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    FOV_WIDTH,
    FOV_HEIGHT,
    ALTITUDE,
    TOTAL_PITCH,
    TOTAL_HEADING,
    CAM_LAT,
    CAM_LNG
)
print(f'Input lat: {CAM_LAT} Input lng: {CAM_LNG}')
# ตรวจจับสี่เหลี่ยมสีแดง
rectangles = detect_red_rectangles("test.png")
for bbox in rectangles:
    print(f"Rectangle: [Xmax={bbox[0]}, Ymax={bbox[1]}, Xmin={bbox[2]}, Ymin={bbox[3]}]")
    info = imggeo.get_bbox_info(bbox)
    print(f'Output lat: {info.lat} Output lng: {info.lng}')

# bbox = [700, 572, 580, 452]  # [Xmax, Ymax, Xmin, Ymin]
# TL_bbox = [515, 422, 365, 302]
# TR_bbox = [915, 422, 765, 302]
# BL_bbox = [515, 722, 365, 602]
# BR_bbox = [915, 722, 765, 602]

# test_bbox = [856, 1209, 129, 795]
# info = imggeo.get_bbox_info(test_bbox)
# print(f'Input lat: {13.756300} Input lng: {100.501800}')
# print(f'Output lat: {info.lat} Output lng: {info.lng}')
