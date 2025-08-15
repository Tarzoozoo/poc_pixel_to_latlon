import json
import cv2
from wildfire_cal import img2geo
from wildfire_box_detect import detect_red_rectangles, detect_red_rectangles_advanced
from wildfire_box_output import generate_box_ouput

def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def get_total_attitude(config):
    drone = config["telemetry"]["drone_attitude_deg"]
    cam = config["telemetry"]["camera_mount_deg"]
    total_pitch = drone["pitch"] + cam["pitch"]
    total_heading = drone["yaw"] + cam["yaw"]
    return total_pitch, total_heading

def get_image_dimensions(config):
    image = config["image"]
    return image["width_px"], image["height_px"], image["horizontal_fov_deg"], image["vertical_fov_deg"]

def get_camera_gps(config):
    gps = config["telemetry"]["center_gps"]
    return gps["lat_deg"], gps["lon_deg"], gps["alt_m"]

def draw_bbox_with_coordinates(image_path: str, config_path: str = "config.json", 
                              output_path: str = "output_with_coordinates.jpg"):
    
    config = load_config(config_path)
    IMAGE_WIDTH, IMAGE_HEIGHT, FOV_WIDTH, FOV_HEIGHT = get_image_dimensions(config)
    CAM_LAT, CAM_LNG, ALTITUDE = get_camera_gps(config)
    TOTAL_PITCH, TOTAL_HEADING = get_total_attitude(config)

    imggeo = img2geo(
        IMAGE_WIDTH, IMAGE_HEIGHT, FOV_WIDTH, FOV_HEIGHT,
        ALTITUDE, TOTAL_PITCH, TOTAL_HEADING, CAM_LAT, CAM_LNG
    )

    image = cv2.imread(image_path)
    if image is None:
        print(f"ไม่สามารถอ่านไฟล์รูปภาพ: {image_path}")
        return

    # Box detection
    print(f'Camera input: Lat={CAM_LAT:.6f}, Lng={CAM_LNG:.6f}')
    print(f'Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}')
    print(f'Center Image: {IMAGE_WIDTH/2}x{IMAGE_HEIGHT/2}')
    print("Detecting red rectangles...")

    rectangles = detect_red_rectangles(image_path)
    print(f"Found {len(rectangles)} red rectangles")
    if not rectangles:
        print("ไม่พบกรอบสี่เหลี่ยมสีแดง")
        return
    
    # Convert pixel to lat/lon
    for i, bbox in enumerate(rectangles):
        xmax, ymax, xmin, ymin = bbox
        
        center_x = (xmax + xmin) / 2
        center_y = (ymax + ymin) / 2
        print(f"\n=== Box {i+1} ===")
        print(f"BBox: [Xmax={xmax}, Ymax={ymax}, Xmin={xmin}, Ymin={ymin}]")
        print(f"Center pixel: ({center_x:.1f}, {center_y:.1f})")

        info = imggeo.get_bbox_info(bbox)
        print(f"Output: Lat={info.lat:.8f}, Lng={info.lng:.8f}")

        # Draw box output on image
        image = generate_box_ouput(image, center_x, center_y, 
                                        xmax, xmin, ymax, ymin, 
                                        info, IMAGE_WIDTH, i)
    
    success = cv2.imwrite(output_path, image)
    if success:
        print(f"\n✅ บันทึกรูปผลลัพธ์แล้ว: {output_path}")
    else:
        print(f"\n❌ ไม่สามารถบันทึกรูปได้: {output_path}")

if __name__ == "__main__":
    print("🔍 เริ่มต้นการวิเคราะห์รูปภาพ...")
    try:
        draw_bbox_with_coordinates(
            image_path="test.jpg",
            config_path="config.json",
            output_path="result_with_coords.jpg"
        )
    except Exception as e:
        print(f"Error with config method: {e}")