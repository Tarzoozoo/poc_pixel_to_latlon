import cv2
def generate_box_ouput(image, center_x: float, center_y: float,
              xmax: float, xmin: float, ymax: float, ymin: float,
              info, IMAGE_WIDTH, box_number, font_scale: float = 0.6, thickness: int = 2):
    
    # สีและการตั้งค่า
    box_color = (0, 255, 0)  # เขียว สำหรับกรอบ
    text_color = (255, 255, 255)  # ขาว สำหรับข้อความ
    bg_color = (0, 0, 0)  # ดำ สำหรับพื้นหลังข้อความ
    
    # วาดกรอบสี่เหลี่ยม
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), box_color, thickness)
    
    # วาดจุดกลาง
    cv2.circle(image, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)  # จุดสีน้ำเงิน
    cv2.circle(image, (int(center_x), int(center_y)), 8, (255, 255, 255), 2)  # วงกลมขาวล้อมรอบ
    
    # เตรียมข้อความ
    texts = [
        f"Box {box_number+1}",
        f"Pixel: ({center_x:.0f}, {center_y:.0f})",
        f"Lat: {info.lat:.6f}",
        f"Lng: {info.lng:.6f}"
    ]
    
    # คำนวณตำแหน่งข้อความ
    text_start_x = max(5, xmin)  # ไม่ให้ข้อความออกนอกขอบซ้าย
    text_start_y = max(25, ymin - 10)  # วางข้อความเหนือกรอบ
    
    # ถ้าข้อความจะออกนอกขอบบน ให้วางข้างในกรอบแทน
    if text_start_y < 80:  # มีข้อความ 4 บรรทัด ต้องการพื้นที่อย่างน้อย 80 pixels
        text_start_y = ymin + 25  # วางข้างในกรอบ
    
    # วาดพื้นหลังสำหรับข้อความ
    text_height = 20
    bg_height = len(texts) * text_height + 10
    bg_width = 250
    
    # ตรวจสอบไม่ให้พื้นหลังออกนอกขอบรูป
    bg_x1 = min(text_start_x - 5, IMAGE_WIDTH - bg_width)
    bg_y1 = text_start_y - text_height - 5
    bg_x2 = bg_x1 + bg_width
    bg_y2 = bg_y1 + bg_height
    
    # วาดสี่เหลี่ยมพื้นหลังแบบโปร่งแสง
    overlay = image.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # วาดกรอบรอบพื้นหลัง
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), box_color, 1)
    
    # เขียนข้อความ
    for j, text in enumerate(texts):
        text_x = bg_x1 + 10
        text_y = bg_y1 + (j + 1) * text_height
        
        cv2.putText(image, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)
    image_height, image_width = image.shape[:2]
    center_img_x = image_width // 2
    center_img_y = image_height // 2
    line_color = (200, 200, 200)  # เทาอ่อน
    line_thickness = 1

    cv2.line(image, (center_img_x, 0), (center_img_x, image_height), line_color, line_thickness)
    cv2.line(image, (0, center_img_y), (image_width, center_img_y), line_color, line_thickness)
    return image