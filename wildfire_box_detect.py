import cv2
import numpy as np
from typing import List, Tuple

def detect_red_rectangles(image_path: str, debug: bool = False) -> List[List[int]]:
    """
    ตรวจจับกรอบสี่เหลี่ยมสีแดงจากรูป และคืนค่า bounding box
    
    Args:
        image_path (str): path ไฟล์รูปภาพ
    
    Returns:
        List[List[int]]: รายการ bbox ในรูปแบบ [Xmax, Ymax, Xmin, Ymin]
    """
    
    # อ่านรูปภาพ
    image = cv2.imread(image_path)
    if image is None:
        print("ไม่สามารถอ่านไฟล์รูปภาพได้")
        return []
    
    original = image.copy()

    # แปลงจาก BGR เป็น HSV เพื่อใช้ในการหาสีแดง
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # กำหนดช่วงสีแดงใน HSV
    # สีแดงอยู่ในช่วง 0-10 และ 160-180 ใน H channel
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # สร้าง mask สำหรับสีแดง
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    # หา contours จาก mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    
    for contour in contours:
        # กรองเฉพาะ contour ที่มีขนาดเหมาะสม
        area = cv2.contourArea(contour)
        if area < 1000:  # กรองพื้นที่เล็กเกินไป
            continue
        
        # ประมาณรูปร่างเป็นสี่เหลี่ยม
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # หา bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # คำนวณ aspect ratio เพื่อตรวจสอบว่าเป็นสี่เหลี่ยมจริง
        aspect_ratio = float(w) / h
        
        # กรองเฉพาะรูปสี่เหลี่ยมที่มี aspect ratio สมเหตุสมผล
        if 0.2 < aspect_ratio < 5.0:
            # สร้าง bbox ในรูปแบบ [Xmax, Ymax, Xmin, Ymin]
            xmax = x + w
            ymax = y + h
            xmin = x
            ymin = y
            
            rectangles.append([xmax, ymax, xmin, ymin])
        if debug:
            # วาดกรอบบนรูปต้นฉบับ
            cv2.rectangle(original, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            # ใส่ข้อความ bbox
            cv2.putText(original, f"[{xmax},{ymax},{xmin},{ymin}]", 
                       (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
    if debug:
        try:
            # แสดงรูปผลลัพธ์
            cv2.imshow('Original with Detection', original)
            cv2.imshow('Red Mask', red_mask)
            cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()

    return rectangles

def detect_red_rectangles_advanced(image_path: str, debug: bool = False) -> List[List[int]]:
    """
    ฟังก์ชันขั้นสูงสำหรับตรวจจับกรอบสี่เหลี่ยมสีแดง
    
    Args:
        image_path (str): path ไฟล์รูปภาพ
        debug (bool): แสดงรูปภาพ debug หรือไม่
    
    Returns:
        List[List[int]]: รายการ bbox ในรูปแบบ [Xmax, Ymax, Xmin, Ymin]
    """
    
    # อ่านรูปภาพ
    image = cv2.imread(image_path)
    if image is None:
        print("ไม่สามารถอ่านไฟล์รูปภาพได้")
        return []
    
    original = image.copy()
    
    # แปลงเป็น HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # กำหนดช่วงสีแดงใน HSV (ปรับให้เข้มงวดขึ้น)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # สร้าง mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    # ทำ morphological operations เพื่อปรับปรุง mask
    kernel = np.ones((2, 2), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # หา contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        print (f'Area: ', area)
        # กรองตาม area และ perimeter
        if area < 750 or perimeter < 200:
            continue
        
        # หา bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # ตรวจสอบ aspect ratio
        aspect_ratio = float(w) / h
        if not (0.3 < aspect_ratio < 3.0):
            continue
        
        # ตรวจสอบว่า contour เป็นสี่เหลี่ยมหรือไม่
        # โดยใช้ area ratio ระหว่าง contour area และ bounding rect area
        rect_area = w * h
        area_ratio = area / rect_area if rect_area > 0 else 0
        
        # ถ้า area ratio น้อยเกินไป แสดงว่าไม่ใช่รูปสี่เหลี่ยม
        if area_ratio < 0.3:
            continue
        
        # สร้าง bbox ในรูปแบบ [Xmax, Ymax, Xmin, Ymin]
        xmax = x + w
        ymax = y + h
        xmin = x
        ymin = y
        
        rectangles.append([xmax, ymax, xmin, ymin])
        
        if debug:
            # วาดกรอบบนรูปต้นฉบับ
            cv2.rectangle(original, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            # ใส่ข้อความ bbox
            cv2.putText(original, f"[{xmax},{ymax},{xmin},{ymin}]", 
                       (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    if debug:
        try:
            # แสดงรูปผลลัพธ์
            cv2.imshow('Original with Detection', original)
            cv2.imshow('Red Mask', red_mask)
            cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()
    
    return rectangles

def process_image_and_print_results(image_path: str):
    """
    ประมวลผลรูปภาพและแสดงผลลัพธ์
    """
    print(f"กำลังประมวลผลรูปภาพ: {image_path}")
    
    # ใช้ฟังก์ชันพื้นฐาน
    basic_results = detect_red_rectangles(image_path, debug=True)
    print(f"\nผลลัพธ์จากฟังก์ชันพื้นฐาน:")
    print(f"พบกรอบสี่เหลี่ยมสีแดง {len(basic_results)} กรอบ")
    for i, bbox in enumerate(basic_results):
        print(f"กรอบที่ {i+1}: [Xmax={bbox[0]}, Ymax={bbox[1]}, Xmin={bbox[2]}, Ymin={bbox[3]}]")
    
    # ใช้ฟังก์ชันขั้นสูง
    advanced_results = detect_red_rectangles_advanced(image_path, debug=True)
    print(f"\nผลลัพธ์จากฟังก์ชันขั้นสูง:")
    print(f"พบกรอบสี่เหลี่ยมสีแดง {len(advanced_results)} กรอบ")
    for i, bbox in enumerate(advanced_results):
        print(f"กรอบที่ {i+1}: [Xmax={bbox[0]}, Ymax={bbox[1]}, Xmin={bbox[2]}, Ymin={bbox[3]}]")
    
    return advanced_results

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # เปลี่ยน path นี้เป็นไฟล์รูปภาพของคุณ
    image_path = "test.jpg"
    
    try:
        results = process_image_and_print_results(image_path)
        
        # แสดงผลลัพธ์ในรูปแบบที่ต้องการ
        print(f"\nผลลัพธ์สุดท้าย:")
        for bbox in results:
            print(f"bbox = {bbox}")
            
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
        print("กรุณาตรวจสอบ path ของไฟล์รูปภาพ และติดตั้ง opencv-python")
        print("ติดตั้งด้วยคำสั่ง: pip install opencv-python")