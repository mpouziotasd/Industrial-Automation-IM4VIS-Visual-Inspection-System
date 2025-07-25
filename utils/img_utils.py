import cv2 as cv
import numpy as np

color_ranges = {
    "red": (0, 0, 255), 
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "white": (255, 255, 255)
}

def draw_text(frame, text, position=(20, 20), background=True, background_color=(255, 255, 255), text_color=(0, 0, 0)):
    x, y = int(position[0]), int(position[1])
    font_scale = 0.7
    thickness = 1
    padding = 4

    (text_w, text_h), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_w, text_h = int(text_w), int(text_h + baseline)

    text_x = x
    text_y = y + text_h // 2 - baseline // 2
    if background:
        bg_top = y - text_h // 2 - padding
        bg_bottom = y + text_h // 2 + padding
        bg_left = x - padding
        bg_right = x + text_w + padding
        cv.rectangle(frame,
                     (bg_left, bg_top),
                     (bg_right, bg_bottom),
                     background_color, -1)

    cv.putText(
        frame,
        text,
        (text_x, text_y),
        cv.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=text_color,
        thickness=thickness,
        lineType=cv.LINE_AA
    )

def is_in_boundaries(obj_point, region_points):
    obj_point = np.array(obj_point, dtype=np.float32)
    region_points = np.array(region_points, dtype=np.float32)
    if region_points.ndim == 2:
        region_points = region_points.reshape((-1, 1, 2))
    return cv.pointPolygonTest(region_points, tuple(obj_point), False) >= 0 

def draw_data(frame, bboxes, clss, region_points):
    for bbox, _cls in zip(bboxes, clss):
        x1, y1, x2, y2 = map(int, bbox)
        col = color_ranges['red'] if is_in_boundaries(((x1+x2)//2, (y1+y2)//2), region_points) else color_ranges['blue']
        cv.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        
    return frame

def draw_polygons(frame, pts):
    return cv.polylines(frame, [pts], True, (255, 0, 0), 1)

def get_mask_area(mask, shape):
    binary_mask = np.zeros(shape[:2], dtype=np.uint8)
    cv.fillPoly(binary_mask, [mask], 1)
    return int(np.sum(binary_mask))

def extract_mask(image, mask):
    mask_img = np.zeros(image.shape[:2], dtype=np.uint8)
    cv.fillPoly(mask_img, [mask], 255)
    masked = cv.bitwise_and(image, image, mask=mask_img)

    x, y, w, h = cv.boundingRect(mask)
    cropped = masked[y:y+h, x:x+w]
    return cropped

def calculate_roughness(mask):
    edges = cv.Canny(mask, threshold1=10, threshold2=30)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0

    cnt = max(contours, key=cv.contourArea)
    peri = cv.arcLength(cnt, True)
    hull = cv.convexHull(cnt)
    hull_peri = cv.arcLength(hull, True)
    if hull_peri == 0:
        return 0.0
    return peri / hull_peri


def draw_counting_region(frame):
    """
        Description:
            Defines an interactable UI on an image to draw a rectangle region of interest using points for the counting model.
            Click on one point to another and form a rectangle. Minimum of 3 points are required.
        Interactions:
            i) 'Mouse Click': Click to a certain location to define a point for the rectangle.
            ii) 'S': Press 'S' to submit the region of interest onto the tracking model.
            iii) 'Q': Press 'Q' to exit the window and the program.
    """

    working_frame = frame.copy()
    original_frame = frame.copy()
    points = []
    submitted = False
    window_name = "Draw Counting Region"
    box = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal working_frame, points, submitted

        if submitted:
            return

        if event == cv.EVENT_LBUTTONDOWN:
            points.append((x, y))

        elif event == cv.EVENT_RBUTTONDOWN and points:
            points.pop()

        working_frame = original_frame.copy()
        for p in points:
            cv.circle(working_frame, p, 5, (0, 255, 0), -1)

        if len(points) >= 3:
            rect = cv.minAreaRect(np.array(points))
            box_local = cv.boxPoints(rect)
            box_local = np.intp(box_local)
            cv.drawContours(working_frame, [box_local], 0, (0, 0, 255), 2)

        cv.imshow(window_name, working_frame)

    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, mouse_callback)
    cv.imshow(window_name, working_frame)

    while True:
        key = cv.waitKey(1) & 0xFF
        if key == ord('s') and len(points) >= 3:
            rect = cv.minAreaRect(np.array(points))
            box = cv.boxPoints(rect)
            box = np.intp(box)
            submitted = True
            break
        elif key == ord('q') or key == 27:
            break

    cv.destroyWindow(window_name)
    cv.waitKey(1)

    if submitted:
        result_frame = original_frame.copy()
        cv.drawContours(result_frame, [box], 0, (0, 255, 0), 2)
        for p in points:
            cv.circle(result_frame, p, 5, (0, 255, 0), -1)
        return box, result_frame
    else:
        return None, original_frame

    