import cv2
import numpy as np

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges
def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # Define a polygon that roughly represents the area where the lanes are
    polygon = np.array([[
        (0, height * 3 / 5),
        (width, height * 3 / 5),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges
def detect_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)
    return lines
def draw_lines(frame, lines):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return combined_image
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        edges = preprocess_frame(frame)
        cropped_edges = region_of_interest(edges)
        lines = detect_lines(cropped_edges)
        result_frame = draw_lines(frame, lines)

        cv2.imshow('Lane Detection', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
process_video('road_video.mp4')
