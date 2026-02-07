import os
os.environ["YOLO_VERBOSE"] = "False"

from ultralytics import YOLO
import cv2
import easyocr
import re
import csv
from datetime import datetime

# Load model
model = YOLO("yolov8n-license-plate.pt", verbose=False)
reader = easyocr.Reader(['en'], gpu=False)

# Regex for Indian number plate
plate_pattern = re.compile(r'\b([A-Z]{2}\d{2}[A-Z]{1,2}\d{4})\b')

# Open video
cap = cv2.VideoCapture("videos/test1.mp4")

if not cap.isOpened():
    print("❌ Video not found")
    exit()

# Dictionary to avoid duplicates
# {plate_number: (last_4_digits, first_seen_time)}
detected_plates = {}

# Create CSV file and write header
csv_file = "number_plates.csv"
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Plate Number", "Last 4 Digits", "First Seen Time"])

print("✅ Detecting plates and saving to CSV...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb, conf=0.25)[0]

    if results.boxes is not None:
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)

            plate_img = frame[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue

            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            texts = reader.readtext(gray, detail=0)

            for t in texts:
                t = t.replace(" ", "").upper()
                match = plate_pattern.search(t)

                if match:
                    full_plate = match.group(1)
                    last_digits = full_plate[-4:]

                    # Save only first occurrence
                    if full_plate not in detected_plates:
                        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        detected_plates[full_plate] = (last_digits, time_now)

                        # Append to CSV
                        with open(csv_file, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([full_plate, last_digits, time_now])

                        print(f"📌 Saved: {full_plate}")

                    # Draw bounding box + text
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 0, 255), 2)
                    cv2.putText(frame, full_plate, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)

    cv2.putText(frame,
                f"Total Plates: {len(detected_plates)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0), 2)

    cv2.imshow("ANPR Output", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

print("\n📋 FINAL CSV SUMMARY:")
for plate, data in detected_plates.items():
    print(f"{plate} → {data[0]}")

print(f"\n✅ TOTAL UNIQUE PLATES SAVED: {len(detected_plates)}")
print(f"📁 CSV FILE CREATED: {csv_file}")