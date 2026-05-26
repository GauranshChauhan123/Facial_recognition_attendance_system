import numpy as np
import cv2


# =========================
# COSINE SIMILARITY
# =========================

def cosine_similarity(vec1, vec2):

    dot_product = np.dot(vec1, vec2)

    norm1 = np.linalg.norm(vec1)

    norm2 = np.linalg.norm(vec2)

    similarity = dot_product / (norm1 * norm2)

    return similarity


# =========================
# DRAW FACE BOX
# =========================

def draw_box(frame, face, name, score):

    x, y, w, h = list(map(int, face[:4]))

    color = (0, 255, 0)

    if name == "Unknown":
        color = (0, 0, 255)

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        color,
        2
    )

    cv2.putText(
        frame,
        f"{name} ({score:.2f})",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


# =========================
# MARK ATTENDANCE
# =========================

def mark_attendance(name, attendance_file):

    import pandas as pd
    from datetime import datetime
    import os

    now = datetime.now()

    current_date = now.strftime("%d-%m-%y")

    current_time = now.strftime("%H:%M:%S")

    if not os.path.exists(attendance_file):

        df = pd.DataFrame(columns=["Name", "Date", "Time"])

        df.to_csv(attendance_file, index=False)

    # Read existing attendance
    df = pd.read_csv(attendance_file)

    # Check duplicate for same date
    already_marked = (
        (df["Name"] == name) &
        (df["Date"] == current_date)
    ).any()

    if already_marked:

        print(f"{name} already marked today")

        return

    new_row = pd.DataFrame([{
        "Name": name,
        "Date": current_date,
        "Time": current_time
    }])

    new_row.to_csv(
        attendance_file,
        mode="a",
        header=False,
        index=False
    )

    print(f"Attendance Marked: {name}")