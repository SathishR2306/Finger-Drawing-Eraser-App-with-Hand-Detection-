import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

colors = [
    (255, 0, 255),  # Purple
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (0, 255, 255),  # Yellow
    (255, 255, 255) # White
]

color_names = ['Purple', 'Blue', 'Green', 'Red', 'Yellow', 'White']
eraser_name = 'Eraser'
eraser_color = (0, 0, 0)
button_size = (80, 80)
button_positions = [(10 + i * 90, 10) for i in range(len(colors) + 1)]

draw_color = colors[0]
brush_thickness = 10
eraser_thickness = 50
eraser_selected = False
canvas = None
prev_x, prev_y = 0, 0

def fingers_up(lm_list):
    fingers = []
    fingers.append(1 if lm_list[8][1] < lm_list[6][1] else 0)   
    fingers.append(1 if lm_list[12][1] < lm_list[10][1] else 0)
    fingers.append(1 if lm_list[16][1] < lm_list[14][1] else 0)
    fingers.append(1 if lm_list[20][1] < lm_list[18][1] else 0)
    return fingers

def draw_navbar(img):
    for i, pos in enumerate(button_positions):
        x, y = pos
        if i < len(colors):
            color = colors[i]
            name = color_names[i]
        else:
            color = (50, 50, 50)
            name = eraser_name

        cv2.rectangle(img, (x, y), (x + button_size[0], y + button_size[1]), color, -1)

        if (not eraser_selected and i < len(colors) and colors[i] == draw_color) or (eraser_selected and i == len(colors)):
            cv2.rectangle(img, (x, y), (x + button_size[0], y + button_size[1]), (255, 255, 255), 5)

        if color == (255, 255, 255) or i == len(colors):
            text_color = (0, 0, 0)
        else:
            text_color = (255, 255, 255)

        cv2.putText(img, name, (x + 5, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    cv2.putText(img, "Raise 2 fingers to select color/tool", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def check_navbar_selection(x, y):
    global draw_color, eraser_selected
    for i, pos in enumerate(button_positions):
        x1, y1 = pos
        if x1 < x < x1 + button_size[0] and y1 < y < y1 + button_size[1]:
            if i < len(colors):
                draw_color = colors[i]
                eraser_selected = False
            else:
                eraser_selected = True

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm_list = []

        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append((cx, cy))

        fingers = fingers_up(lm_list)
        index_finger_tip = lm_list[8]

        if fingers[0] == 1 and fingers[1] == 1:
            check_navbar_selection(*index_finger_tip)
            cv2.circle(frame, index_finger_tip, 20, (0, 255, 255), cv2.FILLED)
            prev_x, prev_y = 0, 0

        elif fingers[0] == 1 and fingers[1] == 0:
            if eraser_selected:
                cv2.circle(frame, index_finger_tip, eraser_thickness//2, (0, 0, 255), 2)
                cv2.circle(canvas, index_finger_tip, eraser_thickness//2, (0, 0, 0), -1)
                prev_x, prev_y = 0, 0
            else:
                cv2.circle(frame, index_finger_tip, 15, draw_color, cv2.FILLED)
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = index_finger_tip
                else:
                    cv2.line(canvas, (prev_x, prev_y), index_finger_tip, draw_color, brush_thickness)
                    prev_x, prev_y = index_finger_tip
        else:
            prev_x, prev_y = 0, 0

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        prev_x, prev_y = 0, 0

    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv_canvas = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY_INV)
    inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv_canvas)
    frame = cv2.bitwise_or(frame, canvas)

    draw_navbar(frame)

    cv2.imshow("Finger Drawing and Erasing", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()
