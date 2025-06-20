import cv2
import textwrap

def draw_caption(frame, caption):
    if not caption:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)

    # Wrap caption into lines (around 50 chars per line)
    wrapped_text = textwrap.wrap(caption, width=50)

    # Determine size of background box
    line_height = 25
    box_height = line_height * len(wrapped_text) + 10
    cv2.rectangle(frame, (0, 0), (frame.shape[1], box_height), bg_color, -1)

    # Draw each line centered
    y = 25
    for line in wrapped_text:
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2  # center horizontally
        cv2.putText(frame, line, (text_x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        y += line_height
