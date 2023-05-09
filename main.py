import cv2

# Load the video
cap = cv2.VideoCapture('Fire.mp4')

# Define the color map for heat detection
color_map = cv2.COLORMAP_HOT

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply the color map for heat detection
    color_heat = cv2.applyColorMap(gray, color_map)

    # Threshold the grayscale image to detect fire
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours of the fire
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a purple square around the fire
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(color_heat, (x, y), (x + w, y + h), (255, 0, 255), 2)

    # Display the image twice (once in black and white, once in color)
    cv2.imshow('Black and White', gray)
    cv2.imshow('Color Heat Map', color_heat)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()