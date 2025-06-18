import cv2
import numpy as np
import os
import sys
import argparse

import inference     

# Global variables
points = []
img = None
img_copy = None
dragging = False
drag_point_index = -1
last_selected_point = -1  # Index of the last selected point
point_radius = 7  # Reduced radius for detecting clicks on points
point_size = 3  # Reduced size for visual circle
line_thickness = 1  # Reduced line thickness
warped = None  # To store the perspective-corrected image
point_move_step = 1  # Pixels to move when using arrow keys

# Grid related variables
grid_squares = []  # To store grid square coordinates (top-left, bottom-right)
selected_squares = []  # To store indices of selected squares
square_size = 0  # Size of each grid square in pixels
square_dst = None  # To store the main square destination points
grid_predictions = {}  # Dictionary to store predictions for grid squares {index: (class_name, confidence)}

# Mouse callback function to collect and manipulate points
def click_event(event, x, y, flags, param):
    global points, img, img_copy, dragging, drag_point_index, last_selected_point, warped
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if we're clicking on an existing point
        for i, point in enumerate(points):
            if np.sqrt((x - point[0])**2 + (y - point[1])**2) < point_radius:
                dragging = True
                drag_point_index = i
                last_selected_point = i  # Update last selected point
                update_display()
                return
        
        # If we already have 4 points and aren't clicking on one, do nothing
        if len(points) >= 4:
            return
            
        # Add a new point
        points.append((x, y))
        last_selected_point = len(points) - 1  # Set as last selected
        update_display()
        
        # If we have 4 points, apply perspective transform
        if len(points) == 4:
            apply_perspective_transform()
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging and drag_point_index != -1:
            # Update the point position
            points[drag_point_index] = (x, y)
            update_display()
            
            # If we have 4 points, update the perspective transform in real-time
            if len(points) == 4:
                apply_perspective_transform()
    
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        # Keep drag_point_index as last_selected_point
        if drag_point_index != -1:
            last_selected_point = drag_point_index
        drag_point_index = -1

# Update the display with current points and lines
def update_display():
    global img, img_copy, points, last_selected_point
    
    # Create a fresh copy of the image
    img_copy = img.copy()
    
    # Draw lines between points
    for i in range(1, len(points)):
        cv2.line(img_copy, points[i-1], points[i], (0, 255, 0), line_thickness)
    
    # If we have 4 points, draw the last line to close the quadrilateral
    if len(points) == 4:
        cv2.line(img_copy, points[0], points[3], (0, 255, 0), line_thickness)
    
    # Draw all points
    for i, point in enumerate(points):
        # Use different color for the last selected point
        if i == last_selected_point:
            # Highlight the last selected point in red - offset from the actual corner
            offset_point = (point[0] - 5, point[1] - 5)  # Offset the circle from the actual point
            cv2.circle(img_copy, offset_point, point_size + 1, (0, 0, 255), -1)
            # Add a small indicator showing it's selected
            cv2.putText(img_copy, "*", (point[0]-10, point[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            # Regular points in green - offset from the actual corner
            offset_point = (point[0] - 5, point[1] - 5)  # Offset the circle from the actual point
            cv2.circle(img_copy, offset_point, point_size, (0, 255, 0), -1)
        
        # Add point numbers - also offset
        cv2.putText(img_copy, str(i+1), (point[0]+7, point[1]+7), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Display the image
    cv2.imshow("Image", img_copy)

def click_warped_image(event, x, y, flags, param):
    global warped, grid_squares, selected_squares, square_size, grid_predictions
    
    if event == cv2.EVENT_LBUTTONDOWN and warped is not None:
        # Find which grid square was clicked
        found = False
        for i, square in enumerate(grid_squares):
            top_left, bottom_right = square
            # Check if click is inside this grid square
            if (top_left[0] <= x <= bottom_right[0] and 
                top_left[1] <= y <= bottom_right[1]):
                
                # Toggle selection of this square
                if i in selected_squares:
                    selected_squares.remove(i)
                else:
                    selected_squares.append(i)
                
                # Extract this grid square for classification
                try:
                    # Create a temp directory if it doesn't exist
                    if not os.path.exists("temp"):
                        os.makedirs("temp")
                    
                    # Extract the square from the warped image
                    square_img = warped[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                    
                    # Save the square as a temp image
                    temp_path = os.path.join("temp", f"temp_square_{i}.jpg")
                    cv2.imwrite(temp_path, square_img)
                    
                    # Use the inference module to classify the image
                    try:
                        # Check if the temp file exists and report its size
                        if os.path.exists(temp_path):
                            file_size = os.path.getsize(temp_path)
                            print(f"Temp image saved at {temp_path} (size: {file_size} bytes)")
                        else:
                            print(f"Error: Temp file {temp_path} was not created")
                            continue
                            
                        # Check if the model is loaded in the inference module
                        if not hasattr(inference, 'model') or inference.model is None:
                            print("Error: Model not loaded in inference module")
                            continue
                            
                        print(f"Attempting to classify image at {temp_path}")
                        predicted_class, class_name, confidence = inference.predict_image(temp_path)
                        print(f"Grid square {i} classified as '{class_name}' with confidence {confidence:.2f}")
                        
                        # Print all prediction probabilities for debugging
                        try:
                            img = inference.Image.open(temp_path).convert('RGB')
                            img = img.resize(inference.IMAGE_SIZE)
                            img_array = inference.tf.keras.utils.img_to_array(img)
                            img_array = inference.tf.expand_dims(img_array, 0)
                            img_array = img_array / 255.0
                            predictions = inference.model.predict(img_array, verbose=0)
                            
                            print("Full probabilities:")
                            for j, prob in enumerate(predictions[0]):
                                print(f"  Class {inference.CLASS_NAMES[j]}: {prob:.4f}")
                        except Exception as e:
                            print(f"Error getting full probabilities: {e}")
                        
                        # Store the prediction for this grid square
                        if 'grid_predictions' not in globals():
                            global grid_predictions
                            grid_predictions = {}
                        
                        grid_predictions[i] = (class_name, confidence)
                    except Exception as e:
                        print(f"Error during inference: {e}")
                        import traceback
                        traceback.print_exc()
                except Exception as e:
                    print(f"Error processing grid square for inference: {e}")
                
                found = True
                # Draw the grid with updated selections and predictions
                draw_grid_with_selections()
                break
        
        if not found and len(grid_squares) > 0:
            print(f"No grid square found at coordinates ({x}, {y})")

def draw_grid_with_selections():
    global warped, grid_squares, selected_squares, grid_predictions
    if warped is None:
        return
        
    # Create a copy of the warped image to draw on
    grid_img = warped.copy()
    
    # Draw all grid squares with predictions if available
    for i, square in enumerate(grid_squares):
        top_left, bottom_right = square
        
        # For selected squares, add blue highlight
        if i in selected_squares:
            # Draw a semi-transparent blue rectangle over selected squares
            overlay = grid_img.copy()
            cv2.rectangle(overlay, top_left, bottom_right, (255, 0, 0), -1)  # Blue fill
            alpha = 0.3  # Transparency factor
            cv2.addWeighted(overlay, alpha, grid_img, 1 - alpha, 0, grid_img)
            # Draw a border around the selected square
            cv2.rectangle(grid_img, top_left, bottom_right, (255, 0, 0), 2)  # Blue border
        
        # If we have a prediction for this square, display it
        if i in grid_predictions:
            class_name, confidence = grid_predictions[i]
            
            # Calculate text position (center of the square)
            center_x = (top_left[0] + bottom_right[0]) // 2
            center_y = (top_left[1] + bottom_right[1]) // 2
            
            # Prepare text with class and confidence
            text = f"{class_name}: {confidence:.2f}"
            
            # Get text size to center it properly
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2
            
            # Draw a background rectangle for better visibility
            cv2.rectangle(grid_img, 
                          (text_x - 5, text_y - text_size[1] - 5),
                          (text_x + text_size[0] + 5, text_y + 5),
                          (0, 0, 0), -1)
            
            # Draw the text
            cv2.putText(grid_img, text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display the updated image
    cv2.imshow("Perspective Corrected", grid_img)

def apply_perspective_transform():
    global points, img, warped, square_size, grid_squares, selected_squares, square_dst
    
    # Clear previous selections when transforming
    selected_squares = []
    grid_squares = []
    
    # Get image dimensions
    h, w = img.shape[:2]
    
    # Add padding to make space for grid lines
    padding = 50  # 50 pixels padding on each side
    padded_w = w + 2 * padding
    padded_h = h + 2 * padding
    
    # Convert points to numpy array
    src_points = np.float32(points)
    
    # Calculate the center of the quadrilateral formed by the selected points
    center_x = sum(point[0] for point in points) / 4
    center_y = sum(point[1] for point in points) / 4
    
    # Calculate the average side length of the quadrilateral to determine square size
    side_lengths = [
        np.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2),  # Top side
        np.sqrt((points[2][0] - points[1][0])**2 + (points[2][1] - points[1][1])**2),  # Right side
        np.sqrt((points[3][0] - points[2][0])**2 + (points[3][1] - points[2][1])**2),  # Bottom side
        np.sqrt((points[0][0] - points[3][0])**2 + (points[0][1] - points[3][1])**2)   # Left side
    ]
    avg_side = sum(side_lengths) / 4
    
    # Define the destination square that's centered at the same relative position but with padding
    # The selected area will become a perfect square in the output
    half_side = avg_side / 2
    # Add padding to the center coordinates
    center_x_padded = center_x + padding
    center_y_padded = center_y + padding
    
    square_dst = np.float32([
        [center_x_padded - half_side, center_y_padded - half_side],  # Top-left
        [center_x_padded + half_side, center_y_padded - half_side],  # Top-right
        [center_x_padded + half_side, center_y_padded + half_side],  # Bottom-right
        [center_x_padded - half_side, center_y_padded + half_side]   # Bottom-left
    ])
    
    # Compute the perspective transform matrix from selected points to square
    H = cv2.getPerspectiveTransform(src_points, square_dst)
    
    # Apply the perspective transformation to the whole image with padding
    warped = cv2.warpPerspective(img, H, (padded_w, padded_h))
    
    # Convert square_dst to integer points
    square_points = square_dst.astype(np.int32)
    
    # Calculate the square size for the grid
    square_size = int(avg_side)
    
    # Generate horizontal grid lines and track grid squares
    horizontal_lines = []
    # From bottom of primary square downward
    for y in range(square_points[2][1], h, square_size):
        horizontal_lines.append(y)
        cv2.line(warped, (0, y), (w, y), (0, 255, 0), 1)
    
    # From top of primary square upward
    for y in range(square_points[0][1], 0, -square_size):
        horizontal_lines.append(y)
        cv2.line(warped, (0, y), (w, y), (0, 255, 0), 1)
    
    # Generate vertical grid lines
    vertical_lines = []
    # From left of primary square leftward
    for x in range(square_points[0][0], 0, -square_size):
        vertical_lines.append(x)
        cv2.line(warped, (x, 0), (x, h), (0, 255, 0), 1)
    
    # From right of primary square rightward
    for x in range(square_points[1][0], w, square_size):
        vertical_lines.append(x)
        cv2.line(warped, (x, 0), (x, h), (0, 255, 0), 1)
    
    # Sort the lines
    horizontal_lines.sort()
    vertical_lines.sort()
    
    # Create grid squares based on intersections
    for i in range(len(vertical_lines) - 1):
        for j in range(len(horizontal_lines) - 1):
            top_left = (vertical_lines[i], horizontal_lines[j])
            bottom_right = (vertical_lines[i+1], horizontal_lines[j+1])
            
            # Only add the square if it's within the image boundaries
            if (0 <= top_left[0] < w and 0 <= top_left[1] < h and 
                0 <= bottom_right[0] < w and 0 <= bottom_right[1] < h):
                grid_squares.append((top_left, bottom_right))
    
    # Draw the main square by connecting the points
    square_color = (0, 0, 255)  # Red color for the square
    thickness = 2
    cv2.line(warped, tuple(square_points[0]), tuple(square_points[1]), square_color, thickness)
    cv2.line(warped, tuple(square_points[1]), tuple(square_points[2]), square_color, thickness)
    cv2.line(warped, tuple(square_points[2]), tuple(square_points[3]), square_color, thickness)
    cv2.line(warped, tuple(square_points[3]), tuple(square_points[0]), square_color, thickness)
    
    # Display the warped image with the grid overlay
    cv2.imshow("Perspective Corrected", warped)
    
    # Set mouse callback for the warped image window
    cv2.setMouseCallback("Perspective Corrected", click_warped_image)

def export_selected_squares():
    global warped, grid_squares, selected_squares, square_size, square_dst
    
    if not selected_squares or not grid_squares or warped is None:
        print("No squares selected or no grid available.")
        return
    
    # Create output directory if it doesn't exist
    output_dir = "exported_squares"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # Clean up old files (optional)
        for file in os.listdir(output_dir):
            if file.endswith(".jpg"):
                os.remove(os.path.join(output_dir, file))
    
    # Find the main square (red square) grid coordinates
    # Convert from pixel coordinates to grid cell indices
    main_square_points = square_dst.astype(np.int32)
    padding = 50  # Same as in apply_perspective_transform
    
    # Find the grid cell that contains the top-left corner of the main square
    main_square_tl = None
    for i, square in enumerate(grid_squares):
        top_left, bottom_right = square
        if (top_left[0] <= main_square_points[0][0] < bottom_right[0] and
            top_left[1] <= main_square_points[0][1] < bottom_right[1]):
            main_square_tl = i
            break
    
    if main_square_tl is None:
        print("Warning: Could not locate main square in grid. Using relative positions from grid.")
        # Just use the first grid square as reference point
        if grid_squares:
            main_tl_x = grid_squares[0][0][0]
            main_tl_y = grid_squares[0][0][1]
        else:
            print("Error: No grid squares available")
            return
    else:
        main_tl_x = grid_squares[main_square_tl][0][0]
        main_tl_y = grid_squares[main_square_tl][0][1]
    
    # Save each selected square as a separate image
    count = 0
    for square_idx in selected_squares:
        if square_idx >= len(grid_squares):
            continue
            
        # Get the coordinates of this square
        (tl_x, tl_y), (br_x, br_y) = grid_squares[square_idx]
        
        # Calculate the relative position to the main square in grid cells
        # Horizontal offset (x): negative is left of main square, positive is right
        # Vertical offset (y): negative is above main square, positive is below
        rel_x = int((tl_x - main_tl_x) / square_size)
        rel_y = int((tl_y - main_tl_y) / square_size)
        
        # The exported image should not contain any grid lines
        # Just the content of the selected square
        
        # Extract this region from the warped image
        square_img = warped[tl_y:br_y, tl_x:br_x]
        
        # Check if we have a prediction for this square
        if square_idx in grid_predictions:
            class_name, confidence = grid_predictions[square_idx]
            # Create filename with position and prediction
            filename = f"{rel_x}_{rel_y}_class-{class_name}_{confidence:.2f}.jpg"
        else:
            # Create filename based on relative position only
            filename = f"{rel_x}_{rel_y}.jpg"
        
        filepath = os.path.join(output_dir, filename)
        
        # Save the image
        cv2.imwrite(filepath, square_img)
        count += 1
    
    print(f"Exported {count} images to folder '{output_dir}'")

def main(image_path=None):
    global img, img_copy, warped, last_selected_point, point_move_step, points
    
    # Load the image
    if image_path is None:
        image_path = "surface3.jpg"  # Default image if none provided
    
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image '{image_path}'")
        return
    
    # Create a copy of the image to draw on
    img_copy = img.copy()
    
    # Create windows
    cv2.namedWindow("Image")
    cv2.namedWindow("Perspective Corrected")
    
    # Set the mouse callback
    cv2.setMouseCallback("Image", click_event)
    
    # Display instructions
    print("\n=== Interactive Perspective Correction Tool ===")
    print("Instructions:")
    print("1. Click on 4 points to define the area for perspective correction.")
    print("2. The points should be in clockwise order starting from top-left.")
    print("3. Click and drag any point to adjust and see real-time correction.")
    print("4. Press number keys 1-4 to select the corresponding point.")
    print("5. Use arrow keys or WASD to move the selected point.")
    print("6. Press '+' or '-' to change movement step size for point movement.")
    print("7. Press 'r' to reset all points.")
    print("8. Press 'e' to export the image.")
    print("9. Press 'q' to quit.")
    
    # Display the image
    cv2.imshow("Image", img_copy)
    
    # Wait for user input
    while True:
        key = cv2.waitKey(20) & 0xFF
        
        # If 'q' is pressed, quit
        if key == ord('q'):
            break
        
        # If 'r' is pressed, reset
        elif key == ord('r'):
            points.clear()
            last_selected_point = -1
            img_copy = img.copy()
            cv2.imshow("Image", img_copy)
            
            # Clear the perspective corrected window if it exists
            if warped is not None:
                warped = np.zeros_like(img)
                cv2.imshow("Perspective Corrected", warped)
                warped = None
        
        # If 'e' is pressed, export the image
        elif key == ord('e'):
            if warped is not None:
                if not selected_squares:
                    # If no squares are selected, export the entire warped image
                    cv2.imwrite("corrected_surface.jpg", warped)
                    print("Image saved as 'corrected_surface.jpg'")
                else:
                    # Export selected grid squares as one combined image
                    export_selected_squares()
                    print(f"Exported {len(selected_squares)} selected squares as 'selected_squares.jpg'")
        
        # Arrow keys can be inconsistent across systems, so we'll rely on WASD for movement
        # UP arrow key
        elif key == 82 or key == 0xFF52 or key == 65362 or key == 63232:
            # Check if we have any points and one is selected
            if points and last_selected_point != -1:
                x, y = points[last_selected_point]
                y -= point_move_step
                points[last_selected_point] = (x, y)
                update_display()
                if len(points) == 4:
                    apply_perspective_transform()
                    
        # DOWN arrow key
        elif key == 84 or key == 0xFF54 or key == 65364 or key == 63233:
            if points and last_selected_point != -1:
                x, y = points[last_selected_point]
                y += point_move_step
                points[last_selected_point] = (x, y)
                update_display()
                if len(points) == 4:
                    apply_perspective_transform()
                    
        # LEFT arrow key
        elif key == 81 or key == 0xFF51 or key == 65361 or key == 63234:
            if points and last_selected_point != -1:
                x, y = points[last_selected_point]
                x -= point_move_step
                points[last_selected_point] = (x, y)
                update_display()
                if len(points) == 4:
                    apply_perspective_transform()
                    
        # RIGHT arrow key
        elif key == 83 or key == 0xFF53 or key == 65363 or key == 63235:
            if points and last_selected_point != -1:
                x, y = points[last_selected_point]
                x += point_move_step
                points[last_selected_point] = (x, y)
                update_display()
                if len(points) == 4:
                    apply_perspective_transform()
        
        # Also support direct WASD keys for movement as a backup
        elif key in [ord('w'), ord('W')]:
            if points and last_selected_point != -1:
                x, y = points[last_selected_point]
                y -= point_move_step
                points[last_selected_point] = (x, y)
                update_display()
                if len(points) == 4:
                    apply_perspective_transform()
        elif key in [ord('s'), ord('S')]:
            if points and last_selected_point != -1:
                x, y = points[last_selected_point]
                y += point_move_step
                points[last_selected_point] = (x, y)
                update_display()
                if len(points) == 4:
                    apply_perspective_transform()
        elif key in [ord('a'), ord('A')]:
            if points and last_selected_point != -1:
                x, y = points[last_selected_point]
                x -= point_move_step
                points[last_selected_point] = (x, y)
                update_display()
                if len(points) == 4:
                    apply_perspective_transform()
        elif key in [ord('d'), ord('D')]:
            if points and last_selected_point != -1:
                x, y = points[last_selected_point]
                x += point_move_step
                
                # Update the point position
                points[last_selected_point] = (x, y)
                update_display()
                
                # Update the perspective transform if we have all 4 points
                if len(points) == 4:
                    apply_perspective_transform()
        
        # Number keys 1-4 for selecting points
        elif key >= ord('1') and key <= ord('4'):
            point_num = key - ord('1')  # Convert to 0-based index
            if point_num < len(points):
                last_selected_point = point_num
                print(f"Selected point {point_num+1}")
                update_display()
            else:
                print(f"Point {point_num+1} not yet defined. Please click to add more points.")
        
        # Adjust movement step size
        elif key == ord('+') or key == ord('='):
            point_move_step = min(10, point_move_step + 1)
            print(f"Movement step size: {point_move_step} pixels")
        elif key == ord('-'):
            point_move_step = max(1, point_move_step - 1)
            print(f"Movement step size: {point_move_step} pixels")
    
    # Close all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive perspective correction tool")
    parser.add_argument("-i", "--image", type=str, help="Path to the input image")
    args = parser.parse_args()
    
    main(args.image)