import cv2
import numpy as np


def apply_thresholding(image):
    """Apply simple thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresholded


def apply_region_growing(image):
    """Apply simple region growing by thresholding based on intensity."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    region_grown = cv2.dilate(thresh, None, iterations=5)
    return region_grown


def apply_watershed(image):
    """Apply watershed segmentation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Ensure binary image is used in the watershed process
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(dilated, cv2.DIST_L2, 5)

    # Apply threshold to get sure foreground
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Sure background is dilated
    sure_bg = cv2.dilate(dilated, kernel, iterations=3)

    # Convert sure_bg and sure_fg to uint8 before subtraction
    sure_bg = np.uint8(sure_bg)
    sure_fg = np.uint8(sure_fg)

    # Unknown regions are the difference between background and foreground
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Convert to 3 channels to use for watershed
    markers = np.zeros_like(gray, dtype=np.int32)
    markers[sure_fg == 255] = 1
    markers[sure_bg == 255] = 2
    markers[unknown == 255] = 0

    # Apply the watershed algorithm
    cv2.watershed(image, markers)

    # Mark boundaries with red color
    image[markers == -1] = [0, 0, 255]

    return image


def display_image(title, image):
    """Utility function to display the image."""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # Load sample image
    image = cv2.imread(r"C:\Users\samik\OneDrive\Documents\clg_assignments\image_registration_dataset\eiffel_base.jpg")  # Replace with your image path

    if image is None:
        print("Error: Image not found.")
        return

    while True:
        print("\nSelect an image segmentation technique:")
        print("1. Thresholding")
        print("2. Region Growing")
        print("3. Watershed Segmentation")
        print("4. Exit")

        choice = input("Enter choice: ")

        if choice == '1':
            segmented_image = apply_thresholding(image)
            display_image("Thresholding Result", segmented_image)

        elif choice == '2':
            segmented_image = apply_region_growing(image)
            display_image("Region Growing Result", segmented_image)

        elif choice == '3':
            segmented_image = apply_watershed(image)
            display_image("Watershed Segmentation Result", segmented_image)

        elif choice == '4':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
