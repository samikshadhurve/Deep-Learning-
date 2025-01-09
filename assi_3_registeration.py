import cv2
import numpy as np
import matplotlib.pyplot as plt

class transformations:
    def __init__(self, fixed_image_path, moving_image_path):
        self.fix_img = r"C:\Users\samik\OneDrive\Documents\clg_assignments\image_registration_dataset\rio_base_1.jpg"
        self.move_img = r"C:\Users\samik\OneDrive\Documents\clg_assignments\image_registration_dataset\rio_register.jpg"

    def load_images(self):
        fixed_image = cv2.imread(self.fix_img, cv2.IMREAD_GRAYSCALE)
        moving_image = cv2.imread(self.move_img, cv2.IMREAD_GRAYSCALE)
        return fixed_image, moving_image

    def affine_transform(self, image, matrix):
        rows, cols = image.shape
        return cv2.warpAffine(image, matrix, (cols, rows))

    def translate_image(self, image, tx, ty):
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        return self.affine_transform(image, matrix)

    def rotate_image(self, image, angle, center=None):
        rows, cols = image.shape
        if center is None:
            center = (cols // 2, rows // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1)
        return self.affine_transform(image, matrix)

    def scale_image(self, image, sx, sy):
        matrix = np.float32([[sx, 0, 0], [0, sy, 0]])
        return self.affine_transform(image, matrix)

    def shear_image(self, image, shx, shy):
        rows, cols = image.shape
        pts1 = np.float32([[0, 0], [cols, 0], [0, rows]])
        pts2 = np.float32([[0, 0], [cols + shx, shy], [shx, rows + shy]])
        matrix = cv2.getAffineTransform(pts1, pts2)
        return self.affine_transform(image, matrix)

    def reflect_image(self, image, axis='x'):
        if axis == 'x':
            return cv2.flip(image, 0)
        elif axis == 'y':
            return cv2.flip(image, 1)
        else:
            raise ValueError("Axis must be 'x' or 'y'")

    def register_images(self, fixed_image, moving_image, ch):
        registered_image = moving_image.copy()
        if ch == 1:
            # Translation
            registered_image = self.translate_image(registered_image, 20, 10)
        elif ch == 2:
            # Rotation
            registered_image = self.rotate_image(registered_image, 15)
        elif ch == 3:
            # Scaling
            registered_image = self.scale_image(registered_image, 1.1, 1.1)
        elif ch == 4:
            # Shearing
            registered_image = self.shear_image(registered_image, 50, 50)
        elif ch == 5:
            # Reflection
            registered_image = self.reflect_image(registered_image, 'y')
        else:
            print("Invalid choice!")
        return registered_image

    def plot_images(self, fixed_image, moving_image, registered_image):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(fixed_image, cmap='gray')
        axs[0].set_title('Fixed Image')
        axs[0].axis('off')

        axs[1].imshow(moving_image, cmap='gray')
        axs[1].set_title('Moving Image')
        axs[1].axis('off')

        axs[2].imshow(registered_image, cmap='gray')
        axs[2].set_title('Registered Image')
        axs[2].axis('off')

        plt.tight_layout()
        plt.show()


def main():
    # Load images (replace with your image paths)
    fixed_image_path = r"C:\Users\hp\Desktop\clg assigments\girl1.PNG"
    moving_image_path = r"C:\Users\hp\Desktop\clg assigments\girl2.PNG"

    # Create an instance of the transformations class
    transform = transformations(fixed_image_path, moving_image_path)

    fixed_image, moving_image = transform.load_images()

    while True:
        print("\nMenu:")
        print("1. Translation")
        print("2. Rotation")
        print("3. Scaling")
        print("4. Shearing")
        print("5. Reflection")
        print("6. Exit")

        ch = int(input("Enter your choice: "))

        if ch == 6:
            print("Exiting...")
            break

        # Perform image registration
        registered_image = transform.register_images(fixed_image, moving_image, ch)

        # Display results
        transform.plot_images(fixed_image, moving_image, registered_image)


if __name__ == "__main__":
    main()
