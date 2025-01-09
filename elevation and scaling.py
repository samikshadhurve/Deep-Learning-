import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d

class VideoStream:
    def __init__(self):
        print("Initialized")
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        self.color_image = None
        self.depth_frame = None

    def release(self):
        # Stop the pipeline and close windows
        self.pipeline.stop()
        cv2.destroyAllWindows()

    def stream_frames(self):
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            self.depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not self.depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(self.depth_frame.get_data())
            self.color_image = np.asanyarray(color_frame.get_data())

            # Apply color map to depth image for visualization
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Display the depth image
            cv2.imshow("Depth Image", depth_colormap)

            # Add grid to the color image and display it
            self.scaling()
            cv2.imshow("Color Image with Grid", self.color_image)


            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def scaling(self):
        grid_points = [(100, 100), (100, 200), (100, 300), (100, 400)]

        x_values = [100, 200, 300, 400, 500, 600]

        for x in x_values:
            cv2.line(self.color_image, (x, 0), (x, self.color_image.shape[0]), (255, 255, 255), 1)

        y_values = [100, 200, 300, 400]
        for y in y_values:
            cv2.line(self.color_image, (0, y), (self.color_image.shape[1], y), (255, 255, 255), 1)

        cv2.line(self.color_image, (100, 0), (100, 480), (0, 0, 255), 2)
        for (x, y) in grid_points:
            distance = self.depth_frame.get_distance(x, y) * 1000
            distance_text = f"{int(distance)} mm"
            cv2.putText(self.color_image, distance_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)



def main():
    stream = VideoStream()
    try:
        stream.stream_frames()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        stream.release()

if __name__ == "__main__":
    main()
