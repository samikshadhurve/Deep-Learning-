#IT WORKS
import numpy as np
import pygame
from rplidar import RPLidar
import threading


class MyRobot:
    def __init__(self):
        pygame.init()
        self.port_name = 'COM4'  # Change to your port
        self.lidar = RPLidar(self.port_name, baudrate=115200, timeout=3)
        self.width, self.height = 1100, 950
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Real-Time LIDAR Scanning")
        self.clock = pygame.time.Clock()
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0, 128)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        self.CIRCLE_COLOR = (0, 0, 0)
        self.TEXT_COLOR = (0, 0, 0)
        self.AXIS_COLOR = (0, 0, 0)
        self.ARROW_COLOR = None
        self.MAX_DISTANCE = 4000
        self.arrow_length = 300
        self.arrow_width = 7
        self.arrow_angle = 0
        self.scan_data = []
        self.obstacle_detected = False
        self.buttons = []
        self.button_color = self.BLUE
        self.button_hover_color = self.GREEN
        self.button_click_color = self.RED
        self.button_width, self.button_height = 200, 50
        self.left_button_pos = None
        self.right_button_pos = None
        self.forward_button_pos = None
        self.backward_button_pos = None
        self.button_clicked = False
        self.clicked_button = None

    def health_check(self):
        # info = self.lidar.get_info()
        # print(info)
        health = self.lidar.get_health()
        print(health)
        if health[0] == 'Good':
            return True
        else:
            return False

    def draw_axes(self):
        font = pygame.font.SysFont(None, 18)
        angle_labels = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
        pygame.draw.line(self.screen, self.AXIS_COLOR, (self.width // 2, 0), (self.width // 2, self.height), 1)
        pygame.draw.line(self.screen, self.AXIS_COLOR, (0, self.height // 2), (self.width, self.height // 2), 1)
        center_x = self.width // 2
        center_y = self.height // 2
        radius = min(self.width, self.height) // 2 - 20
        for angle in angle_labels:
            rad = np.radians(angle)
            end_x = center_x + int(np.cos(rad) * radius)
            end_y = center_y - int(np.sin(rad) * radius)
            pygame.draw.line(self.screen, self.AXIS_COLOR, (center_x, center_y), (end_x, end_y), 1)
            label_x = center_x + int(np.cos(rad) * (radius + 20))
            label_y = center_y - int(np.sin(rad) * (radius + 20))
            label = f"{angle}°"
            text = font.render(label, True, self.TEXT_COLOR)
            text_rect = text.get_rect(center=(label_x, label_y))
            self.screen.blit(text, text_rect)

    def draw_arrow(self, angle):
        self.arrow_angle = angle
        arrow_end_x = self.width // 2 + np.cos(np.radians(angle)) * self.arrow_length
        arrow_end_y = self.height // 2 - np.sin(np.radians(angle)) * self.arrow_length
        pygame.draw.line(self.screen, (255,255,0), (self.width // 2, self.height // 2),
                         (arrow_end_x, arrow_end_y), 5)

    def draw_buttons(self):
        font = pygame.font.SysFont(None, 40)
        button_margin = 20
        button_width = 150
        button_height = 50
        self.left_button_pos = (button_margin, button_margin + 70)
        self.right_button_pos = (self.width - button_width - button_margin - 5, button_margin + 70)
        self.forward_button_pos = ((self.width - button_width) // 2, button_margin + 70)
        self.backward_button_pos = ((self.width - button_width) // 2, self.height - button_height - button_margin - 70)

        buttons = [("Left", self.left_button_pos), ("Right", self.right_button_pos),
                   ("Forward", self.forward_button_pos), ("Backward", self.backward_button_pos)]

        for i, (text, pos) in enumerate(buttons):  #using enumerate
            if self.clicked_button == text:
                button_color = self.button_click_color
            else:
                button_color = self.button_color

            pygame.draw.rect(self.screen, button_color, (*pos, button_width, button_height))
            text_render = font.render(text, True, self.WHITE)
            text_rect = text_render.get_rect(center=(pos[0] + button_width // 2, pos[1] + button_height // 2))
            self.screen.blit(text_render, text_rect)
        return buttons

    def is_mouse_over_button(self, x, y):
        mouse_pos = pygame.mouse.get_pos()
        return x <= mouse_pos[0] <= x + self.button_width and y <= mouse_pos[1] <= y + self.button_height

    # def update_angle_thread(self, forward, backward, arr_angle):
    #     while True:
    #         best_angle = self.update_angle(forward, backward, arr_angle)
    #         if best_angle is not None:
    #             self.arrow_angle = best_angle
    #         pygame.time.delay(100)

    def update_angle(self, forward, backward, arr_angle):
        # Find the best angle dynamically between a range (e.g., 0-90, 90-180, etc.)
        best_angle = None
        min_diff = float('inf')
        range_start,range_end=0,0

        if forward:
            range_start, range_end = 60,120
        elif backward:
            range_start, range_end = 240,300

        for angle, distance in self.scan_data:
            if range_start <= angle <= range_end and distance > 1000:
                angle_diff = abs(arr_angle - angle)
                if angle_diff < min_diff:
                    min_diff = angle_diff
                    best_angle = angle

        return best_angle if best_angle is not None else arr_angle

    def obstacle_scanning(self,arrow_angle):
        self.screen.fill(self.WHITE)
        self.draw_axes()
        self.draw_buttons()
        font = pygame.font.SysFont(None, 24)
        for i in range(1, 11):
            radius = (self.MAX_DISTANCE / 10) * i
            scaled_radius = int((radius / self.MAX_DISTANCE) * (min(self.width, self.height) // 2))
            pygame.draw.circle(self.screen, self.CIRCLE_COLOR, (self.width // 2, self.height // 2), scaled_radius, 1)
            label = f"{int(radius)} mm"
            text = font.render(label, True, self.TEXT_COLOR)
            self.screen.blit(text, (self.width // 2 + 5, (self.height // 2 - scaled_radius)))

        for angle, distance in self.scan_data:
            if distance == 0:
                distance=self.MAX_DISTANCE
                continue
            if distance <= self.MAX_DISTANCE:
                theta = np.radians(angle)
                endpoint_x = self.width / 2 + distance * np.cos(theta) * self.width / (2 * self.MAX_DISTANCE)
                endpoint_y = self.height / 2 - distance * np.sin(theta) * self.height / (2 * self.MAX_DISTANCE)
                pygame.draw.line(self.screen, self.GREEN[:3], (self.width / 2, self.height / 2),
                                 (endpoint_x, endpoint_y), 2)
                self.ARROW_COLOR=(255,255,0)
            if distance <= 800 and ((angle +5) or (angle-5)==arrow_angle):
                self.obstacle_detected = True
                # self.ARROW_COLOR = self.RED
                if self.is_mouse_over_button(*self.left_button_pos):
                    self.clicked_button = "Left"
                elif self.is_mouse_over_button(*self.right_button_pos):
                    self.clicked_button = "Right"

    def fetch_data(self):
        while True:
            try:
                for scan in self.lidar.iter_scans():
                    self.scan_data = [(angle, distance) for (_, angle, distance) in scan]
                    if len(self.scan_data) >= 360:
                        break
            except Exception as e:
                print(f"Error: {e}")
                self.lidar.stop_motor()
                break

    def run(self):
        forward, backward = False, False
        self.health_check()


        # Start a thread to fetch LIDAR data
        data_thread = threading.Thread(target=self.fetch_data)
        data_thread.daemon = True
        data_thread.start()

        while True:

            self.screen.fill(self.WHITE)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.lidar.stop()  # Ensure the LIDAR is stopped properly
                    pygame.quit()
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left mouse button
                    if self.is_mouse_over_button(*self.forward_button_pos):
                        forward = True
                        backward = False
                        self.clicked_button = "Forward"
                        self.arrow_angle = 90
                        # self.ARROW_COLOR=(255,255,0)
                    elif self.is_mouse_over_button(*self.backward_button_pos):
                        forward = False
                        backward = True
                        self.clicked_button = "Backward"
                        self.arrow_angle = 270
                        # self.ARROW_COLOR = (255, 255, 0)
                    # elif self.is_mouse_over_button(*self.left_button_pos):
                    #     self.clicked_button = "Left"
                    #     # Adjust based on LIDAR data
                    #     self.arrow_angle = self.update_angle(forward, backward, self.arrow_angle)
                    # elif self.is_mouse_over_button(*self.right_button_pos):
                    #     self.clicked_button = "Right"
                    #     # Adjust based on LIDAR data
                    #     self.arrow_angle = self.update_angle(forward, backward, self.arrow_angle)


            self.obstacle_scanning(self.arrow_angle)
            # if not self.obstacle_detected:
            #     self.ARROW_COLOR=(255,255,0)
            if self.obstacle_detected:
                self.ARROW_COLOR = self.RED
                self.arrow_angle=self.update_angle(forward,backward,self.arrow_angle)
                self.draw_arrow(self.arrow_angle)

            self.clock.tick(30)
            pygame.display.flip()

        self.lidar.stop_motor()
        self.lidar.disconnect()
        pygame.quit()


if __name__ == "__main__":
    robot = MyRobot()
    robot.run()
