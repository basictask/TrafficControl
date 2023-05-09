import pygame
from pygame import gfxdraw
import numpy as np


class Window:
    def __init__(self, sim, steps_per_update: int, n_steps: int, config=None):
        """
        Sets up a simulation window
        :param sim: Simulation object which passes the locations and coordinates
        :param steps_per_update: How many steps should pass in the simulation before updating the window
        :param n_steps: Number of steps to run for in total
        :param config: Additional configurations
        """
        # Simulation to draw
        if config is None:
            config = {}

        self.sim = sim
        self.fps = None
        self.zoom = None
        self.width = None
        self.screen = None
        self.offset = None
        self.height = None
        self.bg_color = None
        self.text_font = None
        self.mouse_down = None
        self.mouse_last = None

        # Set the default configuration
        self.width = 1400
        self.height = 900
        # self.bg_color = (50, 150, 50)  # Greenish
        self.bg_color = (255, 255, 255)  # White
        # self.bg_color = (40, 42, 54)  # PyCharm-dark
        self.font_color = (248, 248, 242)  # Pycharm-dark font color
        self.n_steps = n_steps
        self.i_steps = 0
        self.steps_per_update = steps_per_update

        self.fps = 60
        self.zoom = 5
        self.offset = (0, 0)

        self.mouse_last = (0, 0)
        self.mouse_down = False

        # Update configurations
        for attr, val in config.items():
            setattr(self, attr, val)

    def loop(self, loop=None):
        """
        Shows a window visualizing the simulation and runs the loop function
        :param loop: A looping function based on which teh update should run
        """
        # Create a pygame window
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.flip()

        # Fixed fps
        clock = pygame.time.Clock()

        # To draw text
        pygame.font.init()
        self.text_font = pygame.font.SysFont('Lucida Console', 16)

        # Draw loop
        running = True
        while running:
            # Update simulation
            if loop:
                loop(self.sim)

            # Draw simulation
            self.draw()

            # Update window
            pygame.display.update()
            clock.tick(self.fps)

            # Handle all events
            for event in pygame.event.get():
                # Quit program if window is closed
                if event.type == pygame.QUIT:
                    running = False
                # Handle mouse events
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # If mouse button down
                    if event.button == 1:
                        # Left click
                        x, y = pygame.mouse.get_pos()
                        x0, y0 = self.offset
                        self.mouse_last = (x - x0 * self.zoom, y - y0 * self.zoom)
                        self.mouse_down = True
                    if event.button == 4:
                        # Mouse wheel up
                        self.zoom *= (self.zoom**2 + self.zoom/4 + 1) / (self.zoom**2 + 1)
                    if event.button == 5:
                        # Mouse wheel down 
                        self.zoom *= (self.zoom**2 + 1) / (self.zoom**2 + self.zoom/4 + 1)
                elif event.type == pygame.MOUSEMOTION:
                    # Drag content
                    if self.mouse_down:
                        x1, y1 = self.mouse_last
                        x2, y2 = pygame.mouse.get_pos()
                        self.offset = ((x2 - x1) / self.zoom, (y2 - y1) / self.zoom)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.mouse_down = False
            self.i_steps += self.steps_per_update
            if self.i_steps >= self.n_steps > 0:
                running = False
        # Exit after the loop is done
        pygame.quit()

    def run(self):
        """
        Runs the simulation by updating in every loop.
        """
        def loop(sim):
            sim.run(self.steps_per_update)
        self.loop(loop)

    def convert(self, x, y=None):
        """
        Converts coordinates from game space to screen space
        :param x: The x coordinate to convert, or a list or tuple of (x, y) coordinates to convert
        :param y: The y coordinate to convert (if x is a float)
        :return: A tuple of (x, y) screen coordinates, or a list of such tuples (if x is a list or tuple)
        """
        if isinstance(x, list):
            return [self.convert(e[0], e[1]) for e in x]
        if isinstance(x, tuple):
            return self.convert(*x)
        return int(self.width / 2 + (x + self.offset[0]) * self.zoom), int(self.height / 2 + (y + self.offset[1]) * self.zoom)

    def inverse_convert(self, x, y=None):
        """
        Converts coordinates from screen space to game space
        :param x: The x coordinate to convert, or a list or tuple of (x, y) coordinates to convert
        :param y: The y coordinate to convert (if x is an int)
        :return: A tuple of (x, y) game coordinates, or a list of such tuples (if x is a list or tuple)
        """
        if isinstance(x, list):
            return [self.convert(e[0], e[1]) for e in x]
        if isinstance(x, tuple):
            return self.convert(*x)
        return int(-self.offset[0] + (x - self.width/2)/self.zoom), int(-self.offset[1] + (y - self.height/2)/self.zoom)

    def background(self, r, g, b):
        """
        Sets the background color of the screen.
        :param r: The red component of the background color (0-255)
        :param g: The green component of the background color (0-255)
        :param b: The blue component of the background color (0-255)
        """
        self.screen.fill((r, g, b))

    def line(self, start_pos, end_pos, color):
        """
        Draws a line on the screen from the starting position to the ending position with the given color.
        :param start_pos: The (x, y) coordinates of the starting position of the line
        :param end_pos: The (x, y) coordinates of the ending position of the line
        :param color: The color of the line, as an (R, G, B) tuple with values from 0 to 255
        """
        gfxdraw.line(self.screen, *start_pos, *end_pos, color)

    def rect(self, pos, size, color):
        """
        Draws a rectangle on the screen with the given position, size, and color
        :param pos: The (x, y) coordinates of the top-left corner of the rectangle
        :param size: The width and height of the rectangle
        :param color: The color of the rectangle, as an (R, G, B) tuple with values from 0 to 255
        """
        gfxdraw.rectangle(self.screen, (*pos, *size), color)

    def box(self, pos, size, color):
        """
        Draws an unfilled box.
        :param pos: Position of the top-left corner of the box
        :param size: Size of the box
        :param color: Color of the box
        :return: None
        """
        gfxdraw.box(self.screen, (*pos, *size), color)

    def circle(self, pos, radius, color, filled=True):
        """
        :param pos: The (x, y) coordinates of the center of the circle
        :param radius: The radius of the circle
        :param color: The color of the circle in RGB format
        :param filled: Whether the circle should be filled in or just an outline. Defaults to True
        :return: None
        """
        gfxdraw.aacircle(self.screen, *pos, radius, color)
        if filled:
            gfxdraw.filled_circle(self.screen, *pos, radius, color)

    def polygon(self, vertices, color, filled=True):
        """
        Draws a polygon on the screen
        :param vertices:  A list of (x, y) tuples representing the vertices of the polygon
        :param color: The color of the polygon in RGB format
        :param filled: Whether the polygon should be filled in or just an outline. Defaults to True
        :return: None
        """
        gfxdraw.aapolygon(self.screen, vertices, color)
        if filled:
            gfxdraw.filled_polygon(self.screen, vertices, color)

    @staticmethod
    def vertex(e1, e2, x, y, w, h, cos, sin):
        """
        Calculates the coordinates of a vertex of a rectangle that has been rotated and/or scaled.
        :param e1: Length of the horizontal edge
        :param e2: Length of the vertical edge
        :param x: Horizontal coordinate location
        :param y: Vertical coordinate location
        :param w: Width of the rectangle
        :param h: Height of the rectangle
        :param cos: Cosine angle of rotation
        :param sin: Sine angle of rotation
        :return: New x and y coordinates of the vertex
        """
        return x + (e1 * w * cos + e2 * h * sin) / 2, y + (e1 * w * sin - e2 * h * cos) / 2

    def rotated_box(self, pos, size, angle=None, cos=None, sin=None, centered=True, color=(0, 0, 255), filled=True):
        """
        Draw a rotated rectangle on the screen.
        :param pos: The position of the center of the rectangle as a tuple of (x, y) coordinates
        :param size: The size of the rectangle as a tuple of (width, height)
        :param angle: The angle of rotation in radians
        :param cos: The cosine of the rotation angle
        :param sin: The sine of the rotation angle
        :param centered: Whether the rectangle should be centered at the given position
        :param color: The color of the rectangle as an RGB tuple
        :param filled: Whether the rectangle should be filled or not
        :return: None
        """
        x, y = pos
        w, h = size
        if angle:
            cos, sin = np.cos(angle), np.sin(angle)
        if centered:
            vertices = self.convert([self.vertex(e1, e2, x, y, w, h, cos, sin) for e1, e2 in [(-1, -1), (-1, 1), (1, 1), (1, -1)]])
        else:
            vertices = self.convert([self.vertex(e1, e2, x, y, w, h, cos, sin) for e1, e2 in [(0, -1), (0, 1), (2, 1), (2, -1)]])
        self.polygon(vertices, color, filled=filled)

    def rotated_rect(self, pos, size, angle=None, cos=None, sin=None, centered=True, color=(0, 0, 255)):
        """
        Draws a rotated rectangle on the screen
        :param pos: The (x, y) coordinates of the center of the rectangle
        :param size: The (width, height) of the rectangle
        :param angle: The angle of rotation for the rectangle in degrees. If not provided, use cos and sin instead
        :param cos: The cosine of the angle of rotation for the rectangle. Only used if angle is not provided
        :param sin: The sine of the angle of rotation for the rectangle. Only used if angle is not provided
        :param centered: Whether the rectangle should be drawn centered on pos or with the top-left corner at pos. Defaults to True
        :param color: The color of the rectangle in RGB format
        :return: None
        """
        self.rotated_box(pos, size, angle=angle, cos=cos, sin=sin, centered=centered, color=color, filled=False)

    def arrow(self, pos, size, angle=None, cos=None, sin=None, color=(150, 150, 190)):
        """
        Draws an arrow shape.
        :param pos: The position of the arrow
        :param size: The size of the arrow
        :param angle: The angle of the arrow in radians
        :param cos: The cosine of the arrow angle
        :param sin: The sine of the arrow angle
        :param color: The color of the arrow
        :return: None
        """
        if angle:
            cos, sin = np.cos(angle), np.sin(angle)
        self.rotated_box(pos, size, cos=(cos - sin) / np.sqrt(2), sin=(cos + sin) / np.sqrt(2), color=color, centered=False)
        self.rotated_box(pos, size, cos=(cos + sin) / np.sqrt(2), sin=(sin - cos) / np.sqrt(2), color=color, centered=False)

    def draw_axes(self, color=(100, 100, 100)):
        """
        Draws X and Y axes on the screen using the provided color
        :param color: A tuple of three integers specifying the RGB color value of the axes (default: (100, 100, 100))
        :return: None
        """
        x_start, y_start = self.inverse_convert(0, 0)
        x_end, y_end = self.inverse_convert(self.width, self.height)
        self.line(self.convert((0, y_start)), self.convert((0, y_end)), color)
        self.line(self.convert((x_start, 0)), self.convert((x_end, 0)), color)

    def draw_grid(self, unit=50, color=(150, 150, 150)):
        """
        Draws a grid on the screen with the specified unit size and color
        :param unit: the size of the unit in pixels
        :param color: the color of the grid lines
        :return: None
        """
        x_start, y_start = self.inverse_convert(0, 0)
        x_end, y_end = self.inverse_convert(self.width, self.height)

        n_x = int(x_start / unit)
        n_y = int(y_start / unit)
        m_x = int(x_end / unit) + 1
        m_y = int(y_end / unit) + 1

        for i in range(n_x, m_x):
            self.line(self.convert((unit * i, y_start)), self.convert((unit * i, y_end)), color)
        for i in range(n_y, m_y):
            self.line(self.convert((x_start, unit * i)), self.convert((x_end, unit * i)), color)

    def draw_roads(self):
        """
        Draws all the roads contained in the simulation based oin starting and ending coordinates
        :return: None
        """
        for road in self.sim.roads:
            # Draw road background
            self.rotated_box(road.start, (road.length, 3.7), cos=road.angle_cos, sin=road.angle_sin, color=(180, 180, 220), centered=False)

            # Draw road arrow
            if road.length > 5:
                for i in np.arange(-0.5 * road.length, 0.5 * road.length, 10):
                    pos = (road.start[0] + (road.length / 2 + i + 3) * road.angle_cos, road.start[1] + (road.length / 2 + i + 3) * road.angle_sin)
                    self.arrow(pos, (-1.25, 0.2), cos=road.angle_cos, sin=road.angle_sin)

    def draw_vehicle(self, vehicle, road):
        """
        Draws a vehicle on a road.
        :param vehicle: A Vehicle object, representing the vehicle to be drawn
        :param road: A Road object, representing the road on which the vehicle is moving
        :return: None
        """
        l, h = vehicle.l,  2
        sin, cos = road.angle_sin, road.angle_cos
        x = road.start[0] + cos * vehicle.x
        y = road.start[1] + sin * vehicle.x
        self.rotated_box((x, y), (l, h), cos=cos, sin=sin, centered=True)

    def draw_vehicles(self):
        """
        Iterates over all the vehicles in the simulation and draws each of them
        :return: None
        """
        for road in self.sim.roads:
            # Draw vehicles
            for vehicle in road.vehicles:
                self.draw_vehicle(vehicle, road)

    def draw_signals(self):
        """
        Iterates over each of the traffic lights and draws all of them with the correct light cycle
        :return: None
        """
        for signal in self.sim.traffic_signals:
            for i in range(len(signal.roads)):
                color = (0, 255, 0) if signal.current_cycle[i] else (255, 0, 0)
                for road in signal.roads[i]:
                    a = 0
                    position = ((1 - a) * road.end[0] + a * road.start[0], (1 - a) * road.end[1] + a * road.start[1])
                    self.rotated_box(position, (1, 3), cos=road.angle_cos, sin=road.angle_sin, color=color)

    def draw_status(self):
        """
        Status info on the top-left corner on the screen
        :return: None
        """
        text_time = self.text_font.render(f'time={self.sim.t:.2f}', False, self.font_color)
        text_steps = self.text_font.render(f'steps={self.i_steps}', False, self.font_color)
        text_dist = self.text_font.render(f'distance={self.sim.total_vehicles_distance:.2f}', False, self.font_color)
        self.screen.blit(text_time, (0, 0))
        self.screen.blit(text_steps, (0, 15))
        self.screen.blit(text_dist, (0, 30))

    def draw(self):
        """
        Drawing iteration from the bottom up: background -> roads -> vehicles -> signals -> status
        :return: None
        """
        self.background(*self.bg_color)  # Fill background
        self.draw_roads()
        self.draw_vehicles()
        self.draw_signals()
        self.draw_status()  # Draw status info
