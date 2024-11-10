import math
import nn
import pygame

pygame.init()
pygame.font.init()
arial = pygame.font.SysFont('Arial', 30)

WIDTH = 28  # number of squares horizontally
HEIGHT = 28  # number of squares vertically
SIZE = 20  # size of each square in pixels

SCREEN_WIDTH = WIDTH * SIZE + 100
SCREEN_HEIGHT = HEIGHT * SIZE + 300

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
ERASER = (237, 148, 155)
PENCIL = (108, 106, 112)
BAR_FILL = (237, 255, 145)
BAR_EMPTY = (90, 97, 54)

def get_distance(coords_1: tuple, coords_2: tuple):
    x1, y1 = coords_1
    x2, y2 = coords_2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_coords_of_square(row, col):
    return col*SIZE + SIZE//2, row*SIZE + SIZE//2


class Button(pygame.Rect):
    def __init__(self, left, top, width, height, colors):
        super().__init__(left, top, width, height)
        self.last_pressed = pygame.time.get_ticks()
        self.state = 1
        self.colors = colors
    
    def has(self, coords: tuple):
        '''check if (x, y) is on the rectangle'''
        x, y = coords
        
        return ((self.left <= x <= self.left + self.width)
                and (self.top <= y <= self.top + self.height))
    
    @property
    def color(self):
        return self.colors.get(self.state, (0, 0, 0))
    
    def press(self) -> bool:
        '''return whether button can be pressed
        
        updates state if can be pressed
        '''
        current_time = pygame.time.get_ticks()
        if current_time - self.last_pressed > 100:
            self.state = -self.state
            self.last_pressed = current_time
            return True
        else:
            return False


class Bar:
    def __init__(self, left, top, width, height):
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.value = 0.3 # value between 0.0 and 1.0
    
    def draw(self, screen: pygame.Surface):
        fill_rect_height = int(self.value * self.height)
        empty_rect_height = self.height - fill_rect_height
        
        pygame.draw.rect(
            screen, BAR_FILL, 
            (self.left, self.top + empty_rect_height, self.width, fill_rect_height)
        )
        
        pygame.draw.rect(
            screen, BAR_EMPTY, 
            (self.left, self.top, self.width, empty_rect_height)
        )


class GridScene:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Digit Recognition')
        self.grid = [[0.0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
        self.clock = pygame.time.Clock()
        self.brush_size = 2
        self.last_paint_time = pygame.time.get_ticks()
        self.toggle_button = Button(WIDTH*SIZE + 25, 25, 50, 50, {1: PENCIL, -1: ERASER})
        self.clear_button = Button(WIDTH*SIZE + 25, 125, 50, 50, {})
        self.BRUSH_SPEED = 20 # lower = less delay
        self.bars = []
        self.digits = []
        for i in range(10):
            bar = Bar(i*50 + 25, HEIGHT*SIZE + 25, 25, 100)
            digit = arial.render(str(i), False, BLACK)
            self.digits.append(digit)
            self.bars.append(bar)
    
    def render(self):
        for row in range(HEIGHT):
            for col in range(WIDTH):
                color = tuple(c * self.grid[row][col] for c in WHITE)
                pygame.draw.rect(
                    self.screen, color,
                    (col * SIZE, row * SIZE, SIZE, SIZE)
                )
        
        pygame.draw.rect(
            self.screen, self.toggle_button.color, self.toggle_button
        )
        
        pygame.draw.rect(
            self.screen, self.clear_button.color, self.clear_button
        )
        
        for i in range(10): # draw bars and digits underneath
            self.bars[i].draw(self.screen)
            self.screen.blit(self.digits[i], (i*50 + 30, HEIGHT*SIZE + 140))
            
        predicted_digit = self.prediction.index(max(self.prediction))
        prediction_text = arial.render('Predicted: {}'.format(predicted_digit), False, BLACK)
        self.screen.blit(prediction_text, (30, HEIGHT*SIZE+240))
    
    def predict(self):
        flattened = nn.flatten(self.grid)
        self.prediction = nn.predict(flattened)
        prediction_min = min(self.prediction)
        prediction_range = max(self.prediction) - min(self.prediction)
        
        for i in range(10): # normalize prediction values
            self.prediction[i] -= prediction_min
            self.prediction[i] /= prediction_range
            
            # update corresponding bar
            self.bars[i].value = self.prediction[i]
    
    def paint(self):
        current_time = pygame.time.get_ticks()
        if pygame.time.get_ticks() - self.last_paint_time < self.BRUSH_SPEED:
            return
        self.last_paint_time = current_time
        
        if self.mouse_down:
            mx, my = pygame.mouse.get_pos()
            
            if not (0 <= mx <= WIDTH * SIZE and 0 <= my <= HEIGHT * SIZE):
                return
            
            for row in range(HEIGHT):
                for col in range(WIDTH):
                    dist = get_distance((mx, my), get_coords_of_square(row, col))
                    dist = max(dist, 0.001) # prevent division by zero
                    paint_intensity = max(0, ((self.brush_size*SIZE - dist) / dist)) * self.toggle_button.state
                    
                    self.grid[row][col] += paint_intensity
                    self.grid[row][col] = min(self.grid[row][col], 1.0)
                    self.grid[row][col] = max(self.grid[row][col], 0.0)

    def run(self):
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.toggle_button.has(pygame.mouse.get_pos()):
                        self.toggle_button.press()
                    
                    if self.clear_button.has(pygame.mouse.get_pos()):
                        self.grid = [[0.0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
            
            self.mouse_down = pygame.mouse.get_pressed()[0]
            self.screen.fill(WHITE)
            self.paint()
            self.predict()
            self.render()
            pygame.display.flip()
            
            self.clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    scene = GridScene()
    scene.run()
