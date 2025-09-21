import numpy as np
from PIL import Image, ImageDraw
import random
import math

def generate_perlin_noise(width, height, scale=10, octaves=6, persistence=0.5, lacunarity=2.0):
    # Implementación básica de ruido Perlin
    noise = np.zeros((height, width))
    for octave in range(octaves):
        freq = lacunarity ** octave
        amp = persistence ** octave
        noise += amp * np.random.randn(height, width) * freq
    return noise

def generate_poisson_disc_samples(width, height, min_distance, num_samples=30):
    # Implementación simplificada de Poisson disc sampling
    points = []
    for _ in range(num_samples):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        points.append((x, y))
    return points

def random_walk_lines(start_x, start_y, length=50, step_size=2):
    # Genera una línea fractal usando random walk
    points = [(start_x, start_y)]
    for _ in range(length):
        angle = random.uniform(0, 2 * math.pi)
        dx = step_size * math.cos(angle)
        dy = step_size * math.sin(angle)
        new_x = int(points[-1][0] + dx)
        new_y = int(points[-1][1] + dy)
        points.append((new_x, new_y))
    return points
