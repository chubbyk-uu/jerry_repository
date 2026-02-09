import math
import tkinter as tk


class Body:
    def __init__(self, mass, x, y, vx, vy):
        self.mass = mass
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy


def compute_forces(bodies, G=1.0):
    forces = []
    for i, b in enumerate(bodies):
        fx = fy = 0.0
        for j, other in enumerate(bodies):
            if i == j:
                continue
            dx = other.x - b.x
            dy = other.y - b.y
            dist2 = dx * dx + dy * dy
            if dist2 == 0:
                continue
            inv_dist3 = 1.0 / (dist2 ** 1.5)
            fx += G * other.mass * dx * inv_dist3
            fy += G * other.mass * dy * inv_dist3
        forces.append((fx, fy))
    return forces


def step(bodies, dt, G=1.0):
    forces = compute_forces(bodies, G)
    for b, (fx, fy) in zip(bodies, forces):
        b.vx += fx * dt
        b.vy += fy * dt
    for b in bodies:
        b.x += b.vx * dt
        b.y += b.vy * dt


def run_animation(steps=1000, dt=0.01, delta=1e-5):
    bodies1 = [
        Body(1.0, -1.0, 0.0, 0.0, 0.35),
        Body(1.0, 1.0, 0.0, 0.0, -0.35),
        Body(1.0, 0.0, 0.0, 0.0, 0.0),
    ]
    bodies2 = [
        Body(1.0, -1.0 + delta, 0.0, 0.0, 0.35),
        Body(1.0, 1.0, 0.0, 0.0, -0.35),
        Body(1.0, 0.0, 0.0, 0.0, 0.0),
    ]

    size = 600
    scale = 120
    r = 5

    root = tk.Tk()
    root.title("Three-body simulation")
    canvas = tk.Canvas(root, width=size, height=size, bg="black")
    canvas.pack()
    info = tk.Label(root, text="", fg="white", bg="black")
    info.pack()

    shapes1 = []
    shapes2 = []
    colors = ["red", "green", "blue"]
    for c in colors:
        shapes1.append(canvas.create_oval(0, 0, 2*r, 2*r, fill=c))
        shapes2.append(canvas.create_oval(0, 0, 2*r, 2*r, outline=c))

    def draw(shape, body):
        x = body.x * scale + size / 2
        y = body.y * scale + size / 2
        canvas.coords(shape, x - r, y - r, x + r, y + r)

    def update(step_num=0):
        step(bodies1, dt)
        step(bodies2, dt)
        diff = sum(math.hypot(a.x - b.x, a.y - b.y) for a, b in zip(bodies1, bodies2))
        for s, b in zip(shapes1, bodies1):
            draw(s, b)
        for s, b in zip(shapes2, bodies2):
            draw(s, b)
        info.config(text=f"Step: {step_num}  Divergence: {diff:.6e}")
        if step_num < steps:
            root.after(10, update, step_num + 1)

    update()
    root.mainloop()


if __name__ == "__main__":
    run_animation()
