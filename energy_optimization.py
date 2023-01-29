'''
    Goal: Iteratively modify an input mesh so that we are guaranteed (based on cone transversability criterion)
          that the resulting control points and B-spline surface is bijective.

    Spline review: A degree d B-spline basis function is supported on interval of length d+1, from [a, a+d+1], meaning
                   it is supported on d+2 knots (i.e. a+0, a+1, a+2, ..., a+d, a+d+1). In 1D, there is 1 basis function
                   per control point of the B-spline surface. Additionally, every point in the domain of the total 1D
                   (padded) B-Spline curve lies in the support of d+1 basis functions. In 2D, this corresponds to
                   (d+1) * (d+1) control points, which in turn, from our method, correspond to (d+2) * (d+2) mesh
                   vertices which generated those.

    TODO: Modify the algorithm to be:
            1. Iterate through all the vertices, and compute a gradient direction:
            2. If it's a boundary vertex, use simple percentage-wise backtracking step.
            3. If it's an interior vertex, choose direction that minimizes difference between total area of
               all the quads that vertex affects in the target vs. current mesh.
'''

from tkinter import *
import numpy as np
import copy
import canvasvg
from functools import partial
from scipy.optimize import minimize

# Window size
n = 700
window_w = int(n*2)
window_h = int(n)

# Tkinter Setup
root = Tk()
root.title("Bijective Spline — Energy Optimization")
root.attributes("-topmost", True)
root.geometry(str(window_w) + "x" + str(window_h))  # window size hardcoded

w = Canvas(root, width=window_w, height=window_h)
w.focus_set()
w.configure(bg='white')
w.pack()

# Spline + Mesh Params
deg = 3  # this is actually one plus the degree of the spline (i.e. deg = d+1)
mesh_dims = np.array([deg+2, deg+2])  # Order is row, col. Sliding window will be of size deg+1 x deg+1 = d+2 x d+2
perturb = 30  # used for randomly generating the target mesh vertex positions

# Mesh Data structure (Stored as: row0col0, row0col1, row0col2, ....)
target_positions = np.zeros((int(mesh_dims[0] * mesh_dims[1]), 2))
current_positions = np.zeros((int(mesh_dims[0] * mesh_dims[1]), 2))

# Splice params
quad_idx = 0
cone_quad = 0

# Display Params
radius = 5  # for displaying vertices
edgelen = 70  # edge lengths of target mesh before perturbation
cone_radius = 200


# TODO: Is this needed anymore?
# Increment the top-left corner of sliding window, modulo the index set of the mesh vertices.
def increment_quad():
    global quad_idx, mesh_dims, deg
    if quad_idx % mesh_dims[1] < mesh_dims[1] - (deg+1):
        quad_idx += 1
    else:
        if mesh_dims[0] - (quad_idx // mesh_dims[1]) > deg + 1:
            quad_idx += mesh_dims[1] - (quad_idx % mesh_dims[1])
        else:
            quad_idx = 0


# Increment the top-left corner of the sliding cone-window, modulo the index set of the mesh vertices.
def increment_cone_quad():
    global cone_quad, mesh_dims, deg
    if cone_quad % mesh_dims[1] < mesh_dims[1] - (deg+1):
        cone_quad += 1
    else:
        if mesh_dims[0] - (cone_quad // mesh_dims[1]) > deg + 1:
            cone_quad += mesh_dims[1] - (cone_quad % mesh_dims[1])
        else:
            cone_quad = 0


# Given 1D index value, check whether it is in the sliding window given by top-left quad_idx corner.
def in_quad(idx):
    global quad_idx, mesh_dims
    col_check = 0 <= (idx % mesh_dims[1]) - (quad_idx % mesh_dims[1]) <= deg
    row_check = 0 <= (idx // mesh_dims[1]) - (quad_idx // mesh_dims[1]) <= deg
    return col_check and row_check


# Given either "target_positions" or "current_positions" list, we return 1D list of points
# within the window described by quad_idx.
def full_to_splice(points):
    global quad_idx, mesh_dims, deg
    splice = []
    idx = quad_idx
    while idx <= quad_idx + (deg * mesh_dims[1]) + deg:
        splice.append(points[idx])
        if (idx % mesh_dims[1]) - (quad_idx % mesh_dims[1]) < deg:
            idx += 1
        else:
            idx += mesh_dims[1] - deg

    return np.array(splice)


# TODO: This is even needed?
def splice_to_full(splice, full_points):
    global quad_idx, mesh_dims, deg
    idx = quad_idx
    i = 0
    while idx <= quad_idx + (deg * mesh_dims[1]) + deg:
        full_points[idx] = splice[i]
        if (idx % mesh_dims[1]) - (quad_idx % mesh_dims[1]) < deg:
            idx += 1
        else:
            idx += mesh_dims[1] - deg
        i += 1

    return full_points


# Given 1D list corresponding to a splice of the current_positions, check if it is transverse.
def splice_transverse(splice):
    global mesh_dims, deg
    sign = None
    for row1 in range(deg+1):
        for col1 in range(deg):
            # Get horizontal arrow
            idx_left, idx_right = ((deg + 1) * row1) + col1, ((deg + 1) * row1) + col1 + 1
            horiz = splice[idx_right] - splice[idx_left]; horiz /= np.linalg.norm(horiz)
            for row2 in range(1, deg+1):
                for col2 in range(deg+1):
                    # Get vertical arrow
                    idx_down, idx_up = ((deg + 1) * row2) + col2, ((deg + 1) * (row2 - 1)) + col2
                    vert = splice[idx_up] - splice[idx_down]; vert /= np.linalg.norm(vert)
                    # Compute cross product and perform check
                    cross_sign = np.sign(np.linalg.det([vert, horiz]))
                    if sign is None:
                        sign = cross_sign
                    elif sign != cross_sign:
                        return False

    return True


# Repaint method
def redraw():
    global target_positions, current_positions, mesh_dims, radius
    w.delete('all')

    # Rewrite text
    w.create_text(window_w / 2, window_h * 0.95,
                  font='AvenirNext 20',
                  text='Yellow=Target Mesh, Red=Current Mesh, Dark Red=Subgrid of current mesh, Orange=Subgrid of target mesh',
                  fill='black')

    # Draw mesh edges
    for i in range(len(target_positions)):
        if i % (mesh_dims[1]) != mesh_dims[1]-1:
            w.create_line(*target_positions[i], *target_positions[i+1], fill='black')
            w.create_line(*current_positions[i], *current_positions[i+1], fill='black', width=2)

        if i < len(target_positions) - mesh_dims[1]:
            w.create_line(*target_positions[i], *target_positions[i+mesh_dims[1]], fill='black')
            w.create_line(*current_positions[i], *current_positions[i+mesh_dims[1]], fill='black', width=2)

    # Draw mesh vertices
    vcolor = None
    r = 0
    for i in range(len(target_positions)):
        if in_quad(i):
            vcolor = 'orange'
            r = 0
        else:
            vcolor = 'yellow'
            r = 0
        x, y = target_positions[i]
        w.create_oval(x-radius-r, y-radius-r, x+radius+r, y+radius+r, outline='black', fill=vcolor)
        if in_quad(i):
            vcolor = 'darkred'
            r = 2
        else:
            vcolor = 'red'
            r = 0
        x, y = current_positions[i]
        w.create_oval(x-radius-r, y-radius-r, x+radius+r, y+radius+r, outline='black', fill=vcolor)


# Draw cone associated with a particular quad with top-left corner given by the input index "quad".
def show_cone(quad):
    global target_positions, current_positions, cone_radius, quad_idx
    temp_quad_idx = copy.copy(quad_idx)
    quad_idx = quad

    w.delete('all')
    redraw()

    positions = copy.copy(current_positions)
    center_x, center_y = window_w / 6, window_h / 2
    for i in range(2):
        if i == 1:
            positions = copy.copy(target_positions)
            center_x, center_y = window_w - (window_w / 6), window_h / 2

        # Draw labels
        w.create_text(center_x, center_y + cone_radius * 1.2,
                      font='AvenirNext 20',
                      text='Cone #'+str(quad)+' for ' + ('GENERATED' if i == 0 else 'TARGET') + ' Mesh',
                      fill='black')
        splice = full_to_splice(positions)
        transverse = splice_transverse(splice)
        w.create_text(center_x, center_y - cone_radius * 1.2,
                      font='AvenirNext 20',
                      text=('NOT ' if not transverse else '') + 'TRANSVERSE',
                      fill='red' if not transverse else 'black')
        # Draw boundary
        w.create_oval(center_x-cone_radius, center_y-cone_radius, center_x+cone_radius, center_y+cone_radius, outline='black')
        # Draw horizontal arrows
        for row in range(deg+1):
            for col in range(deg):
                idx_left, idx_right = ((deg + 1) * row) + col, ((deg + 1) * row) + col + 1
                horiz = splice[idx_right] - splice[idx_left]; horiz /= np.linalg.norm(horiz)
                w.create_line(center_x - (horiz[0] * cone_radius), center_y - (horiz[1] * cone_radius),
                              center_x + (horiz[0] * cone_radius), center_y + (horiz[1] * cone_radius), fill='red', width=2)

        # Draw vertical arrows
        for row in range(1, deg+1):
            for col in range(deg+1):
                # Get vertical arrow
                idx_down, idx_up = ((deg + 1) * row) + col, ((deg + 1) * (row - 1)) + col
                vert = splice[idx_up] - splice[idx_down]; vert /= np.linalg.norm(vert)
                w.create_line(center_x - (vert[0] * cone_radius), center_y - (vert[1] * cone_radius),
                              center_x + (vert[0] * cone_radius), center_y + (vert[1] * cone_radius), fill='blue')

    quad_idx = temp_quad_idx


# The actual area-based energy function to minimize
def energy(i, position):
    global target_positions, current_positions, mesh_dims
    n_cols = mesh_dims[1]
    # Iterate through top-left vertex indices of 4 quads in question
    cost = 0.
    for j in [i-n_cols-1, i-n_cols, i-1, i]:
        corner_idxs = [j, j+n_cols, j+n_cols+1, j+1]  # counter-clockwise direction
        area = 0.
        target_area = 0.
        for k in range(4):
            v1, v2 = current_positions[corner_idxs[k]], current_positions[corner_idxs[(k+1) % 4]]
            if corner_idxs[k] == i:
                v1 = position
            if corner_idxs[(k+1) % 4] == i:
                v2 = position
            area += np.linalg.det([v1, v2])  # shoelace theorem
            v1, v2 = target_positions[corner_idxs[k]], target_positions[corner_idxs[(k+1) % 4]]
            target_area += np.linalg.det([v1, v2])  # shoelace theorem
        area = 0.5 * abs(area)
        target_area = 0.5 * abs(target_area)
        cost += np.power(area - target_area, 2.0)

    return cost


# Energy minimization step (goes ONCE through every vertex)
def minimize_step():
    global target_positions, current_positions, mesh_dims, quad_idx, deg
    n_verts = len(target_positions)

    # We iterate to the minimum vertex by vertex
    for i in range(n_verts):
        # Current position
        v = copy.copy(current_positions[i])
        # Set a "target position" based on whether we're an interior or boundary vertex.
        # 1. If we're a boundary vertex, gradient direction is simply towards the target.
        v_target = None
        n_cols = mesh_dims[1]
        if i % n_cols in [0, n_cols-1] or i < n_cols or i > n_verts - n_cols:
            v_target = target_positions[i]

        # 2. If we're an interior vertex, gradient direction is energy minimizing one.
        else:
            cost_fn = partial(energy, i)
            result = minimize(cost_fn, x0=v)
            v_target = result.x
            print('vertex:', i, 'original cost:', cost_fn(v), 'minimized cost:', result.fun)

        # 3. Make the jump, and back-track line search as needed
        delta = v - v_target
        alpha = 1.
        current_positions[i] = v - (alpha * delta)
        transverse = False
        while not transverse:
            # Check if all splices have transverse (though clearly this is overkill)
            first = True
            transverse = True
            while first or quad_idx != 0:
                first = False
                splice = full_to_splice(current_positions)
                transverse = transverse and splice_transverse(splice)
                increment_quad()

            # If not all windows transverse, then change alpha and position
            if not transverse:
                print('Not all transverse for vertex', i, 'trying', alpha / 2)
                alpha /= 2
                current_positions[i] = v - (alpha * delta)

            # Breaking condition
            if alpha <= 1e-6:
                current_positions[i] = v  # revert to original
                print('Too many epochs, breaking line search.')
                break


# 1. Randomly generate target mesh vertex positions
mesh_mid = np.array([window_w/2, window_h/2])
width, height = (mesh_dims[1] - 1) * edgelen, (mesh_dims[0] - 1) * edgelen
for row in range(mesh_dims[0]):
    for col in range(mesh_dims[1]):
        x = mesh_mid[0] - (width/2) + (col * edgelen) + np.random.uniform(-perturb, perturb)
        y = mesh_mid[1] - (height/2) + (row * edgelen) + np.random.uniform(-perturb, perturb)
        idx = (mesh_dims[1] * row) + col
        target_positions[idx] = [x, y]

# 2. Initialize optimization-mesh's vertices to regular grid given by bounding box of randomly generated target mesh
max_x, min_x = max(target_positions[:, 0]), min(target_positions[:, 0])
max_y, min_y = max(target_positions[:, 1]), min(target_positions[:, 1])
edgelen = (abs(max_x - min_x) / (mesh_dims[1]-1), abs(max_y - min_y) / (mesh_dims[0]-1))  # tuple
for row in range(mesh_dims[0]):
    for col in range(mesh_dims[1]):
        x = min_x + (col * edgelen[0])
        y = min_y + (row * edgelen[1])
        idx = int((mesh_dims[1] * row) + col)
        current_positions[idx] = [x, y]

# 3. Draw initial mesh configuration
redraw()


# Key controls
img_num = 0
def key_pressed(event):
    global cone_quad, img_num

    # For sliding through splices to view its cone
    if event.char == 'd':
        print('Displaying cone:', cone_quad)
        show_cone(cone_quad)
        increment_cone_quad()

    # For saving
    elif event.char == 's':
        canvasvg.saveall('mesh_image_'+str(img_num)+'.svg', w, items=None, margin=10, tounicode=None)
        print('Saved image.')
        img_num += 1

    # For going to next optimization step
    elif event.char == ' ':
        print('Running algorithm again...')
        minimize_step()
        redraw()


# Key bind
w.bind('<Key>', key_pressed)

# Necessary line for Tkinter
mainloop()