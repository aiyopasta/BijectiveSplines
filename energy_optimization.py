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
            1. Iterate through all the vertices, and compute a gradient direction by:
            2. If it's a boundary vertex, use simple percentage-wise backtracking step.
            3. If it's an interior vertex, choose direction that minimizes difference between total area of
               all the quads that vertex affects in the target vs. current mesh.
'''

from tkinter import *
import numpy as np
import copy
import canvasvg

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
mesh_dims = np.array([deg+3, deg+3])  # Order is row, col. Sliding window will be of size deg+1 x deg+1 = d+2 x d+2
perturb = 40  # used for randomly generating the target mesh vertex positions

# Mesh Data structure (Stored as: row0col0, row0col1, row0col2, ....)
target_positions = np.zeros((int(mesh_dims[0] * mesh_dims[1]), 2))
current_positions = np.zeros((int(mesh_dims[0] * mesh_dims[1]), 2))

# Optimization Params
quad_idx = 0  # Index in the above arrays of top-left corner of next window to apply newton to. TODO: But we're not trying to optimize piece by piece.
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
                  text='Yellow=Target Mesh, Red=Current Mesh, Dark Red=Splice to be considered',
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
            vcolor = 'darkred'
            r = 2
        else:
            vcolor = 'red'
            r = 0
        x, y = target_positions[i]
        w.create_oval(x-radius, y-radius, x+radius, y+radius, outline='black', fill='yellow')
        x, y = current_positions[i]
        w.create_oval(x-radius-r, y-radius-r, x+radius+r, y+radius+r, outline='black', fill=vcolor)


# Draw cone associated with a particular quad with top-left corner given by the input index "quad".
def show_cone(quad):
    global current_positions, cone_radius, quad_idx
    temp_quad_idx = copy.copy(quad_idx)
    quad_idx = quad

    w.delete('all')
    redraw()
    # Draw boundary
    center_x, center_y = window_w/6,  window_h/2
    w.create_oval(center_x-cone_radius, center_y-cone_radius, center_x+cone_radius, center_y+cone_radius, outline='black')
    # Draw horizontal arrows
    splice = full_to_splice(current_positions)
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


# Percentage-wise back-tracking line search based on criterion. TODO: We'll use this only for the boundary vertices.
def simple_step():
    global target_positions, current_positions, mesh_dims, quad_idx, deg

    for i in range(len(target_positions)):
        v_target, v = target_positions[i], copy.copy(current_positions[i])
        # Move all the way
        delta = v - v_target
        alpha = 1.
        current_positions[i] = v - (alpha * delta)

        # Move "less all the way" if invalid
        transverse = False
        while alpha > 1e-6 and not transverse:
            first = True
            # Check if ALL windows have transverse. TODO: Though technically, we only gotta check a subset of subgrid this particular vertex affects right?
            transverse = True
            while first or quad_idx != 0:
                first = False
                splice = full_to_splice(current_positions)
                transverse = transverse and splice_transverse(splice)
                increment_quad()

            # If not all windows transverse, then change alpha and position
            if not transverse:
                print('Not all transverse for vertex', i, 'trying', alpha/2)
                alpha /= 2
                current_positions[i] = v - (alpha * delta)


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
        simple_step()
        redraw()


# Key bind
w.bind('<Key>', key_pressed)

# Necessary line for Tkinter
mainloop()