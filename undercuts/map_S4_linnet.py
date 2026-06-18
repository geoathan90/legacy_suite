X_LEFT = 100
Y_TOP = 650

boundaries = {
    0: [(145, 100), (450, 370), (450, 650)],
    1: [(105, 100), (150, 132.5), (400, 355), (400, 650)],
    2: [(100, 147.5), (282.5, 280), (350, 340), (350, 650)],
    3: [(100, 200), (300, 345), (300, 650)],
    4: [(100, 250), (250, 360), (250, 650)],
    5: [(100, 302.5), (200, 375), (200, 650)],
    6: [(100, 355), (150, 390), (150, 650)],
}


################################

def make_polygon(boundary_points):
    """
        Convert one boundary polyline into a closed polygon.

        The polygon represents the area to the left/up-left of the boundary,
        closed using x = 100 and y = 650.
    """

    polygon = []

    first_x, first_y = boundary_points[0]

    # If the boundary starts away from x = 100,
    # add a horizontal connector from the left chart edge.
    if first_x > X_LEFT:
        polygon.append((X_LEFT, first_y))

    # Add the actual boundary points
    polygon.extend(boundary_points)

    last_x, last_y = boundary_points[-1]

    # Add top-left corner to close along y = 650
    if last_x > X_LEFT:
        polygon.append((X_LEFT, Y_TOP))

    return polygon


# make the polygons. Essentially an automation to produce the dictionary 
# that is commented out just below

polygons = {
    boundary_id: make_polygon(points)
    for boundary_id, points in boundaries.items()
}

# polygons ={
#     0: [(100, 100), (145, 100), (450, 370), (450, 650), (100, 650)], 
#     1: [(100, 100), (105, 100), (150, 132.5), (400, 355), (400, 650), (100, 650)], 
#     2: [(100, 147.5), (282.5, 280), (350, 340), (350, 650), (100, 650)], 
#     3: [(100, 200), (300, 345), (300, 650), (100, 650)], 
#     4: [(100, 250), (250, 360), (250, 650), (100, 650)], 
#     5: [(100, 302.5), (200, 375), (200, 650), (100, 650)],
#     6: [(100, 355), (150, 390), (150, 650), (100, 650)]
# }


####################################################


def point_on_segment(px, py, ax, ay, bx, by, tol=1e-9):
    """
        Return True if point P lies on segment AB.
    """

    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)

    if abs(cross) > tol:
        return False

    dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)

    if dot < -tol:
        return False

    length_sq = (bx - ax) ** 2 + (by - ay) ** 2

    if dot > length_sq + tol:
        return False

    return True

def point_in_polygon(x, y, polygon):
    """
        Return True if point (x, y) is inside polygon.

        Points exactly on polygon edges are treated as inside.
    """

    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Boundary check
        if point_on_segment(x, y, x1, y1, x2, y2):
            return True

        # Ray-casting test
        crosses = (y1 > y) != (y2 > y)

        if crosses:
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)

            if x < x_cross:
                inside = not inside

    return inside

def containing_boundary_polygons(x, y):
    """
        Return the boundary polygons that contain point (x, y).
    """

    containing = []

    for boundary_id, polygon in polygons.items():
        if point_in_polygon(x, y, polygon):
            containing.append(boundary_id)

    return containing

def count_containing_boundary_polygons(x, y):
    """
        Return how many boundary polygons contain point (x, y).
    """

    return len(containing_boundary_polygons(x, y))

########################

def boundary_y_at_x(boundary_points, x):
    """
        Return the y-value where one boundary crosses a given x-value.

        Vertical segments are ignored, because for x = constant they do not
        correspond to one unique y-value.
    """

    # Iterate through each segment of a chosen boundary multiline
    # eg boundary_points for the 3rd region corresponds to boundaries[2]
    for p1, p2 in zip(boundary_points[:-1], boundary_points[1:]):
        x1, y1 = p1
        x2, y2 = p2

        # Ignore vertical segments
        if x1 == x2:
            continue

        # Check whether x lies inside this segment's x-range
        xmin = min(x1, x2)
        xmax = max(x1, x2)

        # If it does, compute the corresponding y-value using 
        # linear interpolation on the appropriate straight line segment.,
        if xmin <= x <= xmax:
            t = (x - x1) / (x2 - x1)  # find how far along x is on the segment
            y = y1 + t * (y2 - y1)    # linear adjustment for y based on t
            return y

    return None


def all_boundary_y_values_at_x(x):
    """
        Return all valid boundary y-values for a given x.
    """

    results = []

    for boundary_id, points in boundaries.items():
        y = boundary_y_at_x(points, x)

        if y is not None:
            results.append((boundary_id, y))

    return results



def main():

    x, y = 300, 150

    inside = containing_boundary_polygons(x, y)
    crossings = all_boundary_y_values_at_x(x)
    formatted_crossings = [(i, round(y, 2)) for i, y in crossings]
    
    print("Inside polygons:", inside)
    print(f"Boundary crossings at x = {x:.2f}: {formatted_crossings}")

if __name__ == "__main__":
    main()