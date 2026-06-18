import numpy as np
import matplotlib.pyplot as plt


X_LEFT = 100
Y_TOP = 650

boundary_0 = [(145, 100), (450, 370), (450, 650)]
boundary_1 = [(105, 100), (150, 132.5), (400, 355), (400, 650)]


def lerp(a, b, t):
    """
    Linear interpolation between a and b.
    t = 0 gives a.
    t = 1 gives b.
    """
    return (1 - t) * a + t * b


def cap_point(boundary_points):
    """
    Return the bottom point of the final vertical cap segment.

    Example:
        [(145,100), (450,370), (450,650)]

    returns:
        (450,370)
    """

    p1 = boundary_points[-2]
    p2 = boundary_points[-1]

    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2:
        if y1 < y2:
            return p1
        else:
            return p2

    return boundary_points[-1]


def vmin_from_boundary(boundary_points, H):
    """
    Return Vmin at a given H for one known boundary.

    This includes:
    - the left horizontal connector from x = 100 to the first boundary point
    - the sloped boundary segments
    - the bottom of the final vertical cap segment
    """

    first_x, first_y = boundary_points[0]

    # Left horizontal connector
    if X_LEFT <= H <= first_x:
        return first_y

    # Boundary segments
    for p1, p2 in zip(boundary_points[:-1], boundary_points[1:]):
        x1, y1 = p1
        x2, y2 = p2

        # Vertical segment
        if x1 == x2:
            if np.isclose(H, x1):
                return min(y1, y2)
            continue

        xmin = min(x1, x2)
        xmax = max(x1, x2)

        if xmin <= H <= xmax:
            t = (H - x1) / (x2 - x1)
            return lerp(y1, y2, t)

    return None


def make_known_boundary_plot_points(boundary_points):
    """
    Return H and V arrays for plotting a known boundary lower envelope,
    including the left connector but excluding the vertical cap.
    """

    points = []

    first_x, first_y = boundary_points[0]

    if first_x > X_LEFT:
        points.append((X_LEFT, first_y))

    for p1, p2 in zip(boundary_points[:-1], boundary_points[1:]):
        x1, y1 = p1
        x2, y2 = p2

        points.append(p1)

        # Stop before plotting the vertical cap as part of the lower envelope
        if x1 == x2:
            break

    cap = cap_point(boundary_points)

    if points[-1] != cap:
        points.append(cap)

    H_values = [p[0] for p in points]
    V_values = [p[1] for p in points]

    return H_values, V_values


def make_phantom_boundary_between_0_and_1(angle):
    """
    Construct the phantom Vmin(H) boundary between 0° and 1°.

    angle must satisfy:
        0 <= angle <= 1

    Returns:
        H_phantom, V_phantom, Hcap_phantom, Vcap_phantom
    """

    t = angle

    cap0 = cap_point(boundary_0)
    cap1 = cap_point(boundary_1)

    Hcap0, Vcap0 = cap0
    Hcap1, Vcap1 = cap1

    # Interpolated cap point of the phantom boundary
    Hcap_phantom = lerp(Hcap0, Hcap1, t)
    Vcap_phantom = lerp(Vcap0, Vcap1, t)

    # Shared H-domain: both known boundaries exist here
    H_shared = np.linspace(X_LEFT, Hcap1, 400)

    V0_shared = np.array([
        vmin_from_boundary(boundary_0, H)
        for H in H_shared
    ])

    V1_shared = np.array([
        vmin_from_boundary(boundary_1, H)
        for H in H_shared
    ])

    V_phantom_shared = lerp(V0_shared, V1_shared, t)

    # Terminal cap-extension domain:
    # Boundary 1 no longer exists after H = Hcap1,
    # but the phantom boundary may continue until Hcap_phantom.
    if Hcap_phantom > Hcap1:
        H_A = Hcap1
        V_A = V_phantom_shared[-1]

        H_B = Hcap_phantom
        V_B = Vcap_phantom

        H_extension = np.linspace(H_A, H_B, 100)
        V_extension = V_A + (H_extension - H_A) * (V_B - V_A) / (H_B - H_A)

        # Avoid duplicating point A
        H_phantom = np.concatenate([H_shared, H_extension[1:]])
        V_phantom = np.concatenate([V_phantom_shared, V_extension[1:]])

    else:
        H_phantom = H_shared
        V_phantom = V_phantom_shared

    return H_phantom, V_phantom, Hcap_phantom, Vcap_phantom


def plot_boundaries(angle):
    """
    Plot boundaries 0°, 1°, and the phantom boundary for the requested angle.
    """

    H0, V0 = make_known_boundary_plot_points(boundary_0)
    H1, V1 = make_known_boundary_plot_points(boundary_1)

    H_ph, V_ph, Hcap_ph, Vcap_ph = make_phantom_boundary_between_0_and_1(angle)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Known lower envelopes
    ax.plot(H0, V0, label="Boundary 0°")
    ax.plot(H1, V1, label="Boundary 1°")

    # Known vertical caps
    Hcap0, Vcap0 = cap_point(boundary_0)
    Hcap1, Vcap1 = cap_point(boundary_1)

    ax.plot([Hcap0, Hcap0], [Vcap0, Y_TOP], linestyle="--")
    ax.plot([Hcap1, Hcap1], [Vcap1, Y_TOP], linestyle="--")

    # Phantom boundary lower envelope
    ax.plot(H_ph, V_ph, linewidth=3, label=f"Phantom boundary {angle:.3f}°")

    # Phantom vertical cap
    ax.plot([Hcap_ph, Hcap_ph], [Vcap_ph, Y_TOP], linestyle=":")

    # Mark phantom cap point
    ax.scatter([Hcap_ph], [Vcap_ph])
    ax.annotate(
        f"phantom cap\nH={Hcap_ph:.2f}, V={Vcap_ph:.2f}",
        (Hcap_ph, Vcap_ph),
        xytext=(10, 10),
        textcoords="offset points"
    )

    ax.set_title(f"Boundary interpolation between 0° and 1° | angle = {angle:.3f}°")
    ax.set_xlabel("H")
    ax.set_ylabel("V")
    ax.set_xlim(100, 460)
    ax.set_ylim(90, 670)
    ax.grid(True)
    ax.legend()

    plt.show()
    plt.savefig("undercuts/phantom.png")


def main():
    raw = input("Give angle between 0 and 1 degrees: ")
    angle = abs(float(raw))

    if angle < 0 or angle > 1:
        raise ValueError("This visualization script only accepts angles between 0 and 1 degrees.")

    plot_boundaries(angle)


if __name__ == "__main__":
    main()