from matplotlib import pyplot as plt
from catenary import catenary, catenary_points, catenary_low_point, catenary_y
from line import line_y, line_between_points
from sag import catenary_sag_vertical
from th_solver import Th_for_target_sag



def catenary_plot(cat, n=200):
    """
    Plot the catenary and its two suspension points.
    """
    x, y = catenary_points(cat, n=n)

    x1, y1 = cat["A"]
    x2, y2 = cat["B"]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="Catenary")
    plt.plot([x1, x2], [y1, y2], "o", label="Supports")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D Catenary")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()

def catenary_plot_full(cat, line, n=1000):
    """
    Plot:
    - the catenary
    - the straight line (chord) between A and B
    - the support points
    - the lowest point of the catenary
    - the point of maximum vertical sag
    """

    # Sample the catenary
    x, y = catenary_points(cat, n=n)

    # Support points
    A = cat["A"]
    B = cat["B"]
    x1, y1 = A
    x2, y2 = B

    # Lowest point of the catenary
    x_low = catenary_low_point(cat)
    y_low = catenary_y(cat, x_low)

    # Maximum vertical sag point
    sag_info = catenary_sag_vertical(cat, line)
    x_sag = sag_info["x_sag"]
    y_sag_curve = sag_info["y_curve"]
    y_sag_line = sag_info["y_line"]

    # Build chord line values for plotting
    y_chord = line_y(line, x)

    plt.figure(figsize=(9, 6))

    # Catenary
    plt.plot(x, y, label="Catenary")

    # Chord AB
    plt.plot(x, y_chord, "--", label="Chord AB")

    # Support points
    plt.plot([x1, x2], [y1, y2], "o", label="Supports")

    # Lowest point
    plt.plot(x_low, y_low, "o", label="Lowest point")

    # Max sag point on conductor
    plt.plot(x_sag, y_sag_curve, "o", label="Max vertical sag point")

    # Corresponding point on chord
    plt.plot(x_sag, y_sag_line, "o", label="Chord point above max sag")

    # Vertical segment showing the sag
    plt.plot([x_sag, x_sag], [y_sag_curve, y_sag_line], "--", label="Vertical sag")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Catenary with chord, low point, and max vertical sag")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()    

def catenary_plot_with_insulators(
    cat,
    A,
    B,
    C,
    D,
    n=1000,
    show_actual=True,
    show_supports=True,
    show_attachments=True,
    show_chord=False,
    show_low_point=False,
    show_sag=False,
    show_idealized=False,
    show_idealized_sag=True,
    annotate_tensions=True,
    only_idealized=False,
    w=None,
    idealized_sag=None,
    title="Catenary with insulators",
):
    """
    Plot:
    - the actual conductor catenary between C and D
    - the insulators as straight segments A-C and B-D

    Optional extras:
    - support / attachment points
    - chord AB
    - lowest point of the actual catenary
    - maximum vertical sag of the actual catenary relative to chord AB
    - idealized no-insulator catenary between A and B
    - maximum vertical sag of the idealized catenary relative to chord AB
    - annotation box with Th values

    Special option:
    - if only_idealized=True, the actual system is hidden and only the
      idealized catenary view is shown
    """

    if only_idealized:
        show_actual = False
        show_attachments = False
        show_low_point = False
        show_sag = False
        show_idealized = True

    plt.figure(figsize=(9, 6))

    # actual conductor
    if show_actual:
        x, y = catenary_points(cat, n=n)
        plt.plot(x, y, label="Actual catenary")

        # insulators
        plt.plot([A[0], C[0]], [A[1], C[1]], label="Left insulator")
        plt.plot([B[0], D[0]], [B[1], D[1]], label="Right insulator")

    # support points
    if show_supports:
        plt.plot([A[0], B[0]], [A[1], B[1]], "o", label="Main supports")

    # attachment points
    if show_attachments and show_actual:
        plt.plot([C[0], D[0]], [C[1], D[1]], "o", label="Conductor attachments")

    # chord AB
    if show_chord:
        plt.plot([A[0], B[0]], [A[1], B[1]], "--", label="Chord AB")

    # lowest point of actual catenary
    if show_low_point and show_actual:
        x_low = catenary_low_point(cat)
        y_low = catenary_y(cat, x_low)
        plt.plot(x_low, y_low, "o", label="Lowest point (actual)")

    line_AB = None
    sag_info_actual = None
    sag_info_ideal = None
    Th_ideal = None
    cat_ideal = None

    # We may still need actual sag internally even when actual curve is hidden,
    # because it defines the matching idealized catenary.
    if show_sag or show_idealized or annotate_tensions:
        line_AB = line_between_points(A, B)
        sag_info_actual = catenary_sag_vertical(cat, line_AB)

    # actual sag
    if show_sag and show_actual:
        x_sag = sag_info_actual["x_sag"]
        y_curve = sag_info_actual["y_curve"]
        y_line = sag_info_actual["y_line"]

        plt.plot(x_sag, y_curve, "o", label="Max sag point (actual)")
        plt.plot(x_sag, y_line, "o", label="Chord point above actual max sag")
        plt.plot([x_sag, x_sag], [y_curve, y_line], "--", label="Vertical sag (actual)")

    # idealized catenary
    if show_idealized:
        if w is None:
            raise ValueError("w must be provided when show_idealized=True.")

        if idealized_sag is None:
            idealized_sag = sag_info_actual["sag_max"]

        Th_ideal = Th_for_target_sag(A, B, w, idealized_sag)
        cat_ideal = catenary(A, B, w, Th_ideal)

        x_ideal, y_ideal = catenary_points(cat_ideal, n=n)
        plt.plot(x_ideal, y_ideal, "--", label="Idealized catenary (no insulators)")

        if show_idealized_sag:
            sag_info_ideal = catenary_sag_vertical(cat_ideal, line_AB)

            x_sag_ideal = sag_info_ideal["x_sag"]
            y_curve_ideal = sag_info_ideal["y_curve"]
            y_line_ideal = sag_info_ideal["y_line"]

            plt.plot(x_sag_ideal, y_curve_ideal, "o", label="Max sag point (idealized)")
            plt.plot(
                x_sag_ideal,
                y_line_ideal,
                "o",
                label="Chord point above idealized max sag",
            )
            plt.plot(
                [x_sag_ideal, x_sag_ideal],
                [y_curve_ideal, y_line_ideal],
                "--",
                label="Vertical sag (idealized)",
            )

    # annotate tensions
    if annotate_tensions:
        text_lines = []

        if show_actual and "Th" in cat:
            text_lines.append(f"Th actual = {cat['Th']:.4f}")

        if Th_ideal is not None:
            text_lines.append(f"Th idealized = {Th_ideal:.4f}")

        if show_actual and sag_info_actual is not None:
            text_lines.append(f"Sag actual = {sag_info_actual['sag_max']:.4f}")

        if sag_info_ideal is not None:
            text_lines.append(f"Sag idealized = {sag_info_ideal['sag_max']:.4f}")

        if text_lines:
            plt.text(
                0.02,
                0.98,
                "\n".join(text_lines),
                transform=plt.gca().transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()