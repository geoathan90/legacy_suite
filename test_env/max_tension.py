import numpy as np
import matplotlib.pyplot as plt
import scripts.tensions as ts
import math
from matplotlib.ticker import MultipleLocator, FixedLocator

Tmax = 1000         # megisti tanysi kg
E = 19.33e9         # metro elastikotitas kg/m2
A = 9.6454e-5       # diatomi m2
alpha = 1.152e-5    # thermal expansion coefficient 1/C

arxiki_therm = -19     # C
w_ice = 1.835       # 2.267 (+k)
w_bare = 0.769

spans = np.linspace(100,400,1001)
tension_max = Tmax*np.ones_like(spans)

tension_0 = [ts.solve_for_H2_numeric(span, Tmax, E, A, alpha, arxiki_therm, 0, w_ice, w_bare) for span in spans]
tension_20 = [ts.solve_for_H2_numeric(span, Tmax, E, A, alpha, arxiki_therm, 20, w_ice, w_bare) for span in spans]
tension_40 = [ts.solve_for_H2_numeric(span, Tmax, E, A, alpha, arxiki_therm, 40, w_ice, w_bare) for span in spans]

sag_Tmax = ts.sag(spans, tension_max, w_bare)
sag_0 = ts.sag(spans, tension_0, w_bare)
sag_20 = ts.sag(spans, tension_20, w_bare)  
sag_40 = ts.sag(spans, tension_40, w_bare)

def align_dual_y_axes(
    ax_tension,
    ax_sag,
    max_tension_value,
    max_sag_value,
    sag_grid=0.2,
    tension_grid=25,
):
    """
        Aligns two y-axes so that:
        
            sag_grid on the sag axis
            corresponds visually to
            tension_grid on the tension axis.
        
        Example:
            sag_grid = 0.2 m
            tension_grid = 25 kg
            
        Then every horizontal gridline is meaningful on both axes.
    """

    # How many grid intervals are needed to contain the sag data?
    n_sag_steps = math.ceil(max_sag_value / sag_grid)

    # same for tension data
    n_tension_steps = math.ceil(max_tension_value / tension_grid)

    # Use the larger number so both datasets fit.
    n_steps = max(n_sag_steps, n_tension_steps)

    # Final aligned limits.
    sag_top = n_steps * sag_grid
    tension_top = n_steps * tension_grid

    ax_sag.set_ylim(0, sag_top)
    ax_tension.set_ylim(0, tension_top)

    ax_sag.yaxis.set_major_locator(MultipleLocator(sag_grid))
    ax_tension.yaxis.set_major_locator(MultipleLocator(tension_grid))

    return tension_top, sag_top

if __name__ == "__main__":
    fig, ax_tension = plt.subplots(figsize=(9, 5))

    # ------------------------------------------------------------
    # Left y-axis: tensions
    # ------------------------------------------------------------
    #ax_tension.plot(spans, tension_max, label="Tάνυση -19°C| 1/2\" πάγου 4#")
    ax_tension.plot(spans, tension_0, label="Tάνυση 0°C", color="tab:blue", linewidth=1)
    ax_tension.plot(spans, tension_20, label="Tάνυση 20°C", color="tab:orange", linewidth=1)
    ax_tension.plot(spans, tension_40, label="Tάνυση 40°C", color="tab:green", linewidth=1)

    ax_tension.set_xlabel("Άνοιγμα (m)")
    ax_tension.set_ylabel("Tάνυση (kg)")
    ax_tension.set_ylim(0, 2100)
    ax_tension.grid(True)

    # ------------------------------------------------------------
    # Right y-axis: sags
    # ------------------------------------------------------------
    ax_sag = ax_tension.twinx()

    #ax_sag.plot(spans, sag_Tmax, "--", label="Βέλος -19°C| 1/2\" πάγου 4#")
    ax_sag.plot(spans, sag_0, label="Βέλος 0°C", color="tab:blue", linewidth=1)
    ax_sag.plot(spans, sag_20, label="Βέλος 20°C", color="tab:orange", linewidth=1)
    ax_sag.plot(spans, sag_40, label="Βέλος 40°C", color="tab:green", linewidth=1)

    ax_sag.set_ylabel("Βέλος (m)")

    # ------------------------------------------------------------
    # Align the two y-axes
    # ------------------------------------------------------------
    max_tension_value = max(
        np.max(tension_0),
        np.max(tension_20),
        np.max(tension_40),
    )

    max_sag_value = max(
        np.max(sag_0),
        np.max(sag_20),
        np.max(sag_40),
    )

    align_dual_y_axes(
        ax_tension,
        ax_sag,
        max_tension_value=max_tension_value,
        max_sag_value=max_sag_value,
        sag_grid=1,
        tension_grid=50,
    )

    ax_tension.grid(False)
    ax_sag.grid(True, axis="y")

    # ------------------------------------------------------------
    # Tick and grid styling
    # ------------------------------------------------------------

    # Sag axis:
    # Show labeled ticks only every 5 m
    ax_sag.yaxis.set_major_locator(
        FixedLocator([0, 5, 10, 15, 20, 25, 30, 35 , 40])
    )

    # But draw horizontal gridlines every 0.5 m or 1.0 m
    ax_sag.yaxis.set_minor_locator(MultipleLocator(0.5))

    # Tension axis:
    # Show labeled ticks every 500 kg
    ax_tension.yaxis.set_major_locator(
        FixedLocator([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
    )

    # Optional: minor tension ticks every 100 kg, but no grid from them
    ax_tension.yaxis.set_minor_locator(MultipleLocator(100))

    # X axis:
    # Labeled ticks every 50 m
    ax_tension.xaxis.set_major_locator(MultipleLocator(50))

    # Extra vertical gridlines every 25 m
    ax_tension.xaxis.set_minor_locator(MultipleLocator(25))

    # Remove default grid from tension axis first
    ax_tension.grid(False)

    # Horizontal gridlines controlled by sag axis
    ax_sag.grid(True, axis="y", which="major", linewidth=0.8)
    ax_sag.grid(True, axis="y", which="minor", linewidth=0.3)

    # Vertical gridlines controlled by x-axis
    ax_tension.grid(True, axis="x", which="major")
    ax_tension.grid(True, axis="x", which="minor")

    # # Hide minor tick marks if they clutter the plot
    # ax_sag.tick_params(axis="y", which="minor", length=0)
    # ax_tension.tick_params(axis="y", which="minor", length=0)

    # ------------------------------------------------------------
    # Combined legend
    # ------------------------------------------------------------
    # lines_1, labels_1 = ax_tension.get_legend_handles_labels()
    # lines_2, labels_2 = ax_sag.get_legend_handles_labels()

    # ax_tension.legend(
    #     lines_1 + lines_2,
    #     labels_1 + labels_2,
    #     loc="center left",
    #     bbox_to_anchor=(1.10, 0.5),
    #     fontsize=8,
    #     framealpha=0.9
    # )

    plt.title(f"Μέγιστη τάνυση {Tmax} kg")
    plt.tight_layout()
    plt.savefig("tension_and_sag.png", dpi=200)

    print(f"sag 0 for 240 m span for Tmax = {Tmax} kg: {ts.sag(240, ts.solve_for_H2_numeric(240, Tmax, E, A, alpha, arxiki_th, 0, w_ice, w_bare), w_bare)} m")
    print(f"sag 20 for 240 m span for Tmax = {Tmax} kg: {ts.sag(240, ts.solve_for_H2_numeric(240, Tmax, E, A, alpha, arxiki_th, 20, w_ice, w_bare), w_bare)} m")
    print(f"sag 40 for 240 m span for Tmax = {Tmax} kg: {ts.sag(240, ts.solve_for_H2_numeric(240, Tmax, E, A, alpha, arxiki_th, 40, w_ice, w_bare), w_bare)} m")