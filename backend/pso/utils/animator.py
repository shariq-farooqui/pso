import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

from pso import fitness


class Animator:
    """A class used to create animations for PSO optimization runs.

    Attributes
    ----------
    documents : list
        A list of documents used in the PSO optimization run.
    run_id : str
        A unique identifier for the PSO optimization run.
    animation_type : str
        The type of animation to create. Can be "1d", "2d", "Contour", or "PCA".
    fixed_values : list[float] | None
        A list of fixed values to use in the animation. Defaults to None.
    filepath : str
        The filepath to save the animation as an mp4 file.
    design_index : int
        The design index to use for custom color maps and markers.

    Methods
    -------
    set_custom_cmaps()
        Sets custom color maps and markers based on the design index.
    animate()
        Creates and saves the animation based on the animation type.
    animate_nd()
        Creates and saves an animation for n-dimensional data.
    animate_nd_contour()
        Creates and saves a contour plot animation for n-dimensional data.
    animate_nd_pca()
        Creates and saves a PCA plot animation for n-dimensional data.
    animate_1d()
        Creates and saves a 1-dimensional plot animation.
    animate_2d()
        Creates and saves a 2-dimensional plot animation.
    """

    def __init__(self,
                 documents: list,
                 run_id: str,
                 animation_type: str,
                 design_index: int,
                 fixed_values: list[float] | None = None):
        self.documents = documents
        self.run_id = run_id
        self.animation_type = animation_type
        self.fixed_values = fixed_values if fixed_values else []
        self.filepath = os.path.join(f"/pso_media/{self.run_id}.mp4")
        self.design_index = design_index
        self.set_custom_cmaps()

    def set_custom_cmaps(self):
        """Sets custom color maps and markers based on the design index."""
        colours = [
            ["#f0fcf5", "#a3d9cb", "#1abc9c", "#13876d", "#0b5345"],  # Green
            ["#f5f1e6", "#f0c5ad", "#d89b7c", "#a5644e", "#5a3825"],  # Brown
            ["#f0f8ff", "#b0e0e6", "#7fb3d5", "#2980b9", "#1b2631"],  # Blue
            ["#f5f5f5", "#abb2b9", "#566573", "#2c3e50", "#1c2833"],  # Grey
        ]
        markers = [
            {
                "color": "#FF8256",
                "marker": "o",
            },
            {
                "color": "#8DAC97",
                "marker": "o",
            },
            {
                "color": "#F28D58",
                "marker": "o",
            },
            {
                "color": "#8F5527",
                "marker": "o",
            },
        ]
        selected_colours = colours[self.design_index]
        selected_markers = markers[self.design_index]

        self.cmap = LinearSegmentedColormap.from_list(f"custom_{self.design_index}", selected_colours, N=100)
        self.marker_style = selected_markers

    def animate(self):
        """Creates and saves the animation based on the animation type.

        Returns
        -------
        str
            The filepath to the saved mp4 file.
        """
        if self.animation_type == "1d":
            self.animate_1d()
        elif self.animation_type == "2d":
            self.animate_2d()
        else:
            self.animate_nd()
        return self.filepath

    def animate_nd(self):
        """Creates and saves an animation for n-dimensional data."""
        if self.animation_type == "Contour":
            self.animate_nd_contour()
        elif self.animation_type == "PCA":
            self.animate_nd_pca()

    def animate_1d(self):
        """Creates and saves a 1-dimensional plot animation."""

        objective_function = getattr(fitness, self.documents[0]["settings"]["objective_function"])
        bounds = self.documents[0]["settings"]["bounds"]
        x_min, x_max = bounds[0]

        fig, ax = plt.subplots()
        ax.set_xlim(x_min, x_max)

        line_color = self.cmap(0.7)

        x_values = np.linspace(x_min, x_max, 400)
        y_values = np.array([objective_function(np.array([xi])) for xi in x_values])
        ax.plot(x_values, y_values, color=line_color)

        particles_line, = plt.plot([], [], "o", color=self.marker_style["color"], linestyle="")
        iteration_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

        plt.xlabel("Position")
        plt.ylabel("Objective Function Value")
        plt.title("1-D Particle Swarm Optimization")

        def init():
            return particles_line, iteration_text

        def update(i):
            iteration_document = self.documents[i]
            particle_positions = [particle["position"][0] for particle in iteration_document["particles"]]
            particle_values = [objective_function(np.array([pos])) for pos in particle_positions]

            particles_line.set_data(particle_positions, particle_values)
            iteration_text.set_text(f"Iteration: {iteration_document['iteration'] + 1}")
            return particles_line, iteration_text

        ani = FuncAnimation(fig, update, frames=len(self.documents), init_func=init, blit=True)
        ani.save(self.filepath, writer="ffmpeg")

    def animate_2d(self):
        """Creates and saves a 2-dimensional plot animation."""

        objective_function = getattr(fitness, self.documents[0]["settings"]["objective_function"])
        bounds = self.documents[0]["settings"]["bounds"]
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        num_points = 100
        X = np.linspace(x_min, x_max, num_points)
        Y = np.linspace(y_min, y_max, num_points)
        X, Y = np.meshgrid(X, Y)
        Z = np.array([objective_function(np.array([xi, yi]))
                      for xi, yi in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        z_min, z_max = np.min(Z), np.max(Z)
        ax.set_zlim(z_min, z_max * 1.1)

        surface_plot = [ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.6, cmap=self.cmap)]

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Objective Function Value")
        ax.set_title("2-D Particle Swarm Optimization")

        particles_scatter = [
            ax.scatter([], [], [], s=100, alpha=1.0, edgecolors="black", linewidths=0.5, **self.marker_style),
        ]
        iteration_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

        def init():
            surface_plot[0].remove()
            surface_plot[0] = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.4, cmap=self.cmap)
            return surface_plot[0], particles_scatter[0], iteration_text

        def update(i):
            iteration_document = self.documents[i]
            particle_positions = np.array([particle["position"] for particle in iteration_document["particles"]])
            particle_values = [objective_function(pos) for pos in particle_positions]

            particles_scatter[0]._offsets3d = (particle_positions[:, 0], particle_positions[:, 1], particle_values)
            iteration_text.set_text(f"Iteration: {iteration_document['iteration'] + 1}")

            return particles_scatter[0], iteration_text

        ani = FuncAnimation(fig, update, frames=len(self.documents), init_func=init, blit=False)
        ani.save(self.filepath, writer="ffmpeg")

    def animate_nd_contour(self):
        """Creates and saves a contour plot animation for n-dimensional data."""

        objective_function = getattr(fitness, self.documents[0]["settings"]["objective_function"])
        bounds = self.documents[0]["settings"]["bounds"]
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        dim1, dim2 = 0, 1

        fig, ax = plt.subplots()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        x_range = np.linspace(x_min, x_max, 100)
        y_range = np.linspace(y_min, y_max, 100)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        z_grid = np.zeros_like(x_grid)

        for i in range(x_grid.shape[0]):
            for j in range(x_grid.shape[1]):
                position = np.zeros(self.documents[0]["settings"]["dimensions"])
                position[dim1] = x_grid[i, j]
                position[dim2] = y_grid[i, j]
                for k, fixed_value in enumerate(self.fixed_values, start=2):
                    if k < len(position):
                        position[k] = fixed_value
                z_grid[i, j] = objective_function(position)

        contour = plt.contourf(x_grid, y_grid, z_grid, levels=10, cmap=self.cmap)
        particles_scatter = []
        iteration_text = ax.text(0.02, 0.05, "", transform=ax.transAxes)
        plt.colorbar(contour, label="Objective Function Value")
        plt.legend(loc="upper right")

        def init():
            if particles_scatter:
                particles_scatter[0].remove()
            particles_scatter.append(ax.scatter([], [], **self.marker_style))
            return particles_scatter[0], iteration_text

        def update(i):
            iteration_document = self.documents[i]
            particle_positions = np.array([particle["position"] for particle in iteration_document["particles"]])
            if particles_scatter:
                particles_scatter[0].remove()
            particles_scatter[0] = ax.scatter(particle_positions[:, dim1], particle_positions[:, dim2],
                                              **self.marker_style)
            iteration_text.set_text(f"Iteration: {iteration_document['iteration'] + 1}")
            return particles_scatter[0], iteration_text

        ani = FuncAnimation(fig, update, frames=len(self.documents), init_func=init, blit=False)
        legend_particle = plt.Line2D([0], [0],
                                     marker=self.marker_style["marker"],
                                     color="w",
                                     markerfacecolor=self.marker_style["color"],
                                     markersize=10,
                                     label="Particles")
        plt.legend(handles=[legend_particle], loc="upper right")
        ani.save(self.filepath, writer="ffmpeg")

    def animate_nd_pca(self):
        """Creates and saves a PCA plot animation for n-dimensional data."""

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        all_positions = [particle["position"] for doc in self.documents for particle in doc["particles"]]
        pca = PCA(n_components=3)
        transformed_positions = pca.fit_transform(all_positions)

        x_min, x_max = transformed_positions[:, 0].min(), transformed_positions[:, 0].max()
        y_min, y_max = transformed_positions[:, 1].min(), transformed_positions[:, 1].max()
        z_min, z_max = transformed_positions[:, 2].min(), transformed_positions[:, 2].max()

        def init():
            ax.scatter([], [], [], **self.marker_style)
            return ax

        def update(i):
            ax.clear()
            iteration_document = self.documents[i]
            particle_positions = np.array([particle["position"] for particle in iteration_document["particles"]])
            transformed_positions = pca.transform(particle_positions)
            ax.scatter(transformed_positions[:, 0], transformed_positions[:, 1], transformed_positions[:, 2],
                       **self.marker_style)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.set_title(f"Iteration: {iteration_document['iteration'] + 1}")
            return ax

        ani = FuncAnimation(fig, update, frames=len(self.documents), init_func=init, blit=False)
        ani.save(self.filepath, writer="ffmpeg")
