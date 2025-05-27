#Importing the necessary libraries
import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt

#Defining the MobiusStrip Class
class MobiusStrip:
    #Defining a constructor that is accepting Radius, Width and Resolution
    def __init__(self, R=1.0, w=0.3, resolution=200):
        self.R = R
        self.w = w
        self.n = resolution
        self.u, self.v = np.meshgrid(
            np.linspace(0, 2 * np.pi, self.n),
            np.linspace(-w / 2, w / 2, self.n)
        )
        self.x, self.y, self.z = self._generate_mesh()
    #Defining the 3D mesh on the surface using the parametri equations
    def _generate_mesh(self):
        u, v, R = self.u, self.v, self.R
        x = (R + v * np.cos(u / 2)) * np.cos(u)
        y = (R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        return x, y, z
    #Defining the function for computing the surface area
    def compute_surface_area(self):
        def integrand(v, u):
            dx_du = np.array([
                -np.sin(u) * (self.R + v * np.cos(u / 2)) - 0.5 * v * np.sin(u / 2) * np.cos(u),
                np.cos(u) * (self.R + v * np.cos(u / 2)) - 0.5 * v * np.sin(u / 2) * np.sin(u),
                0.5 * v * np.cos(u / 2)
            ])
            dx_dv = np.array([
                np.cos(u) * np.cos(u / 2),
                np.sin(u) * np.cos(u / 2),
                np.sin(u / 2)
            ])
            return np.linalg.norm(np.cross(dx_du, dx_dv))

        area, _ = dblquad(
            integrand,
            0, 2 * np.pi,
            lambda _: -self.w / 2,
            lambda _: self.w / 2
        )
        return area
    #Defining the function for calculating the edge length
    def compute_edge_length(self):
        u_vals = np.linspace(0, 2 * np.pi, self.n)
        length = 0
        for v_edge in [-self.w / 2, self.w / 2]:
            x = (self.R + v_edge * np.cos(u_vals / 2)) * np.cos(u_vals)
            y = (self.R + v_edge * np.cos(u_vals / 2)) * np.sin(u_vals)
            z = v_edge * np.sin(u_vals / 2)
            points = np.column_stack((x, y, z))
            length += np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        return length
    #Defing a function for visualizing the plot
    def plot(self):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.x, self.y, self.z, color='royalblue', edgecolor='k', alpha=0.8)
        ax.set_title("Mobius Strip - 3D Parametric Model", fontsize=14)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    mobius = MobiusStrip(R=1.0, w=0.3, resolution=200)
    print(f"Surface Area ≈ {mobius.compute_surface_area():.5f}")
    print(f"Edge Length  ≈ {mobius.compute_edge_length():.5f}")
    mobius.plot()
