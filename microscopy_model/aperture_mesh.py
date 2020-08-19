import numpy as np
import matplotlib.pyplot as plt

class DoubleArrow:
    def __init__(self, ax_sys, x_start, x_end, y_start, y_end, label):
        self.ax_sys = ax_sys

        xm = (x_end+x_start)/2
        dx = x_end-x_start
        ym = (y_end+y_start)/2
        dy = y_end-y_start
        self.opts = {"width":0.003, "head_width":0.04, "head_length":0.08,
                "length_includes_head":True, "zorder":3}
        self.ax_arr1 = self.ax_sys.arrow(xm, ym, -dx/2, -dy/2, **self.opts)
        self.ax_arr2 = self.ax_sys.arrow(xm, ym, dx/2, dy/2, **self.opts)
        self.ax_text = self.ax_sys.text(xm, ym, label,
            backgroundcolor="white", horizontalalignment='center', verticalalignment='center')

    def update(self, x_start, x_end, y_start, y_end):
        self.ax_arr1.remove()
        self.ax_arr2.remove()

        xm = (x_end+x_start)/2
        dx = x_end-x_start
        ym = (y_end+y_start)/2
        dy = y_end-y_start
        self.ax_arr1 = self.ax_sys.arrow(xm, ym, -dx/2, -dy/2, **self.opts)
        self.ax_arr2 = self.ax_sys.arrow(xm, ym, dx/2, dy/2, **self.opts)
        self.ax_text.set_position((xm,ym))

class ApertureMesh:
    def __init__(self,ax,fig):
        self.max_NA = 0.4
        self.Nr = 4
        self.update_wavevectors(self.max_NA)

        self.ax = ax
        self.fig = fig

    def update_wavevectors(self, NA):
        self.wavevectors = np.zeros((1+3*self.Nr*(self.Nr-1),2))
        for ir in range(1,self.Nr):
            beta = ir*NA/(self.Nr-1)
            for iphi in range(0,6*ir):
                phi = iphi*np.pi/(3*ir)
                self.wavevectors[1+3*ir*(ir-1)+iphi,0] = beta*np.cos(phi)
                self.wavevectors[1+3*ir*(ir-1)+iphi,1] = beta*np.sin(phi)

    def draw_mesh(self):
        self.ax_lines,self.ax_pts = self.ax.triplot(self.wavevectors[:,0],self.wavevectors[:,1],"ko-")
        self.arrow  = DoubleArrow(
            self.ax, -self.max_NA, self.max_NA, self.max_NA*1.2, self.max_NA*1.2, "2NA")

    def update(self, Nr, NA):
        self.Nr = Nr
        self.update_wavevectors(NA)
        self.ax_lines.remove()
        self.ax_pts.remove()
        self.ax_lines,self.ax_pts = self.ax.triplot(self.wavevectors[:,0],self.wavevectors[:,1],"ko-")
        self.arrow.update(-NA, NA, self.max_NA*1.2, self.max_NA*1.2)
        self.fig.canvas.draw()

def draw_aperture_mesh(*,figsize):
    plt.rcParams['figure.figsize'] = figsize
    plt.ion()

    fig = plt.figure(2)
    ax = fig.add_subplot(111)

    aperture_mesh = ApertureMesh(ax,fig)
    aperture_mesh.draw_mesh()
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.6)
    ax.set_xlabel("kx/k0")
    ax.set_ylabel("ky/k0")
    ax.set_aspect('equal')

    fig.canvas.draw()

    return aperture_mesh
