import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class RayBundle:
    def __init__(self, koehler_setup, y0, col):
        self.koehler_setup = koehler_setup
        self.p0 = np.arcsin(np.pi/30)
        self.y0 = y0
        self.ax = koehler_setup.ax
        self.col = col

        ys1 = self.koehler_setup.get_ray_trajectory(self.y0,-self.p0)
        ys2 = self.koehler_setup.get_ray_trajectory(self.y0,0)
        ys3 = self.koehler_setup.get_ray_trajectory(self.y0,self.p0)

        self.ax_fill = self.ax.fill_between(self.koehler_setup.xs,ys1,ys3,color=col,alpha=0.1)
        self.ax_l1, = self.ax.plot(self.koehler_setup.xs,ys1,"k",linewidth=0.4)
        self.ax_l2, = self.ax.plot(self.koehler_setup.xs,ys2,"k",linewidth=0.4)
        self.ax_l3, = self.ax.plot(self.koehler_setup.xs,ys3,"k",linewidth=0.4)

    def update(self, cond_ap_opening, field_ap_opening):
        self.ax_fill.remove()

        y0 = self.y0*min(max(cond_ap_opening,0),1)
        p0 = self.p0*min(max(field_ap_opening,0),1)
        ys1 = self.koehler_setup.get_ray_trajectory(y0,-p0)
        ys2 = self.koehler_setup.get_ray_trajectory(y0,0)
        ys3 = self.koehler_setup.get_ray_trajectory(y0,p0)

        self.ax_fill = self.ax.fill_between(self.koehler_setup.xs,ys1,ys3,color=self.col,alpha=0.1)
        self.ax_l1.set_ydata(ys1)
        self.ax_l2.set_ydata(ys2)
        self.ax_l3.set_ydata(ys3)

class Aperture:
    def __init__(self, koehler_setup, x, height):
        self.koehler_setup = koehler_setup
        self.ax = koehler_setup.ax

        self.x = x
        self.dy = height
        self.dx = 0.15
        self.xs = np.array([-self.dx/2, -self.dx/2, 0])

        ys = np.array([0.45, self.dy/2+self.dx/4, self.dy/2])
        self.ax_fill1 = self.ax.fill_betweenx(ys, self.x-self.xs, self.x+self.xs, color="tab:blue")
        self.ax_fill2 = self.ax.fill_betweenx(-ys, self.x-self.xs, self.x+self.xs, color="tab:blue")

    def update(self, ap_opening):
        self.ax_fill1.remove()
        self.ax_fill2.remove()

        dy = self.dy*min(max(ap_opening,0),1)
        ys = np.array([0.45, dy/2+self.dx/4, dy/2])
        self.ax_fill1 = self.ax.fill_betweenx(ys, self.x-self.xs, self.x+self.xs, color="tab:blue")
        self.ax_fill2 = self.ax.fill_betweenx(-ys, self.x-self.xs, self.x+self.xs, color="tab:blue")


class KoehlerSetup:
    def __init__(self, ax, fig):
        self.f_col = 1
        self.f_cond = 1.5
        self.x_col = 1.5
        self.filament_length = 0.2

        self.x_field_ap = self.x_col + self.f_col
        self.x_cond_ap = self.x_col + 1/(1/self.f_col-1/self.x_col)
        self.x_cond = self.x_cond_ap+self.f_cond
        self.x_obj = self.x_cond + 1/(1/self.f_cond-1/(self.x_cond-self.x_field_ap))

        self.xs = [0, self.x_col, self.x_cond, self.x_obj]
        self.mag_col = self.x_cond_ap/self.x_col-1
        self.ax = ax
        self.fig = fig

    def _draw_filament(self, x, dx, dy, alpha):
        ys = np.linspace(-dy/2,dy/2,50)
        us = 2*ys/dy
        xs1 = dx/2 * np.exp(-us**6/(1.01-us**2))*(0.4*np.cos(6*np.pi*us)-1)
        xs2 = dx/2 * np.exp(-us**6/(1.01-us**2))*(0.4*np.cos(6*np.pi*us)+1)
        self.ax.fill_betweenx(ys,x+xs1,x+xs2,color="tab:red",alpha=alpha)

    def _draw_lens(self, x, dx, dy):
        ellipse = Ellipse(xy=(x,0),width=dx,height=dy,facecolor="tab:gray")
        self.ax.add_artist(ellipse)

    def get_ray_trajectory(self, y0, p0):
        ys = np.zeros((4,))
        ys[0] = y0
        p = p0
        ys[1] = ys[0]+self.x_col*p/np.sqrt(1-p**2)
        p -= ys[1]*np.sqrt(1-p**2)/self.f_col
        ys[2] = ys[1]+(self.x_cond-self.x_col)*p/np.sqrt(1-p**2)
        p -= ys[2]*np.sqrt(1-p**2)/self.f_cond
        ys[3] = ys[2]+(self.x_obj-self.x_cond)*p/np.sqrt(1-p**2)

        return ys

    def draw_system(self):
        self._draw_filament(0,0.1,self.filament_length,1)
        plt.text(0, 0.2, "Lamp\nfilament", color="tab:red",
            horizontalalignment='center', verticalalignment='center')

        self._draw_lens(self.x_col,0.2,0.55)
        plt.text(self.x_col, 0.35, "Lamp\ncollector",
            horizontalalignment='center', verticalalignment='center')

        self.field_ap = Aperture(self, self.x_field_ap, self.filament_length)
        plt.text(self.x_field_ap, 0.53, "Field\naperture", color="tab:blue",
            horizontalalignment='center', verticalalignment='center')

        self.cond_ap = Aperture(self, self.x_cond_ap, self.filament_length*self.mag_col)
        plt.text(self.x_cond_ap, 0.53, "Condenser\naperture", color="tab:blue",
            horizontalalignment='center', verticalalignment='center')

        self._draw_filament(self.x_cond_ap,0.1,self.filament_length*self.mag_col,0.4)
        self._draw_lens(self.x_cond,0.3,0.9)
        plt.text(self.x_cond, 0.51, "Condenser",
            horizontalalignment='center', verticalalignment='center')

        self.ax.plot(self.x_obj*np.ones((2,)), [-0.2,0.2], "k-.")
        plt.text(self.x_obj, 0.28, "Object\nplane",
            horizontalalignment='center', verticalalignment='center')

        self.ray_bundles = [ RayBundle(self, self.filament_length/2, "tab:blue")
                           , RayBundle(self, 0, "tab:green")
                           , RayBundle(self, -self.filament_length/2, "tab:purple") ]


    def update(self, cond_ap_opening, field_ap_opening):
        for i in range(0,3):
            self.ray_bundles[i].update(cond_ap_opening,field_ap_opening)
        self.field_ap.update(field_ap_opening)
        self.cond_ap.update(cond_ap_opening)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show()


def draw_koehler_setup(*,figsize):
    plt.rcParams['figure.figsize'] = figsize
    plt.ion()

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plt.tight_layout()

    ax.set_ylim(-0.5,0.6)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')
    
    koehler_setup = KoehlerSetup(ax,fig)
    koehler_setup.draw_system()
    fig.canvas.draw()
    
    return koehler_setup
