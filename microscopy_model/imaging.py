import pyfftw 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class OpticalField:
    def __init__(self, **kwargs):
        if len(kwargs["mesh_dimensions"])!=2:
            raise Exception("mesh_dimensions should be an array-like object of size 2")
        if len(kwargs["mesh_lengths"])!=2:
            raise Exception("mesh_lengths should be an array-like object of size 2")

        self._wavelength = 0.5
        self._max_NA_objective = 0.5
        self._cur_NA = self._max_NA_objective

        dims = np.array(kwargs["mesh_dimensions"])
        lengths = np.array(kwargs["mesh_lengths"])

        (self._Nx, self._Ny) = tuple(int(dim) for dim in dims)
        (self._Lx, self._Ly) = tuple(lengths)
        (self._dx, self._dy) = tuple(lengths/np.maximum(np.ones(2),dims-1))

        self._vals = pyfftw.empty_aligned(
            (dims[1],dims[0]), dtype="complex128")
        self._fft_vals = pyfftw.empty_aligned(
            (dims[1],dims[0]), dtype="complex128")
        self._focused_vals = pyfftw.empty_aligned(
            (dims[1],dims[0]), dtype="complex128")
        self._fft_plan = pyfftw.FFTW(
            self._vals, self._fft_vals, axes=(0,1))
        self._ifft_plan = pyfftw.FFTW(
            self._fft_vals, self._focused_vals, axes=(0,1), direction="FFTW_BACKWARD")

        IY, IX = np.meshgrid(range(0,self._Ny), range(0,self._Nx), indexing="ij")
        kx = -np.abs(2*np.pi/self._Lx*(IX-0.5*self._Nx)) + np.pi*self._Nx/self._Lx
        ky = -np.abs(2*np.pi/self._Ly*(IY-0.5*self._Ny)) + np.pi*self._Ny/self._Ly
        k0 = np.tile(2*np.pi/self._wavelength, (self._Ny,self._Nx))
        kSqr = kx**2+ky**2
        mask = kSqr.flatten()<k0.flatten()**2

        filt = 1j*np.zeros(self._Nx*self._Ny)
        filt[mask] = np.exp(1j*np.sqrt(k0.flatten()[mask]**2-kSqr.flatten()[mask]))
        self._objective_filter = np.reshape(filt, (self._Ny,self._Nx))
        self._objective_mask = (kSqr<(k0*self._max_NA_objective)**2).astype(float)
        self._kSqr = kSqr
        self._k0 = k0
        self._z = 0

    @property
    def focused_vals(self):
        """Numpy array for the optical fields values after focalisation by the microscope
        objective, of shape (Ny,Nx).
        """
        return self._focused_vals

    @property
    def vals(self):
        """Numpy array for the optical fields values, of shape
        (Ny,Nx).
        """
        return self._vals

    @vals.setter
    def vals(self, fields_ndarray):
        if self._vals.shape==fields_ndarray.shape:
            self._vals[:] = fields_ndarray
        else:
            raise Exception("Wrong shape for the optical field ndarray")

    @property
    def z_focus(self):
        return self._z

    def update_NA_objective(self, new_NA):
        NA = max(0,min(self._max_NA_objective,new_NA))
        if NA!=self._cur_NA:
            self._cur_NA = NA
            self._objective_mask = (self._kSqr<(self._k0*NA)**2).astype(float)

    def focus_fields(self, z_focus=None):
        """Propagate the optical fields through the objective lens to the screen conjugate
        to the focusing plane (whose altitude inside the sample is set with the parameter
        z_focus)."""
        if z_focus is not None:
            filt = self._objective_mask*self._objective_filter**np.abs(z_focus)
            self._z = z_focus
        else:
            z_focus = self._z
            filt = self._objective_mask*self._objective_filter**np.abs(z_focus)

        self._fft_plan()
        self._fft_vals *= np.conj(filt) if z_focus<0 else filt
        self._ifft_plan()


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


class ImagingSetup:
    def __init__(self, ax_sys, fig):
        self.x_obj = 0
        self.x_foc = 0
        self.x_lens = 2
        self.x_im = 4
        self.f_lens = 1
        self.ax_sys = ax_sys
        self.fig = fig

        N = 101
        L = 20
        coords = np.linspace(-L/2,L/2,N)
        yvals,xvals = np.meshgrid(coords,coords,indexing="ij")
        rvals = np.sqrt(xvals**2+yvals**2)
        self.im_obj = np.ones((N,N))
        self.im_obj[rvals<L/8] = 0

        self.field = OpticalField(mesh_dimensions=[N,N],mesh_lengths=[L,L])
        self.field.vals = np.sqrt(self.im_obj)
        self.field.focus_fields()

    def set_ax_foc(self, ax_foc):
        self.ax_foc = ax_foc

    @property
    def im_foc(self):
        return np.abs(self.field.focused_vals)**2

    def draw_system(self):
        self.ax_obj, = self.ax_sys.plot(self.x_obj*np.ones((2,)), [-1.5,1.5], "k-.")
        self.ax_obj_text = self.ax_sys.text(self.x_obj, -1.7, "Object\nplane",
            horizontalalignment='center', verticalalignment='center')

        self.ax_sys.plot(self.x_foc*np.ones((2,)), [-1.5,1.5], "k-.")
        self.ax_sys.text(self.x_obj, 1.7, "Focusing\nplane",
            horizontalalignment='center', verticalalignment='center')

        self.ax_lens = Ellipse(xy=(self.x_lens,0),width=0.2,height=3,facecolor="tab:gray")
        self.ax_sys.add_artist(self.ax_lens)
        self.ax_sys.text(self.x_lens, 1.6, "Objective",
            horizontalalignment='center', verticalalignment='center')
        self.ax_sys.text(self.x_lens, -1.6, "NA=R/(2f)",
            horizontalalignment='center', verticalalignment='center')

        self.ax_sys.plot(self.x_im*np.ones((2,)), [-1.5,1.5], "k-.")
        self.ax_sys.text(self.x_im, 1.7, "Image\nplane",
            horizontalalignment='center', verticalalignment='center')

        self.arr_2f_1 = DoubleArrow(self.ax_sys, self.x_foc, self.x_lens, 0, 0, "2f")
        self.arr_2f_2 = DoubleArrow(self.ax_sys, self.x_lens, self.x_im, 0, 0, "2f")
        self.arr_R = DoubleArrow(self.ax_sys, self.x_lens-0.4, self.x_lens-0.4, 0, 1.45, "R")

    def update(self, NA, z_foc):
        self.field.update_NA_objective(NA)
        self.field.focus_fields(-z_foc)
        self.ax_foc.set_data(np.abs(self.field.focused_vals)**2)

        self.ax_obj.set_xdata(0.02*z_foc*np.ones((2,)))
        self.ax_obj_text.set_position((0.02*z_foc,-1.7))

        u = NA/self.field._max_NA_objective
        self.ax_lens.width = 0.4*u
        self.ax_lens.height = 3*u
        self.arr_R.update(self.x_lens-0.4, self.x_lens-0.4, 0, 1.45*u)

        self.fig.canvas.draw_idle()


def draw_imaging_setup(*,figsize):
    plt.rcParams['figure.figsize'] = figsize
    plt.ion()

    fig = plt.figure(3)
    ax_sys = fig.add_subplot(121)
    ax_obj = fig.add_subplot(222)
    ax_im = fig.add_subplot(224)
    plt.tight_layout()
    
    #  ax_sys = plt.subplot(1,2,1)

    imaging_setup = ImagingSetup(ax_sys,fig)
    imaging_setup.draw_system()
    ax_sys.set_xlim(-0.5,4.3)
    ax_sys.set_aspect('equal')
    ax_sys.get_xaxis().set_visible(False)
    ax_sys.get_yaxis().set_visible(False)
    ax_sys.axis('off')

    opts = {"cmap": "gray", "interpolation": "bicubic", "vmin": 0, "vmax": 1.5}

    #  ax_obj = plt.subplot(2,2,2)
    ax_obj.axes.xaxis.set_visible(False)
    ax_obj.axes.yaxis.set_visible(False)
    ax_obj.imshow(imaging_setup.im_obj,**opts)
    ax_obj.set_title("Object plane")

    #  ax_im = plt.subplot(2,2,4)
    ax_im.axes.xaxis.set_visible(False)
    ax_im.axes.yaxis.set_visible(False)
    ax_foc = ax_im.imshow(imaging_setup.im_foc,**opts)
    ax_im.set_title("Image plane")

    imaging_setup.set_ax_foc(ax_foc)
    fig.canvas.draw()
    
    return imaging_setup
