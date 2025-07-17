import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
from scipy.interpolate import griddata, RegularGridInterpolator

from discretize import TensorMesh

from SimPEG.utils import plot2Ddata
from SimPEG import (
    maps,
    data,
    inverse_problem,
    data_misfit,
    regularization,
    optimization,
    directives,
    inversion,
    utils,
)



class Plot_tool():
    
    def __init__(self, mesh, mesh_rm, meshinfo_dic, m_dens, m_susc, ind_active):
        '''
        Initialize plotting tool
        mesh: SimPEG mesh with padding zone
        mesh_rm: SimPEG mesh without padding zone (only core zone)
        meshinfo_dic: input parameters
        m_dens, m_susc = wires * recovered_model for jointly inverted model
        ind_active: index of active cells 
        '''
        self.mesh = mesh
        self.mesh_rm = mesh_rm
        self.xpad = meshinfo_dic['xpad']
        self.ypad = meshinfo_dic['ypad']
        self.core_bounds = meshinfo_dic['select_region']

        # Find the mesh grids coordinate along x/y/z direction
        self.x_grid_array = np.unique(mesh.gridN[:,0])
        self.y_grid_array = np.unique(mesh.gridN[:,1])
        self.z_grid_array = np.unique(mesh.gridN[:,2])

        # Find the mesh centers coordinate along x/y/z direction
        self.x_center_array = mesh.cell_centers_x
        self.y_center_array = mesh.cell_centers_y
        self.z_center_array = mesh.cell_centers_z

        # x/y/z coordinate of all flattened meshgrid centers(!) and 3*n grid arrays
        self.meshgrid_x = mesh.cell_centers[:, 0]
        self.meshgrid_y = mesh.cell_centers[:, 1]
        self.meshgrid_z = mesh.cell_centers[:, 2]
        meshgrid_xyz = np.stack((self.meshgrid_x, self.meshgrid_y, self.meshgrid_z), axis=1)

        # meshgrid of mesh centers(!)
        meshgrid_xx, meshgrid_yy, meshgrid_zz = np.meshgrid(self.x_center_array, self.y_center_array, self.z_center_array, indexing='ij')
        meshgridgrid_shape = meshgrid_xx.shape  # shape of meshgrid

        # Cell mapping with topo active cells and map recovered model
        self.plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
        self.recovered_model_padnan_grv = self.plotting_map * m_dens
        self.recovered_model_padnan_mag = self.plotting_map * m_susc
        grid3D_values_grv = self.recovered_model_padnan_grv.reshape(meshgridgrid_shape)  # reshape 1D recovered model array to meshgrid shape
        grid3D_values_mag = self.recovered_model_padnan_mag.reshape(meshgridgrid_shape)  # reshape 1D recovered model array to meshgrid shape
        



    def plot_grv_map(self, folder_name, grv_data, range_east=None, range_north=None, figsize=(8,7), tfs=25, lfs=22, axfs=22, cbfs=20, Mts_flag=True, saveformat="svg"):
        '''
        grv_data: gravity data in shape (n,4) (x,y,z,data)
        Mts_flag: plot marks for mountains or not
        '''
        np.random.seed(19680801)
        npts = grv_data[:, 0].shape[0]
        ngridx = 400
        ngridy = 800
        x = grv_data[:, 0]
        y = grv_data[:, 1]
        z = grv_data[:, 3]

        fig, ax = plt.subplots(1,1,figsize=figsize)

        xi = np.linspace(self.core_bounds[0], self.core_bounds[1], ngridx)
        yi = np.linspace(self.core_bounds[2], self.core_bounds[3], ngridy)
        zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

        ax.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
        cntr1 = ax.contourf(xi, yi, zi, levels=14, cmap="jet")

        if Mts_flag==True:
                mark1 = ax.scatter(520360, 4313800, s=140, marker='^', c="black")
                mark1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot1 = ax.annotate("  Mt. Konocti", (520360, 4313800), fontsize=21, color='black')
                annot1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                mark2 = ax.scatter(522030, 4304100, s=140, marker='^', c="black")
                mark2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot2 = ax.annotate("  Mt. Hannah", (522030, 4304100), fontsize=21, color='black')
                annot2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                mark3 = ax.scatter(522490, 4295200, s=140, marker='^', c="black")
                mark3.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot3 = ax.annotate("  Mt. Cobb", (522490, 4295200), fontsize=21, color='black')
                annot3.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
        ax.plot(x, y, 'ko', ms=2)
        ax.set(xlim=(self.core_bounds[0], self.core_bounds[1]), ylim=(self.core_bounds[2], self.core_bounds[3]))
 
        ax.set_title("Isostatic Gravity Anomaly", fontsize=tfs, pad=15)
        ax.set_aspect('equal')
        ax.set_xlabel("Easting [m]", fontsize=lfs)
        ax.set_ylabel("Northing [m]", fontsize=lfs)
        ax.ticklabel_format(style='sci', scilimits=[-2,2], useMathText=True)
        ax.tick_params(direction='in', length=7, axis='both', which='major', labelsize=axfs, pad=10)
        ax.tick_params(direction='in', length=3, axis='both', which='minor', labelsize=axfs, pad=10)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10000))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1000))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1000))
        ax.xaxis.get_offset_text().set_fontsize(axfs)  # For x-axis
        ax.yaxis.get_offset_text().set_fontsize(axfs)  # For y-axis (if needed)

        cbar = plt.colorbar(cntr1, ax=ax)
        cbar.set_label("$mGal$", rotation=270, labelpad=20, size=cbfs)
        cbar.ax.tick_params(labelsize=cbfs)

        # !!!! This part is used to adjust the y postion (and x position) of
        # the exponent magnitude of y ticklable, beacuse the second parameter of
        # ax2.get_yaxis().get_offset_text().set_position((-0.25,1.1)) is not functional
        fig.canvas.draw()
        offset_text_obj = ax.get_yaxis().get_offset_text()
        offset_str = offset_text_obj.get_text()
        offset_text_obj.set_visible(False)
        ax.text(-0.25, 1.05, offset_str, transform=ax.transAxes, fontsize=axfs)
        # ------------------------------------------
        plt.tight_layout()
        if saveformat == "png":
            plt.savefig("./temp_inv_out/" + folder_name + "/saved_figures/gravity_data.png", dpi=300)
        else:
            plt.savefig("./temp_inv_out/" + folder_name + "/saved_figures/gravity_data.svg")
        plt.close()



    def plot_mag_map(self, folder_name, mag_data, range_east=None, range_north=None, figsize=(8,7), tfs=25, lfs=22, axfs=22, cbfs=20, Mts_flag=True, saveformat="svg"):
        '''
        mag_data: magnetic data in shape (n,4) (x,y,z,data)
        Mts_flag: plot marks for mountains or not
        '''
        np.random.seed(19680801)
        x = mag_data[:, 0]
        y = mag_data[:, 1]
        z = mag_data[:, 3]

        fig, ax = plt.subplots(1,1,figsize=figsize)
        cntr1 = ax.tricontourf(x, y, z, levels=100, cmap='jet')  # 'viridis'

        if Mts_flag==True:
                mark1 = ax.scatter(520360, 4313800, s=140, marker='^', c="black")
                mark1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot1 = ax.annotate("  Mt. Konocti", (520360, 4313800), fontsize=21, color='black')
                annot1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                mark2 = ax.scatter(522030, 4304100, s=140, marker='^', c="black")
                mark2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot2 = ax.annotate("  Mt. Hannah", (522030, 4304100), fontsize=21, color='black')
                annot2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                mark3 = ax.scatter(522490, 4295200, s=140, marker='^', c="black")
                mark3.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot3 = ax.annotate("  Mt. Cobb", (522490, 4295200), fontsize=21, color='black')
                annot3.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])

        ax.set(xlim=(self.core_bounds[0], self.core_bounds[1]), ylim=(self.core_bounds[2], self.core_bounds[3]))
 
        ax.set_title("Total Field Magnetic Anomaly", fontsize=tfs, pad=15)
        ax.set_aspect('equal')
        ax.set_xlabel("Easting [m]", fontsize=lfs)
        ax.set_ylabel("Northing [m]", fontsize=lfs)
        ax.ticklabel_format(style='sci', scilimits=[-2,2], useMathText=True)
        ax.tick_params(direction='in', length=7, axis='both', which='major', labelsize=axfs, pad=10)
        ax.tick_params(direction='in', length=3, axis='both', which='minor', labelsize=axfs, pad=10)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10000))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1000))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1000))
        ax.xaxis.get_offset_text().set_fontsize(axfs)  # For x-axis
        ax.yaxis.get_offset_text().set_fontsize(axfs)  # For y-axis (if needed)

        cbar = plt.colorbar(cntr1, ax=ax)
        cbar.set_label("$nT$", rotation=270, labelpad=20, size=cbfs)
        cbar.ax.tick_params(labelsize=cbfs)

        # !!!! This part is used to adjust the y postion (and x position) of
        # the exponent magnitude of y ticklable, beacuse the second parameter of
        # ax2.get_yaxis().get_offset_text().set_position((-0.25,1.1)) is not functional
        fig.canvas.draw()
        offset_text_obj = ax.get_yaxis().get_offset_text()
        offset_str = offset_text_obj.get_text()
        offset_text_obj.set_visible(False)
        ax.text(-0.25, 1.05, offset_str, transform=ax.transAxes, fontsize=axfs)
        # ------------------------------------------
        plt.tight_layout()
        if saveformat == "png":
            plt.savefig("./temp_inv_out/" + folder_name + "/saved_figures/magnetic_data.png", dpi=300)
        else:
            plt.savefig("./temp_inv_out/" + folder_name + "/saved_figures/magnetic_data.svg")
        plt.close()




    def plot_topo_map(self, folder_name, topo_data, range_east=None, range_north=None, figsize=(8,7), tfs=25, lfs=22, axfs=22, cbfs=20, Mts_flag=True):
        '''
        topo_data: topography data in shape (n,3) (x,y,z)
        Mts_flag: plot marks for mountains or not
        '''
        np.random.seed(19680801)
        x = topo_data[:, 0]
        y = topo_data[:, 1]
        z = topo_data[:, 2]

        fig, ax = plt.subplots(1,1,figsize=figsize)
        ax.tricontour(x, y, z, levels=20, linewidths=0.5, colors='k')
        cntrf = ax.tricontourf(x, y, z, levels=14, cmap='terrain')
        # grv_f = ax.tricontourf(topo_xyz[:, 0], topo_xyz[:, 1], topo_xyz[:, 2], levels=14, cmap='terrain')  # 'viridis'
        # fig.colorbar(grv_f, ax=ax)


        if Mts_flag==True:
                mark1 = ax.scatter(520360, 4313800, s=140, marker='^', c="black")
                mark1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot1 = ax.annotate("  Mt. Konocti", (520360, 4313800), fontsize=21, color='black')
                annot1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                mark2 = ax.scatter(522030, 4304100, s=140, marker='^', c="black")
                mark2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot2 = ax.annotate("  Mt. Hannah", (522030, 4304100), fontsize=21, color='black')
                annot2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                mark3 = ax.scatter(522490, 4295200, s=140, marker='^', c="black")
                mark3.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot3 = ax.annotate("  Mt. Cobb", (522490, 4295200), fontsize=21, color='black')
                annot3.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])

        ax.set(xlim=(self.core_bounds[0], self.core_bounds[1]), ylim=(self.core_bounds[2], self.core_bounds[3]))
 
        ax.set_title("Topography", fontsize=tfs, pad=15)
        ax.set_aspect('equal')
        ax.set_xlabel("Easting [m]", fontsize=lfs)
        ax.set_ylabel("Northing [m]", fontsize=lfs)
        ax.ticklabel_format(style='sci', scilimits=[-2,2], useMathText=True)
        ax.tick_params(direction='in', length=7, axis='both', which='major', labelsize=axfs, pad=10)
        ax.tick_params(direction='in', length=3, axis='both', which='minor', labelsize=axfs, pad=10)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10000))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1000))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1000))
        ax.xaxis.get_offset_text().set_fontsize(axfs)  # For x-axis
        ax.yaxis.get_offset_text().set_fontsize(axfs)  # For y-axis (if needed)

        cbar = plt.colorbar(cntrf, ax=ax)
        cbar.set_label("$m$", rotation=270, labelpad=20, size=cbfs)
        cbar.ax.tick_params(labelsize=cbfs)

        # fig.colorbar(cntr1, ax=ax)
        # cbar = fig.axes[-1]  # The colorbar is usually the last axes
        # cbar.tick_params(labelsize=cbfs)  # Adjust the tick font size
        # cbar.set_label("$mGal$", rotation=270, labelpad=15, size=cbfs)
        # !!!! This part is used to adjust the y postion (and x position) of
        # the exponent magnitude of y ticklable, beacuse the second parameter of
        # ax2.get_yaxis().get_offset_text().set_position((-0.25,1.1)) is not functional
        fig.canvas.draw()
        offset_text_obj = ax.get_yaxis().get_offset_text()
        offset_str = offset_text_obj.get_text()
        offset_text_obj.set_visible(False)
        ax.text(-0.25, 1.05, offset_str, transform=ax.transAxes, fontsize=axfs)
        # ------------------------------------------
        plt.tight_layout()
        # plt.savefig("./temp_inv_out/" + folder_name + "/saved_figures/topography.svg")
        plt.savefig("./temp_inv_out/" + folder_name + "/saved_figures/topography.png", dpi=300)
        plt.close()



    def plot_grv_compare(self, folder_name, stations, dobs, dpred, std_grv, range_east=None, range_north=None, figsize=(19, 6)):
        '''
        dobs and dpred: 1D array
        stations: the spatial coordinate of measument points, in shape (n,3)
        std_grv: gravity data noise level
        '''
        # Observed data | Predicted data | Normalized data misfit
        data_array = np.c_[dobs, dpred, (dobs - dpred)/std_grv]

        fig = plt.figure(figsize=(19, 6))
        plot_title = ["Observed", "Predicted", "Normalized Misfit"]
        plot_units = ["mGal", "mGal", "std={0} mGal".format(std_grv)]
        cmap_str = ["jet", "jet", "bwr"]
        cmap_list = [mpl.cm.jet, mpl.cm.jet, mpl.cm.bwr]

        ax1 = 3 * [None]
        ax2 = 3 * [None]
        norm = 3 * [None]
        cbar = 3 * [None]
        cplot = 3 * [None]
        v_lim = [[np.min(dobs),np.max(dobs)], [np.min(dobs),np.max(dobs)], [-4,4]]

        for ii in range(0, 3):
            ax1[ii] = fig.add_axes([0.335 * ii + 0.06, 0.16, 0.21, 0.73])
            cplot[ii] = plot2Ddata(
                stations,
                data_array[:, ii],
                ax=ax1[ii],
                ncontour=40,
                clim=(v_lim[ii][0], v_lim[ii][1]),
                contourOpts={"cmap": cmap_str[ii]},
            )
            ax1[ii].set_title(plot_title[ii], fontsize=23, pad=10)
            ax1[ii].set_xlabel("Easting [m]", fontsize=20)
            ax1[ii].set_ylabel("Northing [m]", fontsize=20)
            ax1[ii].set_xlim([self.core_bounds[0], self.core_bounds[1]])
            ax1[ii].set_ylim([self.core_bounds[2], self.core_bounds[3]])

            ax1[ii].tick_params(direction='in', length=7, axis='both', which='major', labelsize=18, pad=10)
            ax1[ii].tick_params(direction='in', length=3, axis='both', which='minor', labelsize=18, pad=10)
            ax1[ii].xaxis.set_major_locator(ticker.MultipleLocator(10000))
            ax1[ii].xaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax1[ii].yaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax1[ii].xaxis.get_offset_text().set_fontsize(18)  # For x-axis
            ax1[ii].yaxis.get_offset_text().set_fontsize(18)  # For y-axis (if needed)

            # ax1[ii].plot([504280, 534000],[4281000, 4319850], color="black", linestyle="dotted", linewidth=2)  # Profile Michael AA
            # ax1[ii].plot([526710, 516860],[4281140, 4330000], color="black", linestyle="dotted", linewidth=2)  # Profile Michael BB
            ax1[ii].ticklabel_format(style='sci', scilimits=[-2,2], useMathText=True)
            plt.gca().set_aspect('equal')

            # !!!! This part is used to adjust the y postion (and x position) of
            # the exponent magnitude of y ticklable, beacuse the second parameter of
            # ax2.get_yaxis().get_offset_text().set_position((-0.25,1.1)) is not functional
            fig.canvas.draw()
            offset_text_obj = ax1[ii].get_yaxis().get_offset_text()
            offset_str = offset_text_obj.get_text()
            offset_text_obj.set_visible(False)
            ax1[ii].text(-0.25, 1.05, offset_str, transform=ax1[ii].transAxes, fontsize=20)
            # ------------------------------------------

            ax2[ii] = fig.add_axes([0.335 * ii + 0.27, 0.16, 0.01, 0.73])
            norm[ii] = mpl.colors.Normalize(vmin=v_lim[ii][0], vmax=v_lim[ii][1])
            cbar[ii] = mpl.colorbar.ColorbarBase(
                ax2[ii], norm=norm[ii], orientation="vertical", cmap=cmap_list[ii]
            )
            cbar[ii].set_label(plot_units[ii], rotation=270, labelpad=16, size=18)
            cbar[ii].ax.tick_params(labelsize=18)
        plt.savefig("./temp_inv_out/" + folder_name  + "/saved_figures/grv_data_misfit_jet.svg")



    def plot_mag_compare(self, folder_name, stations, dobs, dpred, std_mag, range_east=None, range_north=None, figsize=(19, 6)):
        '''
        dobs and dpred: 1D array
        stations: the spatial coordinate of measument points, in shape (n,3)
        std_mag: magnetic data noise level
        '''
        # Observed data | Predicted data | Normalized data misfit
        data_array = np.c_[dobs, dpred, (dobs - dpred)/std_mag]

        fig = plt.figure(figsize=(19, 6))
        plot_title = ["Observed", "Predicted", "Normalized Misfit"]
        plot_units = ["nT", "nT", "std={0} nT".format(std_mag)]
        cmap_str = ["jet", "jet", "bwr"]
        cmap_list = [mpl.cm.jet, mpl.cm.jet, mpl.cm.bwr]

        ax1 = 3 * [None]
        ax2 = 3 * [None]
        norm = 3 * [None]
        cbar = 3 * [None]
        cplot = 3 * [None]
        v_lim = [[np.min(dobs),np.max(dobs)], [np.min(dobs),np.max(dobs)], [-4,4]]

        for ii in range(0, 3):
            ax1[ii] = fig.add_axes([0.335 * ii + 0.06, 0.16, 0.21, 0.73])
            cplot[ii] = plot2Ddata(
                stations,
                data_array[:, ii],
                ax=ax1[ii],
                ncontour=30,
                clim=(v_lim[ii][0], v_lim[ii][1]),
                contourOpts={"cmap": cmap_str[ii]},
            )
            ax1[ii].set_title(plot_title[ii], fontsize=23, pad=10)
            ax1[ii].set_xlabel("Easting [m]", fontsize=20)
            ax1[ii].set_ylabel("Northing [m]", fontsize=20)
            ax1[ii].set_xlim([self.core_bounds[0], self.core_bounds[1]])
            ax1[ii].set_ylim([self.core_bounds[2], self.core_bounds[3]])

            ax1[ii].tick_params(direction='in', length=7, axis='both', which='major', labelsize=18, pad=10)
            ax1[ii].tick_params(direction='in', length=3, axis='both', which='minor', labelsize=18, pad=10)
            ax1[ii].xaxis.set_major_locator(ticker.MultipleLocator(10000))
            ax1[ii].xaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax1[ii].yaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax1[ii].xaxis.get_offset_text().set_fontsize(18)  # For x-axis
            ax1[ii].yaxis.get_offset_text().set_fontsize(18)  # For y-axis (if needed)

            # ax1[ii].plot([504280, 534000],[4281000, 4319850], color="black", linestyle="dotted", linewidth=2)  # Profile Michael AA
            # ax1[ii].plot([526710, 516860],[4281140, 4330000], color="black", linestyle="dotted", linewidth=2)  # Profile Michael BB
            ax1[ii].ticklabel_format(style='sci', scilimits=[-2,2], useMathText=True)
            plt.gca().set_aspect('equal')

            # !!!! This part is used to adjust the y postion (and x position) of
            # the exponent magnitude of y ticklable, beacuse the second parameter of
            # ax2.get_yaxis().get_offset_text().set_position((-0.25,1.1)) is not functional
            fig.canvas.draw()
            offset_text_obj = ax1[ii].get_yaxis().get_offset_text()
            offset_str = offset_text_obj.get_text()
            offset_text_obj.set_visible(False)
            ax1[ii].text(-0.25, 1.05, offset_str, transform=ax1[ii].transAxes, fontsize=20)
            # ------------------------------------------

            ax2[ii] = fig.add_axes([0.335 * ii + 0.27, 0.16, 0.01, 0.73])
            norm[ii] = mpl.colors.Normalize(vmin=v_lim[ii][0], vmax=v_lim[ii][1])
            cbar[ii] = mpl.colorbar.ColorbarBase(
                ax2[ii], norm=norm[ii], orientation="vertical", cmap=cmap_list[ii]
            )
            cbar[ii].set_label(plot_units[ii], rotation=270, labelpad=16, size=18)
            cbar[ii].ax.tick_params(labelsize=18)
        plt.savefig("./temp_inv_out/" + folder_name  + "/saved_figures/mag_data_misfit_jet.svg")












    def plot_easting_slices(self, folder_name, range_north=None, figsize=(9,10), den_clim=(-0.3, 0.3), sus_clim=(-0.05, 0.05), tfs=25, lfs=22, axfs=22, cbfs=20):
        '''
        Plot vertical slices for recovered models along northing direction at each easting grid
        
        '''
        if range_north is None:
            range_north = (self.core_bounds[2],self.core_bounds[3])

        for ii in range(self.xpad, self.x_grid_array.shape[0]-self.xpad):
        # for ii in range(self.xpad, self.xpad+1):
        # Plot Recovered Model
            slicePosition = self.x_center_array[ii]
            # sliceInd = int(round(np.searchsorted(self.x_center_array, slicePosition)))
            fig = plt.figure(figsize=figsize)
            ax1 = plt.subplot(211)
            (im,) = self.mesh.plot_slice(
                self.recovered_model_padnan_grv,
                normal="X",
                ax=ax1,
                ind=int(ii),
                range_x=range_north,
                clim=den_clim,
                pcolor_opts={"cmap": "bwr"}
            )
            # ax1.set_title("Inverted density model, x = {0}~{1} [m]".format(self.x_grid_array[ii], self.x_grid_array[ii+1]), fontsize=tfs)
            ax1.set_title("Density".format((self.x_grid_array[ii]+self.x_grid_array[ii+1])/2/1000), fontsize=tfs, pad=15)
            ax1.set_aspect('equal') 
            ax1.set_xlabel("Northing [m]", fontsize=lfs)
            ax1.set_ylabel("Elevation [m]", fontsize=lfs)
            ax1.ticklabel_format(style='sci', axis='x', scilimits=[-2,2], useMathText=True)
            ax1.ticklabel_format(style='sci', axis='y', scilimits=[-2,2], useMathText=True)
            ax1.get_yaxis().get_offset_text().set_position((-0.15,0))
            ax1.tick_params(direction='in', length=7, axis='both', which='major', labelsize=axfs, pad=10)
            ax1.tick_params(direction='in', length=3, axis='both', which='minor', labelsize=axfs, pad=10)
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(10000))
            ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax1.xaxis.get_offset_text().set_fontsize(axfs)  # For x-axis
            ax1.yaxis.get_offset_text().set_fontsize(axfs)  # For y-axis (if needed)
            cbar = plt.colorbar(im)
            cbar.set_label("$g/cm^3$", rotation=270, labelpad=15, size=cbfs)
            cbar.ax.tick_params(labelsize=cbfs)

            ax2 = plt.subplot(212)
            (im,) = self.mesh.plot_slice(
                self.recovered_model_padnan_mag,
                normal="X",
                ax=ax2,
                ind=int(ii),
                range_x=range_north,
                clim=sus_clim,
                pcolor_opts={"cmap": "bwr"}
            )
            # ax2.set_title("Inverted susceptibility model, x = {0}~{1} [m]".format(self.x_grid_array[ii], self.x_grid_array[ii+1]), fontsize=tfs)
            ax2.set_title("Susceptibility".format((self.x_grid_array[ii]+self.x_grid_array[ii+1])/2/1000), fontsize=tfs, pad=15)
            ax2.set_aspect('equal')
            ax2.set_xlabel("Northing [m]", fontsize=lfs)
            ax2.set_ylabel("Elevation [m]", fontsize=lfs)
            ax2.ticklabel_format(style='sci', axis='x', scilimits=[-2,2], useMathText=True)
            ax2.ticklabel_format(style='sci', axis='y', scilimits=[-2,2], useMathText=True)
            ax2.get_yaxis().get_offset_text().set_position((-0.15,0))
            ax2.tick_params(direction='in', length=7, axis='both', which='major', labelsize=axfs, pad=10)
            ax2.tick_params(direction='in', length=3, axis='both', which='minor', labelsize=axfs, pad=10)
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(10000))
            ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax2.xaxis.get_offset_text().set_fontsize(axfs)  # For x-axis
            ax2.yaxis.get_offset_text().set_fontsize(axfs)  # For y-axis (if needed)
            cbar = plt.colorbar(im)
            cbar.set_label("SI", rotation=270, labelpad=15, size=cbfs)
            cbar.ax.tick_params(labelsize=cbfs)

            plt.tight_layout()
            plt.savefig("./temp_inv_out/" + folder_name + "/saved_figures/x/X_slice_{0}m.svg".format(slicePosition))
            plt.close()



    def plot_northing_slices(self, folder_name, range_east=None, figsize=(8.5,9.1), den_clim=(-0.3, 0.3), sus_clim=(-0.05, 0.05), tfs=25, lfs=22, axfs=22, cbfs=20):
        '''
        Plot vertical slices for recovered models along easting direction at each northing grid
        '''
        if range_east is None:
            range_east = (self.core_bounds[0],self.core_bounds[1])

        for ii in range(self.ypad, self.y_grid_array.shape[0]-self.ypad):
        # for ii in range(self.ypad, self.ypad+1):
        # Plot Recovered Model
            slicePosition = self.y_center_array[ii]
            # sliceInd = int(round(np.searchsorted(self.y_center_array, slicePosition)))
            fig = plt.figure(figsize=figsize)
            ax1 = plt.subplot(211)
            (im,) = self.mesh.plot_slice(
                self.recovered_model_padnan_grv,
                normal="Y",
                ax=ax1,
                ind=int(ii),
                range_x=range_east,
                clim=den_clim,
                pcolor_opts={"cmap": "bwr"}
            )

            # ax1.set_title("Inverted density model, y = {0}~{1} [m]".format(self.y_grid_array[ii], self.y_grid_array[ii+1]), fontsize=tfs)
            ax1.set_title("Density".format((self.y_grid_array[ii]+self.y_grid_array[ii+1])/2/1000), fontsize=tfs, pad=15)
            ax1.set_aspect('equal') 
            ax1.set_xlabel("Easting [m]", fontsize=lfs)
            ax1.set_ylabel("Elevation [m]", fontsize=lfs)

            ax1.ticklabel_format(style='sci', axis='x', scilimits=[-2,2], useMathText=True)
            ax1.ticklabel_format(style='sci', axis='y', scilimits=[-2,2], useMathText=True)
            ax1.get_yaxis().get_offset_text().set_position((-0.20,0))
            ax1.tick_params(direction='in', length=7, axis='both', which='major', labelsize=axfs, pad=10)
            ax1.tick_params(direction='in', length=3, axis='both', which='minor', labelsize=axfs, pad=10)
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(10000))
            ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax1.xaxis.get_offset_text().set_fontsize(axfs)  # For x-axis
            ax1.yaxis.get_offset_text().set_fontsize(axfs)  # For y-axis (if needed)
            cbar = plt.colorbar(im)
            cbar.set_label("$g/cm^3$", rotation=270, labelpad=15, size=cbfs)
            cbar.ax.tick_params(labelsize=cbfs)

            ax2 = plt.subplot(212)
            (im,) = self.mesh.plot_slice(
                self.recovered_model_padnan_mag,
                normal="Y",
                ax=ax2,
                ind=int(ii),
                range_x=range_east,
                clim=sus_clim,
                pcolor_opts={"cmap": "bwr"}
            )
            # ax2.set_title("Inverted susceptibility model, y = {0}~{1} [m]".format(self.y_grid_array[ii], self.y_grid_array[ii+1]), fontsize=tfs)
            ax2.set_title("Susceptibility".format((self.y_grid_array[ii]+self.y_grid_array[ii+1])/2/1000), fontsize=tfs, pad=15)
            ax2.set_aspect('equal')
            ax2.set_xlabel("Easting [m]", fontsize=lfs)
            ax2.set_ylabel("Elevation [m]", fontsize=lfs)
            ax2.ticklabel_format(style='sci', axis='x', scilimits=[-2,2], useMathText=True)
            ax2.ticklabel_format(style='sci', axis='y', scilimits=[-2,2], useMathText=True)
            ax2.get_yaxis().get_offset_text().set_position((-0.20,0))
            ax2.tick_params(direction='in', length=7, axis='both', which='major', labelsize=axfs, pad=10)
            ax2.tick_params(direction='in', length=3, axis='both', which='minor', labelsize=axfs, pad=10)
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(10000))
            ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax2.xaxis.get_offset_text().set_fontsize(axfs)  # For x-axis
            ax2.yaxis.get_offset_text().set_fontsize(axfs)  # For y-axis (if needed)
            cbar = plt.colorbar(im)
            cbar.set_label("SI", rotation=270, labelpad=15, size=cbfs)
            cbar.ax.tick_params(labelsize=cbfs)

            plt.tight_layout()
            plt.savefig("./temp_inv_out/" + folder_name + "/saved_figures/y/Y_slice_{0}m.svg".format(slicePosition))
            plt.close()



    def plot_depth_slices(self, folder_name, range_east=None, range_north=None, figsize=(16,7), den_clim=(-0.3, 0.3), sus_clim=(-0.05, 0.05), tfs=25, lfs=22, axfs=22, cbfs=20, Mts_flag=True):
        '''
        Plot depth slices for recovered models
        Mts_flag: plot marks for mountains or not
        '''
        if range_east is None:
            range_east = (self.core_bounds[0],self.core_bounds[1])
        if range_north is None:
            range_north = (self.core_bounds[2],self.core_bounds[3])

        for ii in range(0, self.z_grid_array.shape[0]-1):
        # for ii in range(15, 16):
        # Plot Recovered Model
            slicePosition = self.z_center_array[ii]
            # sliceInd = int(round(np.searchsorted(self.y_center_array, slicePosition)))
            
            fig = plt.figure(figsize=figsize)
            # ax1 = fig.add_axes([0.04, 0.1, 0.4, 0.8])
            ax1 = plt.subplot(121)
            p1 = self.mesh.plot_slice(
                self.recovered_model_padnan_grv,
                normal="Z",
                ax=ax1,
                ind=int(ii),
                grid=False,
                # grid_opts={'linewidth': 0.05, 'color': 'black'},
                # clim=(np.min(recovered_model), 0.1),
                clim=den_clim,
                range_x=range_east,
                range_y=range_north,
                pcolor_opts={"cmap": "bwr"},
            )

            if Mts_flag==True:
                ax1.scatter(520360, 4313800, s=140, marker='^', c="black")
                ax1.annotate("  Mt. Konocti", (520360, 4313800), fontsize=21)
                ax1.scatter(522030, 4304100, s=140, marker='^', c="black")
                ax1.annotate("  Mt. Hannah", (522030, 4304100), fontsize=21)
                ax1.scatter(522490, 4295200, s=140, marker='^', c="black")
                ax1.annotate("  Mt. Cobb", (522490, 4295200), fontsize=21)

            # ax1.set_title("Inverted density model, z = {0}~{1} [m]".format(self.z_grid_array[ii], self.z_grid_array[ii+1]), fontsize=tfs)
            ax1.set_title("Density, z: {0}[m]".format((self.z_grid_array[ii]+self.z_grid_array[ii+1])/2), fontsize=tfs, pad=15)
            ax1.set_aspect('equal') 
            ax1.set_xlabel("Easting [m]", fontsize=lfs)
            ax1.set_ylabel("Northing [m]", fontsize=lfs)
            ax1.ticklabel_format(style='sci', scilimits=[-2,2], useMathText=True)
            # ax1.get_yaxis().get_offset_text().set_position((-0.25,1.05))  # This is replaced by the lines in the end
            ax1.tick_params(direction='in', length=7, axis='both', which='major', labelsize=axfs, pad=10)
            ax1.tick_params(direction='in', length=3, axis='both', which='minor', labelsize=axfs, pad=10)
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(10000))
            ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax1.xaxis.get_offset_text().set_fontsize(axfs)  # For x-axis
            ax1.yaxis.get_offset_text().set_fontsize(axfs)  # For y-axis (if needed)
            cbar = plt.colorbar(p1[0], ax=ax1)
            cbar.set_label("$g/cm^3$", rotation=270, labelpad=15, size=cbfs)
            cbar.ax.tick_params(labelsize=cbfs)
            # !!!! This part is used to adjust the y postion (and x position) of
            # the exponent magnitude of y ticklable, beacuse the second parameter of
            # ax2.get_yaxis().get_offset_text().set_position((-0.25,1.1)) is not functional
            fig.canvas.draw()
            offset_text_obj = ax1.get_yaxis().get_offset_text()
            offset_str = offset_text_obj.get_text()
            offset_text_obj.set_visible(False)
            ax1.text(-0.25, 1.05, offset_str, transform=ax1.transAxes, fontsize=axfs)
            # ------------------------------------------

            # ax2 = fig.add_axes([0.54, 0.1, 0.4, 0.8])
            ax2 = plt.subplot(122)
            p2 = self.mesh.plot_slice(
                self.recovered_model_padnan_mag,
                normal="Z",
                ax=ax2,
                ind=int(ii),
                grid=False,
                # grid_opts={'linewidth': 0.05, 'color': 'black'},
                # clim=(np.min(recovered_model), 0.1),
                clim=sus_clim,
                range_x=range_east,
                range_y=range_north,
                pcolor_opts={"cmap": "bwr"},
            )

            if Mts_flag==True:
                ax2.scatter(520360, 4313800, s=140, marker='^', c="black")
                ax2.annotate("  Mt. Konocti", (520360, 4313800), fontsize=21)
                ax2.scatter(522030, 4304100, s=140, marker='^', c="black")
                ax2.annotate("  Mt. Hannah", (522030, 4304100), fontsize=21)
                ax2.scatter(522490, 4295200, s=140, marker='^', c="black")
                ax2.annotate("  Mt. Cobb", (522490, 4295200), fontsize=21)

            # ax2.set_title("Inverted susceptibility model, z = {0}~{1} [m]".format(self.z_grid_array[ii], self.z_grid_array[ii+1]), fontsize=tfs)
            ax2.set_title("Susceptibility, z: {0}[m]".format((self.z_grid_array[ii]+self.z_grid_array[ii+1])/2), fontsize=tfs, pad=15)
            ax2.set_aspect('equal') 
            ax2.set_xlabel("Easting [m]", fontsize=lfs)
            ax2.set_ylabel("Northing [m]", fontsize=lfs)
            ax2.ticklabel_format(style='sci', scilimits=[-2,2], useMathText=True)
            # ax2.get_yaxis().get_offset_text().set_position((-0.25,1.05))   # This is replaced by the lines in the end
            ax2.tick_params(direction='in', length=7, axis='both', which='major', labelsize=axfs, pad=10)
            ax2.tick_params(direction='in', length=3, axis='both', which='minor', labelsize=axfs, pad=10)
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(10000))
            ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax2.xaxis.get_offset_text().set_fontsize(axfs)  # For x-axis
            ax2.yaxis.get_offset_text().set_fontsize(axfs)  # For y-axis (if needed)
            cbar = plt.colorbar(p2[0], ax=ax2)
            cbar.set_label("SI", rotation=270, labelpad=15, size=cbfs)
            cbar.ax.tick_params(labelsize=cbfs)
            # !!!! This part is used to adjust the y postion (and x position) of
            # the exponent magnitude of y ticklable, beacuse the second parameter of
            # ax2.get_yaxis().get_offset_text().set_position((-0.25,1.1)) is not functional
            fig.canvas.draw()
            offset_text_obj = ax2.get_yaxis().get_offset_text()
            offset_str = offset_text_obj.get_text()
            offset_text_obj.set_visible(False)
            ax2.text(-0.25, 1.05, offset_str, transform=ax2.transAxes, fontsize=axfs)
            # ------------------------------------------
            plt.tight_layout()
            plt.savefig("./temp_inv_out/" + folder_name + "/saved_figures/z/Z_slice_{0}m.svg".format(slicePosition))
            plt.close()













    def prep_GD_result(self, model_dens_rm, model_susc_rm):
        '''
        Preprocess geology differentiation results
        model_dens_rm: Inverted density model (without padding cells, UBC 1D format)
        model_susc_rm: Inverted magnetic susceptibility model (without padding cells, UBC 1D format)
        '''
        self.model_dens_rm = model_dens_rm
        self.model_susc_rm = model_susc_rm

        model_unit_temp = TensorMesh.read_model_UBC(self.mesh_rm, file_name="./saved_GD_result/model_inpolygon.txt")
        model_unit_temp_3d = np.reshape(
            model_unit_temp, (self.mesh_rm.shape_cells[0],self.mesh_rm.shape_cells[1],self.mesh_rm.shape_cells[2]), 
            order="F"
            )
        self.model_unit_3d = model_unit_temp_3d.copy()
        self.vmax, self.vmin = np.nanmax(model_unit_temp), np.nanmin(model_unit_temp)
        
        gd_result_1 = np.loadtxt("./saved_GD_result/inpolygon_1.txt")
        gd_result_2 = np.loadtxt("./saved_GD_result/inpolygon_2.txt")
        gd_result_3 = np.loadtxt("./saved_GD_result/inpolygon_3.txt")
        gd_result_4 = np.loadtxt("./saved_GD_result/inpolygon_4.txt")
        gd_result_5 = np.loadtxt("./saved_GD_result/inpolygon_5.txt")
        gd_result_6 = np.loadtxt("./saved_GD_result/inpolygon_6.txt")
        gd_result_7 = np.loadtxt("./saved_GD_result/inpolygon_7.txt")
        gd_result_8 = np.loadtxt("./saved_GD_result/inpolygon_8.txt")
        gd_result_9 = np.loadtxt("./saved_GD_result/inpolygon_9.txt")
        gd_result_10 = np.loadtxt("./saved_GD_result/inpolygon_10.txt")

        gd_result_3d_1 = model_unit_temp_3d.copy()
        gd_result_3d_1[gd_result_3d_1!=0] = 10
        self.gd3d_1 = gd_result_3d_1

        gd_result_3d_2 = model_unit_temp_3d.copy()
        gd_result_3d_2[gd_result_3d_2!=1] = 0
        self.gd3d_2 = gd_result_3d_2

        gd_result_3d_3 = model_unit_temp_3d.copy()
        gd_result_3d_3[gd_result_3d_3!=2] = 0
        self.gd3d_3 = gd_result_3d_3

        gd_result_3d_4 = model_unit_temp_3d.copy()
        gd_result_3d_4[gd_result_3d_4!=3] = 0
        self.gd3d_4 = gd_result_3d_4

        gd_result_3d_5 = model_unit_temp_3d.copy()
        gd_result_3d_5[gd_result_3d_5!=4] = 0
        self.gd3d_5 = gd_result_3d_5

        gd_result_3d_6 = model_unit_temp_3d.copy()
        gd_result_3d_6[gd_result_3d_6!=5] = 0
        self.gd3d_6 = gd_result_3d_6

        gd_result_3d_7 = model_unit_temp_3d.copy()
        gd_result_3d_7[gd_result_3d_7!=6] = 0
        self.gd3d_7 = gd_result_3d_7

        gd_result_3d_8 = model_unit_temp_3d.copy()
        gd_result_3d_8[gd_result_3d_8!=7] = 0
        self.gd3d_8 = gd_result_3d_8

        gd_result_3d_9 = model_unit_temp_3d.copy()
        gd_result_3d_9[gd_result_3d_9!=8] = 0
        self.gd3d_9 = gd_result_3d_9

        gd_result_3d_10 = model_unit_temp_3d.copy()
        gd_result_3d_10[gd_result_3d_10!=9] = 0
        self.gd3d_10 = gd_result_3d_10


        # x/y/z coordinate of mesh centers and 3*n grid arrays
        meshrm_x = self.mesh_rm.cell_centers[:, 0]
        meshrm_y = self.mesh_rm.cell_centers[:, 1]
        meshrm_z = self.mesh_rm.cell_centers[:, 2]
        gridrm_xyz = np.stack((meshrm_x, meshrm_y, meshrm_z), axis=1)

        meshrm_x_c = np.unique(meshrm_x)
        meshrm_y_c = np.unique(meshrm_y)
        meshrm_z_c = np.unique(meshrm_z)

        interp1 = RegularGridInterpolator((meshrm_x_c,meshrm_y_c,meshrm_z_c), self.gd3d_1)
        interp2 = RegularGridInterpolator((meshrm_x_c,meshrm_y_c,meshrm_z_c), self.gd3d_2)
        interp3 = RegularGridInterpolator((meshrm_x_c,meshrm_y_c,meshrm_z_c), self.gd3d_3)
        interp4 = RegularGridInterpolator((meshrm_x_c,meshrm_y_c,meshrm_z_c), self.gd3d_4)
        interp5 = RegularGridInterpolator((meshrm_x_c,meshrm_y_c,meshrm_z_c), self.gd3d_5)
        interp6 = RegularGridInterpolator((meshrm_x_c,meshrm_y_c,meshrm_z_c), self.gd3d_6)
        interp7 = RegularGridInterpolator((meshrm_x_c,meshrm_y_c,meshrm_z_c), self.gd3d_7)
        interp8 = RegularGridInterpolator((meshrm_x_c,meshrm_y_c,meshrm_z_c), self.gd3d_8)
        interp9 = RegularGridInterpolator((meshrm_x_c,meshrm_y_c,meshrm_z_c), self.gd3d_9)
        interp10 = RegularGridInterpolator((meshrm_x_c,meshrm_y_c,meshrm_z_c), self.gd3d_10)

        # Created finer mesh for unit contour plot to mitigate interpolation (cause countour plot outside or inside the unit)
        mesh_x_cd = np.arange(510250, 534750, 100)
        mesh_y_cd = np.arange(4290250, 4319750, 100)
        mesh_z_cd = np.arange(-14792.5, 1375, 50)
        self.mesh_x_cd = np.arange(510250, 534750, 100)
        self.mesh_y_cd = np.arange(4290250, 4319750, 100)
        self.mesh_z_cd = np.arange(-14792.5, 1375, 50)

        self.XXd, self.YYd = np.meshgrid(mesh_x_cd, mesh_y_cd) # depth slice
        self.Xd, self.Zd = np.meshgrid(mesh_x_cd, mesh_z_cd) # cross-section
        self.XXXd, self.YYYd, self.ZZZd = np.meshgrid(mesh_x_cd, mesh_y_cd, mesh_z_cd)
        ct_mesh_flat = np.array([self.XXXd.ravel(), self.YYYd.ravel(), self.ZZZd.ravel()]).T

        # Created unit boundaries
        ct_data_flat_1 = interp1(ct_mesh_flat)
        ct_data_3d_1 = ct_data_flat_1.reshape(mesh_y_cd.shape[0], mesh_x_cd.shape[0], mesh_z_cd.shape[0])
        ct_data_3d_1[ct_data_3d_1<5] = 0
        ct_data_3d_1[ct_data_3d_1>=5] = 1
        self.contour3d_1 = ct_data_3d_1

        ct_data_flat_2 = interp2(ct_mesh_flat)
        ct_data_3d_2 = ct_data_flat_2.reshape(mesh_y_cd.shape[0], mesh_x_cd.shape[0], mesh_z_cd.shape[0])
        ct_data_3d_2[ct_data_3d_2<0.5] = 0
        ct_data_3d_2[ct_data_3d_2>=0.5] = 1
        self.contour3d_2 = ct_data_3d_2

        ct_data_flat_3 = interp3(ct_mesh_flat)
        ct_data_3d_3 = ct_data_flat_3.reshape(mesh_y_cd.shape[0], mesh_x_cd.shape[0], mesh_z_cd.shape[0])
        ct_data_3d_3[ct_data_3d_3<1] = 0
        ct_data_3d_3[ct_data_3d_3>=1] = 1
        self.contour3d_3 = ct_data_3d_3

        ct_data_flat_4 = interp4(ct_mesh_flat)
        ct_data_3d_4 = ct_data_flat_4.reshape(mesh_y_cd.shape[0], mesh_x_cd.shape[0], mesh_z_cd.shape[0])
        ct_data_3d_4[ct_data_3d_4<1.5] = 0
        ct_data_3d_4[ct_data_3d_4>=1.5] = 1
        self.contour3d_4 = ct_data_3d_4

        ct_data_flat_5 = interp5(ct_mesh_flat)
        ct_data_3d_5 = ct_data_flat_5.reshape(mesh_y_cd.shape[0], mesh_x_cd.shape[0], mesh_z_cd.shape[0])
        ct_data_3d_5[ct_data_3d_5<2] = 0
        ct_data_3d_5[ct_data_3d_5>=2] = 1
        self.contour3d_5 = ct_data_3d_5

        ct_data_flat_6 = interp6(ct_mesh_flat)
        ct_data_3d_6 = ct_data_flat_6.reshape(mesh_y_cd.shape[0], mesh_x_cd.shape[0], mesh_z_cd.shape[0])
        ct_data_3d_6[ct_data_3d_6<2.5] = 0
        ct_data_3d_6[ct_data_3d_6>=2.5] = 1
        self.contour3d_6 = ct_data_3d_6

        ct_data_flat_7 = interp7(ct_mesh_flat)
        ct_data_3d_7 = ct_data_flat_7.reshape(mesh_y_cd.shape[0], mesh_x_cd.shape[0], mesh_z_cd.shape[0])
        ct_data_3d_7[ct_data_3d_7<2.5] = 0
        ct_data_3d_7[ct_data_3d_7>=2.5] = 1
        self.contour3d_7 = ct_data_3d_7

        ct_data_flat_8 = interp8(ct_mesh_flat)
        ct_data_3d_8 = ct_data_flat_8.reshape(mesh_y_cd.shape[0], mesh_x_cd.shape[0], mesh_z_cd.shape[0])
        ct_data_3d_8[ct_data_3d_8<3.5] = 0
        ct_data_3d_8[ct_data_3d_8>=3.5] = 1
        self.contour3d_8 = ct_data_3d_8

        ct_data_flat_9 = interp9(ct_mesh_flat)
        ct_data_3d_9 = ct_data_flat_9.reshape(mesh_y_cd.shape[0], mesh_x_cd.shape[0], mesh_z_cd.shape[0])
        ct_data_3d_9[ct_data_3d_9<3.5] = 0
        ct_data_3d_9[ct_data_3d_9>=3.5] = 1
        self.contour3d_9 = ct_data_3d_9

        ct_data_flat_10 = interp10(ct_mesh_flat)
        ct_data_3d_10 = ct_data_flat_10.reshape(mesh_y_cd.shape[0], mesh_x_cd.shape[0], mesh_z_cd.shape[0])
        ct_data_3d_10[ct_data_3d_10<6] = 0
        ct_data_3d_10[ct_data_3d_10>=6] = 1
        self.contour3d_10 = ct_data_3d_10



    def plot_contour_depth_slices(self, folder_name, slicePosition, plot_unit, range_east=None, range_north=None, figsize=(16,7), den_clim=(-0.3, 0.3), sus_clim=(-0.05, 0.05), tfs=25, lfs=22, axfs=22, cbfs=20, Mts_flag=True, PPs_flag=False, Ws_flag=False, gtWs_flag=False):
        '''
        Plot a depth slice for the recovered models with contours outline a chosen quasi-geology model unit (at chosen depth)
        slicePosition: the chosen depth to make depth slice 
        plot_unit: The unit (serial number) to be outlined with black contour on the slice 
        Mts_flag: plot marks for mountains or not
        PPs_flag: plot marks for power plants or not (for unit 2)
        Ws_flag: plot marks for 2 wells between Mt. Hannah and Ht. Konocti or not
        gtWs_flag: plot marks for geothermal wells or not(for unit 10)
        '''
        if range_east is None:
            range_east = (self.core_bounds[0],self.core_bounds[1])
        if range_north is None:
            range_north = (self.core_bounds[2],self.core_bounds[3])

        if plot_unit == 1:
            contour3d = self.contour3d_1
        elif plot_unit == 2:
            contour3d = self.contour3d_2
        elif plot_unit == 3:
            contour3d = self.contour3d_3
        elif plot_unit == 4:
            contour3d = self.contour3d_4
        elif plot_unit == 5:
            contour3d = self.contour3d_5
        elif plot_unit == 6:
            contour3d = self.contour3d_6
        elif plot_unit == 7:
            contour3d = self.contour3d_7
        elif plot_unit == 8:
            contour3d = self.contour3d_8
        elif plot_unit == 9:
            contour3d = self.contour3d_9
        elif plot_unit == 10:
            contour3d = self.contour3d_10


        sliceInd = int(round(np.searchsorted(self.mesh_rm.cell_centers_z, slicePosition))) - 1
        sliceInd_d = int(round(np.searchsorted(self.mesh_z_cd, slicePosition))) - 1

        # for ii in range(0, self.z_grid_array.shape[0]-1):
        for ii in range(sliceInd, sliceInd+1):
        # Plot Recovered Model
            slicePosition = self.z_center_array[ii]
            
            fig = plt.figure(figsize=figsize)
            # ax1 = fig.add_axes([0.04, 0.1, 0.4, 0.8])
            ax1 = plt.subplot(121)
            p1 = self.mesh_rm.plot_slice(
                self.model_dens_rm,
                normal="Z",
                ax=ax1,
                ind=int(ii),
                grid=False,
                # grid_opts={'linewidth': 0.05, 'color': 'black'},
                # clim=(np.min(recovered_model), 0.1),
                clim=den_clim,
                range_x=range_east,
                range_y=range_north,
                pcolor_opts={"cmap": "bwr"},
            )

            ax1.contour(self.XXd, self.YYd, contour3d[:,:,sliceInd_d], colors="black", levels=0, linewidths=4)

            if Mts_flag==True:
                mark1 = ax1.scatter(520360, 4313800, s=140, marker='^', c="black")
                mark1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot1 = ax1.annotate("  Mt. Konocti", (520360, 4313800), fontsize=21, color='black')
                annot1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                mark2 = ax1.scatter(522030, 4304100, s=140, marker='^', c="black")
                mark2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot2 = ax1.annotate("  Mt. Hannah", (522030, 4304100), fontsize=21, color='black')
                annot2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                mark3 = ax1.scatter(522490, 4295200, s=140, marker='^', c="black")
                mark3.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot3 = ax1.annotate("  Mt. Cobb", (522490, 4295200), fontsize=21, color='black')
                annot3.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])

            # Power plant coordinate
            PP_x = [516671, 517276.7, 517418.9, 518828.7, 519019.1, 520205.2, 521260.9, 521351.1, 522021.1, 522142.7, 523599.9, 524453, 525171.5, 525547.4]
            PP_y = [4295220.4, 4296264.5, 4297563.8, 4295150.8, 4297215.8, 4298445.5, 4293457.1, 4292320.2, 4291090.5, 4293503.5, 4291577.7, 4289257.5, 4288909.5, 4290928.1] 
            if PPs_flag==True:
                mpp1 = ax1.scatter(PP_x, PP_y, s=120, marker='X', c="magenta", zorder=10)
                mpp1.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])
            if Ws_flag==True:
                mws1 = ax1.scatter(521747, 4310720, s=120, marker='D', c="lime", zorder=10)
                mws1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                aws1 = ax1.annotate("   K1", (521747, 4310720), fontsize=18)
                aws1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                mws2 = ax1.scatter(521031, 4308744, s=120, marker='D', c="lime", zorder=10)
                mws2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                aws2 = ax1.annotate("   N1", (521031, 4308744), fontsize=18)
                aws2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
            if gtWs_flag==True:
                ax1.scatter(521676, 4300256, s=120, marker='X', c="white", zorder=10)
                agtw1 = ax1.annotate("  #18", (521676, 4300256), zorder=10, fontsize=15)
                agtw1.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])
                ax1.scatter(518011, 4304037, s=120, marker='X', c="white", zorder=10)
                agtw2 = ax1.annotate("#20", (518011, 4304037), zorder=10, fontsize=15, xytext=(-40, 0), textcoords='offset points')
                agtw2.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])
                ax1.scatter(519343, 4305767, s=120, marker='X', c="white", zorder=10)
                agtw3 = ax1.annotate("  #25", (519343, 4305767), zorder=10, fontsize=15, xytext=(0, -10), textcoords='offset points')
                agtw3.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])
                ax1.scatter(519494, 4306273, s=120, marker='X', c="white", zorder=10)
                agtw4 = ax1.annotate("  #26", (519494, 4306273), zorder=10, fontsize=15)
                agtw4.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])
                ax1.scatter(523328, 4302153, s=120, marker='X', c="white", zorder=10)
                agtw5 = ax1.annotate("  #28", (523328, 4302153), zorder=10, fontsize=15)
                agtw5.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])
                ax1.scatter(517000, 4307147, s=120, marker='X', c="white", zorder=10)
                agtw6 = ax1.annotate("  #29", (517000, 4307147), zorder=10, fontsize=15)
                agtw6.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])

            # ax1.set_title("Inverted density model, z = {0}~{1} [m]".format(self.z_grid_array[ii], self.z_grid_array[ii+1]), fontsize=tfs)
            ax1.set_title("Density, z: {0}[m]".format((self.z_grid_array[ii]+self.z_grid_array[ii+1])/2), fontsize=tfs, pad=15)
            ax1.set_aspect('equal') 
            ax1.set_xlabel("Easting [m]", fontsize=lfs)
            ax1.set_ylabel("Northing [m]", fontsize=lfs)
            ax1.ticklabel_format(style='sci', scilimits=[-2,2], useMathText=True)
            # ax1.get_yaxis().get_offset_text().set_position((-0.25,1.05))  # This is replaced by the lines in the end
            ax1.tick_params(direction='in', length=7, axis='both', which='major', labelsize=axfs, pad=10)
            ax1.tick_params(direction='in', length=3, axis='both', which='minor', labelsize=axfs, pad=10)
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(10000))
            ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax1.xaxis.get_offset_text().set_fontsize(axfs)  # For x-axis
            ax1.yaxis.get_offset_text().set_fontsize(axfs)  # For y-axis (if needed)
            cbar = plt.colorbar(p1[0], ax=ax1)
            cbar.set_label("$g/cm^3$", rotation=270, labelpad=15, size=cbfs)
            cbar.ax.tick_params(labelsize=cbfs)
            # !!!! This part is used to adjust the y postion (and x position) of
            # the exponent magnitude of y ticklable, beacuse the second parameter of
            # ax2.get_yaxis().get_offset_text().set_position((-0.25,1.1)) is not functional
            fig.canvas.draw()
            offset_text_obj = ax1.get_yaxis().get_offset_text()
            offset_str = offset_text_obj.get_text()
            offset_text_obj.set_visible(False)
            ax1.text(-0.25, 1.05, offset_str, transform=ax1.transAxes, fontsize=axfs)
            # ------------------------------------------

            # ax2 = fig.add_axes([0.54, 0.1, 0.4, 0.8])
            ax2 = plt.subplot(122)
            p2 = self.mesh_rm.plot_slice(
                self.model_susc_rm,
                normal="Z",
                ax=ax2,
                ind=int(ii),
                grid=False,
                # grid_opts={'linewidth': 0.05, 'color': 'black'},
                # clim=(np.min(recovered_model), 0.1),
                clim=sus_clim,
                range_x=range_east,
                range_y=range_north,
                pcolor_opts={"cmap": "bwr"},
            )

            ax2.contour(self.XXd, self.YYd, contour3d[:,:,sliceInd_d], colors="black", levels=0, linewidths=4)

            if Mts_flag==True:
                mark1 = ax2.scatter(520360, 4313800, s=140, marker='^', c="black")
                mark1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot1 = ax2.annotate("  Mt. Konocti", (520360, 4313800), fontsize=21, color='black')
                annot1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                mark2 = ax2.scatter(522030, 4304100, s=140, marker='^', c="black")
                mark2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot2 = ax2.annotate("  Mt. Hannah", (522030, 4304100), fontsize=21, color='black')
                annot2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                mark3 = ax2.scatter(522490, 4295200, s=140, marker='^', c="black")
                mark3.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot3 = ax2.annotate("  Mt. Cobb", (522490, 4295200), fontsize=21, color='black')
                annot3.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])

            # Power plant coordinate
            PP_x = [516671, 517276.7, 517418.9, 518828.7, 519019.1, 520205.2, 521260.9, 521351.1, 522021.1, 522142.7, 523599.9, 524453, 525171.5, 525547.4]
            PP_y = [4295220.4, 4296264.5, 4297563.8, 4295150.8, 4297215.8, 4298445.5, 4293457.1, 4292320.2, 4291090.5, 4293503.5, 4291577.7, 4289257.5, 4288909.5, 4290928.1] 
            if PPs_flag==True:
                mpp1 = ax2.scatter(PP_x, PP_y, s=120, marker='X', c="magenta", zorder=10)
                mpp1.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])
            if Ws_flag==True:
                mws1 = ax2.scatter(521747, 4310720, s=120, marker='D', c="lime", zorder=10)
                mws1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                aws1 = ax2.annotate("   K1", (521747, 4310720), fontsize=18)
                aws1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                mws2 = ax2.scatter(521031, 4308744, s=120, marker='D', c="lime", zorder=10)
                mws2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                aws2 = ax2.annotate("   N1", (521031, 4308744), fontsize=18)
                aws2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
            if gtWs_flag==True:
                ax2.scatter(521676, 4300256, s=120, marker='X', c="white", zorder=10)
                agtw1 = ax2.annotate("  #18", (521676, 4300256), zorder=10, fontsize=15)
                agtw1.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])
                ax2.scatter(518011, 4304037, s=120, marker='X', c="white", zorder=10)
                agtw2 = ax2.annotate("#20", (518011, 4304037), zorder=10, fontsize=15, xytext=(-40, 0), textcoords='offset points')
                agtw2.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])
                ax2.scatter(519343, 4305767, s=120, marker='X', c="white", zorder=10)
                agtw3 = ax2.annotate("  #25", (519343, 4305767), zorder=10, fontsize=15, xytext=(0, -10), textcoords='offset points')
                agtw3.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])
                ax2.scatter(519494, 4306273, s=120, marker='X', c="white", zorder=10)
                agtw4 = ax2.annotate("  #26", (519494, 4306273), zorder=10, fontsize=15)
                agtw4.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])
                ax2.scatter(523328, 4302153, s=120, marker='X', c="white", zorder=10)
                agtw5 = ax2.annotate("  #28", (523328, 4302153), zorder=10, fontsize=15)
                agtw5.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])
                ax2.scatter(517000, 4307147, s=120, marker='X', c="white", zorder=10)
                agtw6 = ax2.annotate("  #29", (517000, 4307147), zorder=10, fontsize=15)
                agtw6.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])

            # ax2.set_title("Inverted susceptibility model, z = {0}~{1} [m]".format(self.z_grid_array[ii], self.z_grid_array[ii+1]), fontsize=tfs)
            ax2.set_title("Susceptibility, z: {0}[m]".format((self.z_grid_array[ii]+self.z_grid_array[ii+1])/2), fontsize=tfs, pad=15)
            ax2.set_aspect('equal') 
            ax2.set_xlabel("Easting [m]", fontsize=lfs)
            ax2.set_ylabel("Northing [m]", fontsize=lfs)
            ax2.ticklabel_format(style='sci', scilimits=[-2,2], useMathText=True)
            # ax2.get_yaxis().get_offset_text().set_position((-0.25,1.05))   # This is replaced by the lines in the end
            ax2.tick_params(direction='in', length=7, axis='both', which='major', labelsize=axfs, pad=10)
            ax2.tick_params(direction='in', length=3, axis='both', which='minor', labelsize=axfs, pad=10)
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(10000))
            ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax2.xaxis.get_offset_text().set_fontsize(axfs)  # For x-axis
            ax2.yaxis.get_offset_text().set_fontsize(axfs)  # For y-axis (if needed)
            cbar = plt.colorbar(p2[0], ax=ax2)
            cbar.set_label("SI", rotation=270, labelpad=15, size=cbfs)
            cbar.ax.tick_params(labelsize=cbfs)
            # !!!! This part is used to adjust the y postion (and x position) of
            # the exponent magnitude of y ticklable, beacuse the second parameter of
            # ax2.get_yaxis().get_offset_text().set_position((-0.25,1.1)) is not functional
            fig.canvas.draw()
            offset_text_obj = ax2.get_yaxis().get_offset_text()
            offset_str = offset_text_obj.get_text()
            offset_text_obj.set_visible(False)
            ax2.text(-0.25, 1.05, offset_str, transform=ax2.transAxes, fontsize=axfs)
            # ------------------------------------------
            plt.tight_layout()          
            plt.savefig("./temp_inv_out/" + folder_name + "/saved_figures/z_contour/Z_slice_ct_u{0}_{1}m.svg".format(plot_unit, slicePosition))
            plt.close()





    def plot_qusi_depth_slices(self, folder_name, slicePosition, plot_unit, range_east=None, range_north=None, figsize=(16,7), tfs=25, lfs=22, axfs=22, cbfs=20, Mts_flag=True, PPs_flag=False, Ws_flag=False, gtWs_flag=False):
        '''
        Plot a depth slice for the quasi-geology model with contours outline a chosen quasi-geology model unit (at chosen depth)
        slicePosition: the chosen depth to make depth slice 
        plot_unit: The unit (serial number) to be outlined with black contour on the slice 
        Mts_flag: plot marks for mountains or not
        PPs_flag: plot marks for power plants or not (for unit 2)
        Ws_flag: plot marks for 2 wells between Mt. Hannah and Ht. Konocti or not
        gtWs_flag: plot marks for geothermal wells or not(for unit 10)
        '''
        if range_east is None:
            range_east = (self.core_bounds[0],self.core_bounds[1])
        if range_north is None:
            range_north = (self.core_bounds[2],self.core_bounds[3])

        if plot_unit == 1:
            contour3d = self.contour3d_1
        elif plot_unit == 2:
            contour3d = self.contour3d_2
        elif plot_unit == 3:
            contour3d = self.contour3d_3
        elif plot_unit == 4:
            contour3d = self.contour3d_4
        elif plot_unit == 5:
            contour3d = self.contour3d_5
        elif plot_unit == 6:
            contour3d = self.contour3d_6
        elif plot_unit == 7:
            contour3d = self.contour3d_7
        elif plot_unit == 8:
            contour3d = self.contour3d_8
        elif plot_unit == 9:
            contour3d = self.contour3d_9
        elif plot_unit == 10:
            contour3d = self.contour3d_10


        sliceInd = int(round(np.searchsorted(self.mesh_rm.cell_centers_z, slicePosition))) - 1
        sliceInd_d = int(round(np.searchsorted(self.mesh_z_cd, slicePosition))) - 1

        # for ii in range(0, self.z_grid_array.shape[0]-1):
        for ii in range(sliceInd, sliceInd+1):
        # Plot Recovered Model
            slicePosition = self.z_center_array[ii]
            
            fig = plt.figure(figsize=figsize)
            # ax1 = fig.add_axes([0.04, 0.1, 0.4, 0.8])
            ax1 = plt.subplot(121)
            p1 = self.mesh_rm.plot_slice(
                self.model_unit_3d,
                normal="Z",
                ax=ax1,
                ind=int(ii),
                grid=False,
                # grid_opts={'linewidth': 0.05, 'color': 'black'},
                # clim=(np.min(recovered_model), 0.1),
                clim=(0, max(self.vmax,1)),
                range_x=range_east,
                range_y=range_north,
                pcolor_opts={"cmap":"RdYlBu_r"},
            )

            ax1.contour(self.XXd, self.YYd, contour3d[:,:,sliceInd_d], colors="black", levels=0, linewidths=4)

            if Mts_flag==True:
                mark1 = ax1.scatter(520360, 4313800, s=140, marker='^', c="black")
                mark1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot1 = ax1.annotate("  Mt. Konocti", (520360, 4313800), fontsize=21, color='black')
                annot1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                mark2 = ax1.scatter(522030, 4304100, s=140, marker='^', c="black")
                mark2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot2 = ax1.annotate("  Mt. Hannah", (522030, 4304100), fontsize=21, color='black')
                annot2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                mark3 = ax1.scatter(522490, 4295200, s=140, marker='^', c="black")
                mark3.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot3 = ax1.annotate("  Mt. Cobb", (522490, 4295200), fontsize=21, color='black')
                annot3.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])


            # Power plant coordinate
            PP_x = [516671, 517276.7, 517418.9, 518828.7, 519019.1, 520205.2, 521260.9, 521351.1, 522021.1, 522142.7, 523599.9, 524453, 525171.5, 525547.4]
            PP_y = [4295220.4, 4296264.5, 4297563.8, 4295150.8, 4297215.8, 4298445.5, 4293457.1, 4292320.2, 4291090.5, 4293503.5, 4291577.7, 4289257.5, 4288909.5, 4290928.1] 
            if PPs_flag==True:
                mpp1 = ax1.scatter(PP_x, PP_y, s=120, marker='X', c="magenta", zorder=10)
                mpp1.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])
            if Ws_flag==True:
                mws1 = ax1.scatter(521747, 4310720, s=120, marker='D', c="lime", zorder=10)
                mws1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                aws1 = ax1.annotate("   K1", (521747, 4310720), fontsize=18)
                aws1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                mws2 = ax1.scatter(521031, 4308744, s=120, marker='D', c="lime", zorder=10)
                mws2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                aws2 = ax1.annotate("   N1", (521031, 4308744), fontsize=18)
                aws2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
            if gtWs_flag==True:
                ax1.scatter(521676, 4300256, s=120, marker='X', c="white", zorder=10)
                agtw1 = ax1.annotate("  #18", (521676, 4300256), zorder=10, fontsize=15)
                agtw1.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])
                ax1.scatter(518011, 4304037, s=120, marker='X', c="white", zorder=10)
                agtw2 = ax1.annotate("#20", (518011, 4304037), zorder=10, fontsize=15, xytext=(-40, 0), textcoords='offset points')
                agtw2.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])
                ax1.scatter(519343, 4305767, s=120, marker='X', c="white", zorder=10)
                agtw3 = ax1.annotate("  #25", (519343, 4305767), zorder=10, fontsize=15, xytext=(0, -10), textcoords='offset points')
                agtw3.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])
                ax1.scatter(519494, 4306273, s=120, marker='X', c="white", zorder=10)
                agtw4 = ax1.annotate("  #26", (519494, 4306273), zorder=10, fontsize=15)
                agtw4.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])
                ax1.scatter(523328, 4302153, s=120, marker='X', c="white", zorder=10)
                agtw5 = ax1.annotate("  #28", (523328, 4302153), zorder=10, fontsize=15)
                agtw5.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])
                ax1.scatter(517000, 4307147, s=120, marker='X', c="white", zorder=10)
                agtw6 = ax1.annotate("  #29", (517000, 4307147), zorder=10, fontsize=15)
                agtw6.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()])


            # ax1.set_title("Inverted density model, z = {0}~{1} [m]".format(self.z_grid_array[ii], self.z_grid_array[ii+1]), fontsize=tfs)
            ax1.set_title("Units, z: {0}[m]".format((self.z_grid_array[ii]+self.z_grid_array[ii+1])/2), fontsize=tfs, pad=15)
            ax1.set_aspect('equal') 
            ax1.set_xlabel("Easting [m]", fontsize=lfs)
            ax1.set_ylabel("Northing [m]", fontsize=lfs)
            ax1.ticklabel_format(style='sci', scilimits=[-2,2], useMathText=True)
            # ax1.get_yaxis().get_offset_text().set_position((-0.25,1.05))  # This is replaced by the lines in the end
            ax1.tick_params(direction='in', length=7, axis='both', which='major', labelsize=axfs, pad=10)
            ax1.tick_params(direction='in', length=3, axis='both', which='minor', labelsize=axfs, pad=10)
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(10000))
            ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax1.xaxis.get_offset_text().set_fontsize(axfs)  # For x-axis
            ax1.yaxis.get_offset_text().set_fontsize(axfs)  # For y-axis (if needed)
            cbar = plt.colorbar(p1[0], ax=ax1)
            cbar.set_label("$Units$", rotation=270, labelpad=15, size=cbfs)
            cbar.ax.tick_params(labelsize=cbfs)
            # !!!! This part is used to adjust the y postion (and x position) of
            # the exponent magnitude of y ticklable, beacuse the second parameter of
            # ax2.get_yaxis().get_offset_text().set_position((-0.25,1.1)) is not functional
            fig.canvas.draw()
            offset_text_obj = ax1.get_yaxis().get_offset_text()
            offset_str = offset_text_obj.get_text()
            offset_text_obj.set_visible(False)
            ax1.text(-0.25, 1.05, offset_str, transform=ax1.transAxes, fontsize=axfs)
            # ------------------------------------------

            # ax2 = fig.add_axes([0.54, 0.1, 0.4, 0.8])
            ax2 = plt.subplot(122)
            p2 = self.mesh_rm.plot_slice(
                self.model_unit_3d,
                normal="Z",
                ax=ax2,
                ind=int(ii),
                grid=False,
                # grid_opts={'linewidth': 0.05, 'color': 'black'},
                # clim=(np.min(recovered_model), 0.1),
                clim=(0, max(self.vmax,1)),
                range_x=range_east,
                range_y=range_north,
                pcolor_opts={"cmap":"RdYlBu_r"},
            )

            ax2.contour(self.XXd, self.YYd, contour3d[:,:,sliceInd_d], colors="black", levels=0, linewidths=4)

            if Mts_flag==True:
                mark1 = ax2.scatter(520360, 4313800, s=140, marker='^', c="black")
                mark1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot1 = ax2.annotate("  Mt. Konocti", (520360, 4313800), fontsize=21, color='black')
                annot1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                mark2 = ax2.scatter(522030, 4304100, s=140, marker='^', c="black")
                mark2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot2 = ax2.annotate("  Mt. Hannah", (522030, 4304100), fontsize=21, color='black')
                annot2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                mark3 = ax2.scatter(522490, 4295200, s=140, marker='^', c="black")
                mark3.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])
                annot3 = ax2.annotate("  Mt. Cobb", (522490, 4295200), fontsize=21, color='black')
                annot3.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                        path_effects.Normal()])

            # ax2.set_title("Inverted susceptibility model, z = {0}~{1} [m]".format(self.z_grid_array[ii], self.z_grid_array[ii+1]), fontsize=tfs)
            ax2.set_title("Units, z: {0}[m]".format((self.z_grid_array[ii]+self.z_grid_array[ii+1])/2), fontsize=tfs, pad=15)
            ax2.set_aspect('equal') 
            ax2.set_xlabel("Easting [m]", fontsize=lfs)
            ax2.set_ylabel("Northing [m]", fontsize=lfs)
            ax2.ticklabel_format(style='sci', scilimits=[-2,2], useMathText=True)
            # ax2.get_yaxis().get_offset_text().set_position((-0.25,1.05))   # This is replaced by the lines in the end
            ax2.tick_params(direction='in', length=7, axis='both', which='major', labelsize=axfs, pad=10)
            ax2.tick_params(direction='in', length=3, axis='both', which='minor', labelsize=axfs, pad=10)
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(10000))
            ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax2.xaxis.get_offset_text().set_fontsize(axfs)  # For x-axis
            ax2.yaxis.get_offset_text().set_fontsize(axfs)  # For y-axis (if needed)
            cbar = plt.colorbar(p2[0], ax=ax2)
            cbar.set_label("$Units$", rotation=270, labelpad=15, size=cbfs)
            cbar.ax.tick_params(labelsize=cbfs)
            # !!!! This part is used to adjust the y postion (and x position) of
            # the exponent magnitude of y ticklable, beacuse the second parameter of
            # ax2.get_yaxis().get_offset_text().set_position((-0.25,1.1)) is not functional
            fig.canvas.draw()
            offset_text_obj = ax2.get_yaxis().get_offset_text()
            offset_str = offset_text_obj.get_text()
            offset_text_obj.set_visible(False)
            ax2.text(-0.25, 1.05, offset_str, transform=ax2.transAxes, fontsize=axfs)
            # ------------------------------------------
            plt.tight_layout()          
            plt.savefig("./temp_inv_out/" + folder_name + "/saved_figures/z_contour/Z_slice_qusi_u{0}_{1}m.svg".format(plot_unit, slicePosition))
            plt.close()





    def plot_contour_northing_slices(self, folder_name, slicePosition, plot_unit, range_east=None, figsize=(8.5,9.1), den_clim=(-0.3, 0.3), sus_clim=(-0.05, 0.05), tfs=25, lfs=22, axfs=22, cbfs=20):
        '''
        Plot a vertical slice (along easting direction) for the recovered models with contours outline a chosen quasi-geology model unit (at chosen northing grid)
        slicePosition: the chosen northing coordinate to make vertical slice 
        plot_unit: The unit (serial number) to be outlined with black contour on the slice 
        '''
        if range_east is None:
            range_east = (self.core_bounds[0],self.core_bounds[1])

        if plot_unit == 1:
            contour3d = self.contour3d_1
        elif plot_unit == 2:
            contour3d = self.contour3d_2
        elif plot_unit == 3:
            contour3d = self.contour3d_3
        elif plot_unit == 4:
            contour3d = self.contour3d_4
        elif plot_unit == 5:
            contour3d = self.contour3d_5
        elif plot_unit == 6:
            contour3d = self.contour3d_6
        elif plot_unit == 7:
            contour3d = self.contour3d_7
        elif plot_unit == 8:
            contour3d = self.contour3d_8
        elif plot_unit == 9:
            contour3d = self.contour3d_9
        elif plot_unit == 10:
            contour3d = self.contour3d_10

        sliceInd = int(round(np.searchsorted(self.mesh_rm.cell_centers_y, slicePosition)))
        sliceInd_d = int(round(np.searchsorted(self.mesh_y_cd, slicePosition)))+1


        # for ii in range(0, self.z_grid_array.shape[0]-1):
        for ii in range(sliceInd, sliceInd+1):
        # Plot Recovered Model
            slicePosition = self.y_center_array[ii]
            fig = plt.figure(figsize=figsize)
            ax1 = plt.subplot(211)
            (im,) = self.mesh_rm.plot_slice(
                self.model_dens_rm,
                normal="Y",
                ax=ax1,
                ind=int(ii),
                range_x=range_east,
                clim=den_clim,
                pcolor_opts={"cmap": "bwr"}
            )

            ax1.contour(self.Xd.T, self.Zd.T, contour3d[sliceInd_d,:,:], colors="black", levels=0, linewidths=4)

            # ax1.set_title("Inverted density model, y = {0}~{1} [m]".format(self.y_grid_array[ii], self.y_grid_array[ii+1]), fontsize=tfs)
            ax1.set_title("Density".format((self.y_grid_array[ii]+self.y_grid_array[ii+1])/2/1000), fontsize=tfs, pad=15)
            ax1.set_aspect('equal') 
            ax1.set_xlabel("Easting [m]", fontsize=lfs)
            ax1.set_ylabel("Elevation [m]", fontsize=lfs)
            ax1.ticklabel_format(style='sci', axis='x', scilimits=[-2,2], useMathText=True)
            ax1.ticklabel_format(style='sci', axis='y', scilimits=[-2,2], useMathText=True)
            ax1.get_yaxis().get_offset_text().set_position((-0.20,0))
            ax1.tick_params(direction='in', length=7, axis='both', which='major', labelsize=axfs, pad=10)
            ax1.tick_params(direction='in', length=3, axis='both', which='minor', labelsize=axfs, pad=10)
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(10000))
            ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax1.xaxis.get_offset_text().set_fontsize(axfs)  # For x-axis
            ax1.yaxis.get_offset_text().set_fontsize(axfs)  # For y-axis (if needed)
            cbar = plt.colorbar(im)
            cbar.set_label("$g/cm^3$", rotation=270, labelpad=15, size=cbfs)
            cbar.ax.tick_params(labelsize=cbfs)

            ax2 = plt.subplot(212)
            (im,) = self.mesh_rm.plot_slice(
                self.model_susc_rm,
                normal="Y",
                ax=ax2,
                ind=int(ii),
                range_x=range_east,
                clim=sus_clim,
                pcolor_opts={"cmap": "bwr"}
            )

            ax2.contour(self.Xd.T, self.Zd.T, contour3d[sliceInd_d,:,:], colors="black", levels=0, linewidths=4)

            # ax2.set_title("Inverted susceptibility model, y = {0}~{1} [m]".format(self.y_grid_array[ii], self.y_grid_array[ii+1]), fontsize=tfs)
            ax2.set_title("Susceptibility".format((self.y_grid_array[ii]+self.y_grid_array[ii+1])/2/1000), fontsize=tfs, pad=15)
            ax2.set_aspect('equal')
            ax2.set_xlabel("Easting [m]", fontsize=lfs)
            ax2.set_ylabel("Elevation [m]", fontsize=lfs)
            ax2.ticklabel_format(style='sci', axis='x', scilimits=[-2,2], useMathText=True)
            ax2.ticklabel_format(style='sci', axis='y', scilimits=[-2,2], useMathText=True)
            ax2.get_yaxis().get_offset_text().set_position((-0.20,0))
            ax2.tick_params(direction='in', length=7, axis='both', which='major', labelsize=axfs, pad=10)
            ax2.tick_params(direction='in', length=3, axis='both', which='minor', labelsize=axfs, pad=10)
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(10000))
            ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax2.xaxis.get_offset_text().set_fontsize(axfs)  # For x-axis
            ax2.yaxis.get_offset_text().set_fontsize(axfs)  # For y-axis (if needed)
            cbar = plt.colorbar(im)
            cbar.set_label("SI", rotation=270, labelpad=15, size=cbfs)
            cbar.ax.tick_params(labelsize=cbfs)

            plt.tight_layout()
            plt.savefig("./temp_inv_out/" + folder_name + "/saved_figures/y_contour/Y_slice_ct_u{0}_{1}m.svg".format(plot_unit, slicePosition))
            plt.close()






    def plot_qusi_northing_slices(self, folder_name, slicePosition, plot_unit, range_east=None, figsize=(8.5,9.1), den_clim=(-0.3, 0.3), sus_clim=(-0.05, 0.05), tfs=25, lfs=22, axfs=22, cbfs=20):
        '''
        Plot a vertical slice (along easting direction) for the quasi-geology model with contours outline a chosen quasi-geology model unit (at chosen northing grid)
        slicePosition: the chosen northing coordinate to make vertical slice 
        plot_unit: The unit (serial number) to be outlined with black contour on the slice 
        '''
        if range_east is None:
            range_east = (self.core_bounds[0],self.core_bounds[1])

        if plot_unit == 1:
            contour3d = self.contour3d_1
        elif plot_unit == 2:
            contour3d = self.contour3d_2
        elif plot_unit == 3:
            contour3d = self.contour3d_3
        elif plot_unit == 4:
            contour3d = self.contour3d_4
        elif plot_unit == 5:
            contour3d = self.contour3d_5
        elif plot_unit == 6:
            contour3d = self.contour3d_6
        elif plot_unit == 7:
            contour3d = self.contour3d_7
        elif plot_unit == 8:
            contour3d = self.contour3d_8
        elif plot_unit == 9:
            contour3d = self.contour3d_9
        elif plot_unit == 10:
            contour3d = self.contour3d_10

        sliceInd = int(round(np.searchsorted(self.mesh_rm.cell_centers_y, slicePosition)))
        sliceInd_d = int(round(np.searchsorted(self.mesh_y_cd, slicePosition)))+1


        # for ii in range(0, self.y_grid_array.shape[0]-1):
        for ii in range(sliceInd, sliceInd+1):
        # Plot Recovered Model
            slicePosition = self.y_center_array[ii]
            fig = plt.figure(figsize=figsize)
            ax1 = plt.subplot(211)
            (im,) = self.mesh_rm.plot_slice(
                self.model_unit_3d,
                normal="Y",
                ax=ax1,
                ind=int(ii),
                range_x=range_east,
                clim=(0, max(self.vmax,1)),
                pcolor_opts={"cmap":"RdYlBu_r"}
            )

            ax1.contour(self.Xd.T, self.Zd.T, contour3d[sliceInd_d,:,:], colors="black", levels=0, linewidths=4)

            # ax1.set_title("Inverted density model, y = {0}~{1} [m]".format(self.y_grid_array[ii], self.y_grid_array[ii+1]), fontsize=tfs)
            ax1.set_title("Units".format((self.y_grid_array[ii]+self.y_grid_array[ii+1])/2/1000), fontsize=tfs, pad=15)
            ax1.set_aspect('equal') 
            ax1.set_xlabel("Easting [m]", fontsize=lfs)
            ax1.set_ylabel("Elevation [m]", fontsize=lfs)
            ax1.ticklabel_format(style='sci', axis='x', scilimits=[-2,2], useMathText=True)
            ax1.ticklabel_format(style='sci', axis='y', scilimits=[-2,2], useMathText=True)
            ax1.get_yaxis().get_offset_text().set_position((-0.20,0))
            ax1.tick_params(direction='in', length=7, axis='both', which='major', labelsize=axfs, pad=10)
            ax1.tick_params(direction='in', length=3, axis='both', which='minor', labelsize=axfs, pad=10)
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(10000))
            ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax1.xaxis.get_offset_text().set_fontsize(axfs)  # For x-axis
            ax1.yaxis.get_offset_text().set_fontsize(axfs)  # For y-axis (if needed)
            cbar = plt.colorbar(im)
            cbar.set_label("$Units$", rotation=270, labelpad=15, size=cbfs)
            cbar.ax.tick_params(labelsize=cbfs)

            ax2 = plt.subplot(212)
            (im,) = self.mesh_rm.plot_slice(
                self.model_unit_3d,
                normal="Y",
                ax=ax2,
                ind=int(ii),
                range_x=range_east,
                clim=(0, max(self.vmax,1)),
                pcolor_opts={"cmap":"RdYlBu_r"}
            )

            ax2.contour(self.Xd.T, self.Zd.T, contour3d[sliceInd_d,:,:], colors="black", levels=0, linewidths=4)

            # ax2.set_title("Inverted susceptibility model, y = {0}~{1} [m]".format(self.y_grid_array[ii], self.y_grid_array[ii+1]), fontsize=tfs)
            ax2.set_title("Units".format((self.y_grid_array[ii]+self.y_grid_array[ii+1])/2/1000), fontsize=tfs, pad=15)
            ax2.set_aspect('equal')
            ax2.set_xlabel("Easting [m]", fontsize=lfs)
            ax2.set_ylabel("Elevation [m]", fontsize=lfs)
            ax2.ticklabel_format(style='sci', axis='x', scilimits=[-2,2], useMathText=True)
            ax2.ticklabel_format(style='sci', axis='y', scilimits=[-2,2], useMathText=True)
            ax2.get_yaxis().get_offset_text().set_position((-0.20,0))
            ax2.tick_params(direction='in', length=7, axis='both', which='major', labelsize=axfs, pad=10)
            ax2.tick_params(direction='in', length=3, axis='both', which='minor', labelsize=axfs, pad=10)
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(10000))
            ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1000))
            ax2.xaxis.get_offset_text().set_fontsize(axfs)  # For x-axis
            ax2.yaxis.get_offset_text().set_fontsize(axfs)  # For y-axis (if needed)
            cbar = plt.colorbar(im)
            cbar.set_label("Units", rotation=270, labelpad=15, size=cbfs)
            cbar.ax.tick_params(labelsize=cbfs)

            plt.tight_layout()
            plt.savefig("./temp_inv_out/" + folder_name + "/saved_figures/y_contour/Y_slice_qusi_u{0}_{1}m.svg".format(plot_unit, slicePosition))
            plt.close()































    def plot_topography(self, folder_name, range_east=None, range_north=None, figsize=(18,7), tfs=25, lfs=22, axfs=22, cbfs=20, Mts_flag=True):
        return






    def init_profile_interpolate(self):
        return


