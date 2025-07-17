import os
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import time
import h5py
from datetime import datetime, timezone, timedelta

import discretize
from discretize import TensorMesh
from discretize.utils import active_from_xyz
from SimPEG.potential_fields import gravity, magnetics
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
np.random.seed(0)



## Timestamp & Folder name
# Get the current time in UTC
now_utc = datetime.now(timezone.utc)
# Define the UTC-5 offset
utc_offset = timedelta(hours=-5)
# Adjust the current time to UTC-5
now_utc_minus_5 = now_utc + utc_offset
# Format the time string
timestamp = now_utc_minus_5.strftime("%m%d%y_%H%M%S")

mesh_name = "mesh_joint_Hannah_exfine"
folder_name = mesh_name + "_" + timestamp

mkdir_list = ["./temp_inv_out/",
              "./temp_inv_out/" + folder_name + "/",
              "./temp_inv_out/" + folder_name + "/saved_model/",
              "./temp_inv_out/" + folder_name + "/saved_data/"
              ]
for dirs in mkdir_list:
    if os.path.exists(dirs):
        print("{0:s} already exists! Skip creating this directory.".format(dirs))
    else:
        os.mkdir(dirs)
        print("Directory {0:s} created.".format(dirs))
        
## Read mesh
mesh = TensorMesh._readUBC_3DMesh("./mesh/" + mesh_name + ".txt")
mesh_rm = TensorMesh._readUBC_3DMesh("./mesh/" + mesh_name + "_rm.txt")
select_region = [510000, 535000, 4290000, 4320000]  # min_east, max_east, min_north, max_north
xpad = 6
ypad = 6

## Adjustable parameters
magDSrate = 10
std_grv = 0.25  # mGal
std_mag = 10  # nT 
(grv_lb, mag_lb, grv_ub, mag_ub) = (-10.0, -10.0, 10.0, 10.0)

grv_alpha_s = 20  # ~~~~~~~~~
grv_alpha_x = 1
grv_alpha_y = 1
grv_alpha_z = 1
mag_alpha_s = 10  # ~~~~~~~~~
mag_alpha_x = 1
mag_alpha_y = 1
mag_alpha_z = 1

reg_grv_norm = [1,2,2,2]  # ~~~~~~~~~
reg_mag_norm = [1,2,2,2]  # ~~~~~~~~~
# reg_grv_norm = [0,0,0,0]  # ~~~~~~~~~
# reg_mag_norm = [0,0,0,0]  # ~~~~~~~~~

maxGNCG = 100  # ~~~~~~~~~
maxLS = 10
maxCG = 1000
tolCG = 1e-2
tolX = 1e-2
maxIRLSiter = 100  # ~~~~~~~~~
IRLSstart = 5e4
IRLS_mindelta = 1e-2
IRLSbeta_tol = 1e-2

beta0_ratio = 10
betacool = 1.1

CGlambda = 1e12  # weight for coupling term  # ~~~~~~~~~



## Load topography data
df_topo = pd.read_csv("./data/Cali_topo_meter_m12315m12130_3840.csv", names=["easting", "northing", "topo"])  # TFMA: total field magnetic anomaly [nT]
easting_topo_raw = np.array(df_topo['easting'])
northing_topo_raw = np.array(df_topo['northing'])
topo_raw = np.array(df_topo['topo'])

select_index_topo = np.where(
    (easting_topo_raw>select_region[0]-30000) &
    (easting_topo_raw<select_region[1]+30000) &
    (northing_topo_raw>select_region[2]-30000) &
    (northing_topo_raw<select_region[3]+30000))[0]
easting_topo = easting_topo_raw[select_index_topo]
northing_topo = northing_topo_raw[select_index_topo]
topo = topo_raw[select_index_topo]

topo_xyz = np.zeros((easting_topo.shape[0], 3))
topo_xyz[:, 0] = easting_topo
topo_xyz[:, 1] = northing_topo
topo_xyz[:, 2] = topo

## Load gravity data
df_grv = pd.read_csv("./data/Mitchell_2022_CLVF_Gravity-main/groundGrav_Combined_zEllipsoid_Full.csv")

easting_grv_raw = np.array(df_grv['xWGS84_UTM10N'])
northing_grv_raw = np.array(df_grv['yWGS84_UTM10N'])
elevation_grv_raw = np.array(df_grv['zWGS84'])
iso_grv_raw = np.array(df_grv['ISO'])

select_index = np.where((easting_grv_raw>select_region[0]) & (easting_grv_raw<select_region[1]) & (northing_grv_raw>select_region[2]) & (northing_grv_raw<select_region[3]))[0]

easting_grv = easting_grv_raw[select_index]
northing_grv = northing_grv_raw[select_index]
elevation_grv = elevation_grv_raw[select_index]
iso_grv = iso_grv_raw[select_index]

data_grv_ori = np.zeros((easting_grv.shape[0], 4))
data_grv_ori[:, 0] = easting_grv
data_grv_ori[:, 1] = northing_grv
data_grv_ori[:, 2] = elevation_grv
data_grv_ori[:, 3] = iso_grv  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

data_grv = np.zeros((easting_grv.shape[0], 4))
data_grv[:, 0] = easting_grv
data_grv[:, 1] = northing_grv
data_grv[:, 2] = elevation_grv
data_grv[:, 3] = (-1) * iso_grv  # Considering SimPEG is using a opposite +z direction (+z upwards in SimPEG)

## Load magnetic data
df_mag = pd.read_csv("./data/mag_grid_meter_NAD27.csv")  # TFMA: total field magnetic anomaly [nT]

easting_mag_raw = np.array(df_mag['easting'])
northing_mag_raw = np.array(df_mag['northing'])
tfma_mag_raw = np.array(df_mag['TFMA'])

select_index = np.where((easting_mag_raw>select_region[0]) & (easting_mag_raw<select_region[1]) & (northing_mag_raw>select_region[2]) & (northing_mag_raw<select_region[3]))[0]

easting_mag = easting_mag_raw[select_index]
northing_mag = northing_mag_raw[select_index]
tfma_mag = tfma_mag_raw[select_index]


flight_h = 1000 / 3.2808399
data_mag_ori = np.zeros((easting_mag.shape[0], 4))
data_mag_ori[:, 0] = easting_mag
data_mag_ori[:, 1] = northing_mag
data_mag_ori[:, 3] = tfma_mag
# Intepolate aeromagnetic receiver elevation according to topo data
interp_rec = griddata(topo_xyz[:, 0:2], topo_xyz[:, 2], data_mag_ori[:, 0:2], method='linear')
data_mag_ori[:, 2] = interp_rec + flight_h

if np.sum(np.isnan(data_mag_ori[:, 2])) > 0:
    print("NaNs exist in the interpolated receiver elevation!")
    nan_ind = np.argwhere(np.isnan(data_mag_ori[:, 2]))
    data_mag_ori[nan_ind, 2] = 1200

data_mag_temp = data_mag_ori.copy()
data_mag = data_mag_temp[::magDSrate,:]



## Defining the Survey
# Define the receivers. The data consist of vertical gravity anomaly measurements.
# The set of receivers must be defined as a list.
receiver_grv = gravity.receivers.Point(data_grv[:,0:3], components="gz")

# Define the source field and survey for gravity data
source_field_grv = gravity.sources.SourceField(receiver_list=[receiver_grv])
survey_grv = gravity.survey.Survey(source_field_grv)

# Define the component(s) of the field we want to simulate as a list of strings.
# Here we simulation total magnetic intensity data.
# Use the observation locations and components to define the receivers. To
# simulate data, the receivers must be defined as a list.
receiver_mag = magnetics.receivers.Point(data_mag[:,0:3], components="tmi")

# Define the inducing field H0 = (intensity [nT], inclination [deg], declination [deg])
inclination = 62
declination = 15
strength = 50686

# Define the source field and survey for gravity data
source_field_mag = magnetics.sources.UniformBackgroundField(
    receiver_list=[receiver_mag],
    amplitude=strength,
    inclination=inclination,
    declination=declination,
)
survey_mag = magnetics.survey.Survey(source_field_mag)

## Defining the Data
maximum_anomaly_grv = np.max(np.abs(data_grv[:,3]))
uncertainties_grv = std_grv * np.ones(np.shape(data_grv[:,3]))

maximum_anomaly_mag = np.max(np.abs(data_mag[:,3]))
uncertainties_mag = std_mag * np.ones(np.shape(data_mag[:,3]))

data_object_grv = data.Data(
    survey_grv, dobs=data_grv[:,3], standard_deviation=uncertainties_grv
)
data_object_mag = data.Data(
    survey_mag, dobs=data_mag[:,3], standard_deviation=uncertainties_mag
)

## Define Starting/Reference Model and Mapping on Tensor Mesh
# Define density contrast values for each unit in g/cc.
background_dens, background_susc = 1e-6, 1e-6

# Find the indicies of the active cells in forward model (ones below surface)
ind_active = active_from_xyz(mesh, topo_xyz)

# Define mapping from model to active cells
nC = int(ind_active.sum())
model_map = maps.IdentityMap(nP=nC)  # model consists of a value for each active cell

# Create Wires Map that maps from stacked models to individual model components
# m1 refers to density model, m2 refers to susceptibility
wires = maps.Wires(("density", nC), ("susceptibility", nC))

# Define and plot starting model
starting_model = np.r_[background_dens * np.ones(nC), background_susc * np.ones(nC)]

np.save("./temp_inv_out/" + folder_name + "/saved_data/data_grv_ori.npy",data_grv_ori)
np.save("./temp_inv_out/" + folder_name + "/saved_data/data_mag_ori.npy",data_mag_ori)
np.save("./temp_inv_out/" + folder_name + "/saved_data/topo_xyz.npy",topo_xyz)
np.save("./temp_inv_out/" + folder_name + "/saved_data/ind_active.npy",ind_active)
print("---------------------------------------------------")
print("Pre-processed isostatic gravity anomaly, total field magnetic anomaly, topography, and SimPEG active mesh cells are saved in ./temp_inv_out/{0:s}/saved_data directory".format(folder_name))
print("---------------------------------------------------")

## Define the Physics
simulation_grv = gravity.simulation.Simulation3DIntegral(
    survey=survey_grv,
    mesh=mesh,
    rhoMap=wires.density,
    ind_active=ind_active,
    engine="choclo",
)
simulation_mag = magnetics.simulation.Simulation3DIntegral(
    survey=survey_mag,
    mesh=mesh,
    model_type="scalar",
    chiMap=wires.susceptibility,
    ind_active=ind_active,
)

## Define the Inverse Problem
# Define the data misfit. Here the data misfit is the L2 norm of the weighted
# residual between the observed data and the data predicted for a given model.
# Within the data misfit, the residual between predicted and observed data are
# normalized by the data's standard deviation.
dmis_grv = data_misfit.L2DataMisfit(data=data_object_grv, simulation=simulation_grv)
dmis_mag = data_misfit.L2DataMisfit(data=data_object_mag, simulation=simulation_mag)

# Define the regularization (model objective function)
reg_grv = regularization.Sparse(
    mesh, active_cells=ind_active, mapping=wires.density,
    gradient_type = "components"
)
reg_mag = regularization.Sparse(
    mesh, active_cells=ind_active, mapping=wires.susceptibility,
    gradient_type = "components"
)

# Norms for regularization terms
reg_grv.norms = reg_grv_norm  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
reg_mag.norms = reg_mag_norm

# Weights for regularization terms
reg_grv.alpha_s = grv_alpha_s
reg_grv.alpha_x = grv_alpha_x
reg_grv.alpha_y = grv_alpha_y
reg_grv.alpha_z = grv_alpha_z
reg_mag.alpha_s = mag_alpha_s
reg_mag.alpha_x = mag_alpha_x
reg_mag.alpha_y = mag_alpha_y
reg_mag.alpha_z = mag_alpha_z

# Define the coupling term to connect two different physical property models
lamda = CGlambda # weight for coupling term
cross_grad = regularization.CrossGradient(mesh, wires, active_cells=ind_active)

# Combine data misfit and regularization
dmis = dmis_grv + dmis_mag
reg = reg_grv + reg_mag + lamda * cross_grad

# # Define how the optimization problem is solved. Here we will use a projected
# # Gauss-Newton approach that employs the conjugate gradient solver.
opt = optimization.ProjectedGNCG(
    maxIter=maxGNCG,
    lower=np.concatenate([grv_lb * np.ones(nC), mag_lb * np.ones(nC)], 0),
    upper=np.concatenate([grv_ub * np.ones(nC), mag_ub * np.ones(nC)], 0),
    maxIterLS=maxLS,
    maxIterCG=maxCG,
    tolCG=tolCG,
    tolX=tolX,
)

# Here we define the inverse problem that is to be solved
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

# Defining a starting value for the trade-off parameter (beta) between the data
# misfit and the regularization.
starting_beta = directives.PairedBetaEstimate_ByEig(beta0_ratio=beta0_ratio)
# starting_beta.n_pw_iter = 10

# Defines the directives for the IRLS regularization. This includes setting
# the cooling schedule for the trade-off parameter.
update_IRLS = directives.Update_IRLS(
    f_min_change=IRLS_mindelta,
    max_irls_iterations=maxIRLSiter,
    # coolEpsFact=1.5,
    beta_tol=IRLSbeta_tol,
    verbose=True
)
update_IRLS.start = IRLSstart

# Defining the fractional decrease in beta and the number of Gauss-Newton solves
# for each beta value.
beta_schedule = directives.PairedBetaSchedule(cooling_factor=betacool, cooling_rate=1)
joint_inv_dir = directives.SimilarityMeasureInversionDirective()
stopping = directives.MovingAndMultiTargetStopping(tol=1e-6)
sensitivity_weights = directives.UpdateSensitivityWeights(every_iteration=False)

# Updating the preconditionner if it is model dependent.
update_jacobi = directives.UpdatePreconditioner()

save_output = directives.SimilarityMeasureSaveOutputEveryIteration(directory="./temp_inv_out/" + folder_name + "/saved_model/", name='Output')
save_model = directives.SaveModelEveryIteration(directory="./temp_inv_out/" + folder_name + "/saved_model/", name='InversionModel')

# The directives are defined as a list.
directives_list = [
    joint_inv_dir,
    sensitivity_weights,
    starting_beta,
    # stopping,
    beta_schedule,
    update_IRLS,
    save_output,
    save_model,
    update_jacobi,
    # target_misfit
]

# Here we combine the inverse problem and the set of directives
inv = inversion.BaseInversion(inv_prob, directives_list)




# -------------------------------------------------------------------------------------------------
tic_inv = time.time()
# Run inversion
recovered_model = inv.run(starting_model)
print("The inversion runtime is: {0:.2f} [h]".format((time.time()-tic_inv)/3600))
# -------------------------------------------------------------------------------------------------



np.save("./temp_inv_out/" + folder_name + "/saved_model/recovered_model.npy", recovered_model)
np.save("./temp_inv_out/" + folder_name + "/saved_model/inv_prob.npy", inv_prob.dpred)
print("---------------------------------------------------")
print("Direct inversion output (both models) saved as ./temp_inv_out/{0:s}/saved_model/recovered_model.npy".format(folder_name))
print("Recovered data saved as ./temp_inv_out/{0:s}/saved_model/inv_prob.npy".format(folder_name))

# Save inverted models (1D, with padding) in UBC format for geology differentiation. (Pad 0 for inactive cells)
pad0_map = maps.InjectActiveCells(mesh, ind_active, 0)
mesh.write_model_UBC("./temp_inv_out/" + folder_name + "/saved_model/joint_dens_model_UBC.txt", pad0_map * (wires * recovered_model)[0])
mesh.write_model_UBC("./temp_inv_out/" + folder_name + "/saved_model/joint_susc_model_UBC.txt", pad0_map * (wires * recovered_model)[1])
print("---------------------------------------------------")
print("Jointly inverted models (with padding cells) saved as ./temp_inv_out/{0:s}/saved_model/joint_dens_model_UBC.txt and joint_susc_model_UBC.txt".format(folder_name))


# Load inverted models in UBC format and created models without padding (1D)
modelGD_dens = discretize.TensorMesh.read_model_UBC(mesh, file_name="./temp_inv_out/" + folder_name + "/saved_model/joint_dens_model_UBC.txt")
modelGD_susc = discretize.TensorMesh.read_model_UBC(mesh, file_name="./temp_inv_out/" + folder_name + "/saved_model/joint_susc_model_UBC.txt")
modelGD_dens_3d = np.reshape(
    modelGD_dens, (mesh.shape_cells[0],mesh.shape_cells[1],mesh.shape_cells[2]), 
    order="F"
    )
modelGD_susc_3d = np.reshape(
    modelGD_susc, (mesh.shape_cells[0],mesh.shape_cells[1],mesh.shape_cells[2]), 
    order="F"
    )
# remove padding cells
modelGD_dens_rm = modelGD_dens_3d[
    xpad:mesh.shape_cells[0]-xpad,
    ypad:mesh.shape_cells[1]-ypad,
    0:mesh.shape_cells[2] # 0 is deep part, mesh.shape_cells[2] is near surface layer
    ]
modelGD_dens_rm = discretize.utils.mkvc(modelGD_dens_rm)
modelGD_susc_rm = modelGD_susc_3d[
    xpad:mesh.shape_cells[0]-xpad,
    ypad:mesh.shape_cells[1]-ypad,
    0:mesh.shape_cells[2]
    ]
modelGD_susc_rm = discretize.utils.mkvc(modelGD_susc_rm)


# Save the inverted models without padding (1D) in UBC format
mesh_rm.write_model_UBC("./temp_inv_out/" + folder_name + "/saved_model/joint_dens_model_rm_UBC.txt", modelGD_dens_rm)
mesh_rm.write_model_UBC("./temp_inv_out/" + folder_name + "/saved_model/joint_susc_model_rm_UBC.txt", modelGD_susc_rm)
print("---------------------------------------------------")
print("Jointly inverted models (without padding cells) saved as ./temp_inv_out/{0:s}/saved_model/joint_dens_model_rm_UBC.txt and joint_susc_model_rm_UBC.txt".format(folder_name))

# Save other parameters
with h5py.File("./temp_inv_out/" + folder_name + "/saved_model/paras.h5", "w") as h5f:
    h5f.create_dataset("core_zone", data=np.array(select_region))
    h5f.attrs["xpad"] = xpad
    h5f.attrs["ypad"] = ypad
    h5f.attrs["magDSrate"] = magDSrate
    h5f.attrs["std_grv"] = std_grv
    h5f.attrs["std_mag"] = std_mag

    h5f.create_dataset("inv_bound", data=(grv_lb, mag_lb, grv_ub, mag_ub))
    h5f.create_dataset("weight_grv", data=(grv_alpha_s, grv_alpha_x, grv_alpha_y, grv_alpha_z))
    h5f.create_dataset("weight_mag", data=(mag_alpha_s, mag_alpha_x, mag_alpha_y, mag_alpha_z))
    h5f.create_dataset("reg_grv_norm", data=reg_grv_norm)
    h5f.create_dataset("reg_mag_norm", data=reg_mag_norm)
    
    h5f.attrs["maxGNCG"] = maxGNCG
    h5f.attrs["maxLS"] = maxLS
    h5f.attrs["maxCG"] = maxCG
    h5f.attrs["tolCG"] = tolCG
    h5f.attrs["tolX"] = tolX
    h5f.attrs["maxIRLSiter"] = maxIRLSiter
    h5f.attrs["IRLSstart"] = IRLSstart
    h5f.attrs["IRLS_mindelta"] = IRLS_mindelta
    h5f.attrs["IRLSbeta_tol"] = IRLSbeta_tol
    h5f.attrs["beta0_ratio"] = beta0_ratio
    h5f.attrs["betacool"] = betacool
    h5f.attrs["CGlambda"] = CGlambda
print("---------------------------------------------------")
print("Inversion parameters saved in ./temp_inv_out/{0:s}/saved_model/paras.h5".format(folder_name))
print("---------------------------------------------------")
print("Please use the plot_results.ipynb with mesh name ({0:s}) and folder_name ({1:s}) to plot the data and results.".format(mesh_name, folder_name))
