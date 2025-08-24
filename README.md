# Natural_Hydrogen_YS
<p>This repository contains the geophysical measurements, joint inversion scripts, and output geophysical/quasi-geology models for the article "<em>Natural hydrogen exploration by joint sparse inversion of geophysical measurements and integrated geological interpretation</em>" published on <em>International Journal of Hydrogen Energy</em>.</p>

<p>In this work, we developed a workflow to identify serpentinized ophiolite targets by 3D joint inversion of gravity and magnetic data followed by geology differentiation. These targets are interpreted to indicate the potential accumulation of natural hydrogen in the subsurface. </p>

<p>The script <kbd>joint_inversion.py</kbd> allows one to conduct 3D joint inversion in Clear Lake volcanic field, and the jupyter notebook <kbd>plot_results.ipynb</kbd> allows one to plot the geophysical measurements, jointly inverted models, and quasi-geology models as the results of geology differentiation.</p>

#### Authors:
<p><b>*Yawei Su</b> (ysu8@cougarnet.uh.edu, first author), University of Houston</p>
<p> Sihong Wu, Jiajia Sun, Xuqing Wu, Yueqin Huang, and Jiefu Chen, University of Houston; Ligang Lu, Shell Information Technology International, Inc.; Xiaolong Wei, Stanford University; Rodolfo Christiansen, Leibniz Institute for Applied Geophysics.</p>


## Recommend Environment

<p>Python 3.10.12 or later.</p>
<p>SimPEG 0.21.1 or later.</p>
<p>discretize 0.10.0 or later.</p>
<p>numpy 1.24.4 or later.</p>
<p>scipy 1.11.4 or later.</p>
<p>matplolib 3.9.0 or later.</p>

## Instructions
<p>**Original datasets are not included in this repository to save space. In order to run joint inversion, please access them through Google Drive to get either <kbd>data.zip</kbd> or <kbd>data.tar.gz</kbd> and decompress them into the <kbd>./data/</kbd> directory.</p>

[Access datesets from Zenodo](https://zenodo.org/records/16066506)

### I. Run joint inversion

<p><kbd>$ python joint_inversion.py</kbd></p>

<p>The script creates a new folder as <kbd>./temp_inv_out/&lt;folder_name&gt;/</kbd> on each execution to store the joint inversion results and figures. The <kbd>&lt;folder_name&gt;</kbd> is in form of <kbd>&lt;mesh_name&gt;_&lt;timestamp&gt;</kbd>


<p>Inversion results used in the article is saved in route <kbd>./temp_inv_out/mesh_joint_Hannah_exfine_040825_032959_results_in_article/</kbd> (figures are not included to save space), so please skip this code if you only want to plot the data and models used in the article. Plot the results by setting<kbd> mesh_name="mesh_joint_Hannah_exfine"</kbd> and <kbd>folder_name="mesh_joint_Hannah_exfine_040825_032959_results_in_article"</kbd> in the plotting script below.</p>


### II. Plot results

<p>Use jupyter notebook script <kbd>plot_results.ipynb</kbd> (recommend to use VScode with jupyter notebook extension).</p>
<p>Run the preprocessing/initialization blocks first, then run following blocks according to the figure you want to plot.</p>
<p>The mesh_name should always be "mesh_joint_Hannah_exfine", and the folder_name should be the name of the folder (saved joint inversion results) in <kbd>./temp_inv_out/</kbd>.</p>

<p>Please find detailed instructions on using the plotting function according to the comment in this code and the tool <kbd>./myutils/plot_tool.py</kbd>. </p>
<p>**The geology differentiation results used in the article is saved in <kbd>./saved_GD_result/</kbd> folder, which will be automatically loaded into the plotting tool. Please do not modify them. The code for geology differentiation to generate these results will be included in future releases.</p>