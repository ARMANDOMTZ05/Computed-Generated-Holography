import vamtoolbox as vam
import numpy as np


target_geo = vam.geometry.TargetGeometry(stlfilename=r"C:\Users\armar\OneDrive\Documentos\CE3E3V2_OpticsTest.stl", resolution=250)
target_geo.show()


num_angles = 360
angles = np.linspace(0, 360 - 360 / num_angles, num_angles)
proj_geo = vam.geometry.ProjectionGeometry(angles,ray_type='parallel',CUDA=True)

optimizer_params = vam.optimize.Options(method='OSMO',n_iter=25,d_h=0.85,d_l=0.6,filter='hamming',verbose='plot')
opt_sino, opt_recon, error = vam.optimize.optimize(target_geo, proj_geo,optimizer_params)

opt_sino.save('Optics_Test')