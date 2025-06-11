from sphere_surface import plot_sphere_pv, get_thresh_map_from_fname, get_thresh_diff_map_from_fnames
import pyvista as pv
from pyvista.plotting.plotter import Plotter

cell_name = 'L5_TTPC2_cADpyr_1'
exc_monophasic_name='L5_TTPC2_cADpyr_run8' # (60.0, 123.75)
threshold_fname = f'data/tms_thresholds/{exc_monophasic_name}/{exc_monophasic_name}_results.json'
data_map_by_cell = get_thresh_map_from_fname(threshold_fname)

exc_monophasic_name='L5_TTPC2_cADpyr_monophasic'
exc_half_sine_name='L5_TTPC2_cADpyr_half_sine'
# exc_monophasic_name='L5_TTPC2_cADpyr_run7'
# exc_half_sine_name='L5_TTPC2_cADpyr_run8'
threshold_fnames = [f'data/tms_thresholds/{exc_monophasic_name}/{exc_monophasic_name}_results.json',
                   f'data/tms_thresholds/{exc_half_sine_name}/{exc_half_sine_name}_results.json']
data_diff_map_by_cell = get_thresh_diff_map_from_fnames(threshold_fnames)


pl = plot_sphere_pv(data_map_by_cell[cell_name], cmap_name='inferno_r', radius=1, angular_resolution=10)

pl.export_html('sphere.html')