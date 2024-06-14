from tmsneurosim.neuron_cell_parser.hoc_neuron_cell_to_py_neuron_cell import convert_hoc_to_python_cell
import tmsneurosim.nrn
import pathlib

nrn_path = pathlib.Path(tmsneurosim.nrn.__file__).parent
neuron_cell_names = [folder_paths.name for folder_paths
                        in sorted(pathlib.Path(nrn_path.joinpath(f'cells/cells_hoc').absolute()).iterdir())
                        if folder_paths.is_dir()]

python_cell_names = []
for neuron_cell_name in neuron_cell_names:
    python_cell_names.append(convert_hoc_to_python_cell(neuron_cell_name))

with open(nrn_path.joinpath(f'cells/__init__.py'), 'w') as f:
    for file_name, class_name in python_cell_names:
        f.write(f'from .{file_name} import {class_name}\n')
    f.write(f'from .neuron_cell import NeuronCell\n')