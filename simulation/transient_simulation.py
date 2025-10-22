import ansys.fluent.core as pyfluent
from ansys.fluent.core import SolverEvent
import os
import pandas as pd 
from ansys.fluent.core.services.field_data import SurfaceDataType
import argparse
import numpy as np
from typing import List
import pickle
import random
# set your environment
# awp_root = os.environ.get('AWP_ROOT232')
# os.environ['AWP_ROOT232']= "/home/byx/ansys_inc/v232"

solver = pyfluent.launch_fluent(mode="solver", precision="double",processor_count=4)
solver.file.read_case_data(file_name='result_case/steady_seed0.cas.h5')
solver.settings.file.auto_save.data_frequency = 0


def make_dir(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")


def record_field_data(param_data, seed, session, event_info):
    # record data
    if event_info.index %  3 == 1:
        field_data = session.field_data
        final_data = {}
        for surface in ["z-sit", "z-stand", "z-hvac"]:
            pos_data = field_data.get_surface_data(
                data_type = SurfaceDataType.Vertices, surface_name  = surface
            )
            if seed == 0:
                final_data[surface] = np.array([[pos_data.data[i].x, pos_data.data[i].y, pos_data.data[i].z] for i in range(len(pos_data.data))])
            for field in ["co2-ppm"]:
                data = field_data.get_scalar_field_data(
                        field_name=field, surface_name = surface
                    )
                final_data[field+surface] = np.array([data.data[i].scalar_data for i in range(len(pos_data.data))])
        if event_info.index == 1:
            for k in param_data.keys():
                final_data[k] = param_data[k]
        final_data["t"] = event_info.index 
        path = f'result_data/unsteady_{seed}.pkl'
        if os.path.exists(path):
            data = pickle.load(open(path, "rb"))
        else:
            data = []
        data.append(final_data)
        pickle.dump(final_data, open(path, "wb"))


def add_animation(name, save_path, solver):
    solver.settings.solution.calculation_activity.solution_animations[f"animate-{name}"] = {
        "animate_on": name,
        "storage_type": 7,
        "frequency_of": "time-step", 
        "frequency": 12,
        "view": "front",
        "storage_dir": save_path,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="seed")
    args = parser.parse_args()
    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    # variable
    param_data = {}
    p = np.random.randint(10, 80)
    param_data["p"] = p

    # setting boundary conditions
    for inlet_mode in range(1,7):
        inlet = solver.setup.boundary_conditions.velocity_inlet[f"inlet{inlet_mode}"]
        v = np.random.uniform(0.1, 1) * 3.24
        inlet.vmag.value = v
        angle = np.random.uniform(45, 135)
        inlet.flow_direction[0].value = 0.
        inlet.flow_direction[1].value = np.cos(np.deg2rad(angle)) # x direction
        inlet.flow_direction[2].value = -np.sin(np.deg2rad(angle)) # z direction
        inlet.mf["co2"].value = 0.0004
        param_data[f"inlet{inlet_mode}_v"] =  v
        param_data[f"inlet{inlet_mode}_angle"] = angle
    people_source = solver.setup.boundary_conditions.mass_flow_inlet["source"]
    people_source.mf['co2'].value = 0.04
    people_source.mass_flow.value = 0.012 / 80 * p

    method = "unsteady-1st-order"
    solver.settings.setup.general.solver.time.set_state(method)
    solver.settings.solution.initialization.hybrid_initialize()
    
    # run transient 
    option = np.random.randint(10)
    solver.settings.file.read_data(file_name = f"result_case/steady_seed{option}.dat.h5")
    param_data["steady_case"] = option
    callback_case = solver.events.register_callback(SolverEvent.TIMESTEP_ENDED, record_field_data, param_data, seed)
    solver.settings.solution.run_calculation.transient_controls.time_step_size = 10.
    solver.settings.solution.run_calculation.dual_time_iterate(
        time_step_count=int(12*30), max_iter_per_step=40
    )

    solver.exit()


if __name__ == "__main__":
    main()