"""
This module primarily implements the `parameter_generator()` function which
generates the parameters dict for `EnergyStorageEnv`.
"""
import yaml
from pint import UnitRegistry
from ernestogym.ernesto.utils import read_csv
from ernestogym.ernesto import read_yaml, validate_yaml_parameters

BATTERY_OPTIONS = "ernestogym/ernesto/data/battery/battery.yaml"
INPUT_VAR = 'power'     # 'power'/'current'/'voltage'

ELECTRICAL_MODEL = "ernestogym/ernesto/data/battery/models/electrical/thevenin.yaml"
THERMAL_MODEL = "ernestogym/ernesto/data/battery/models/thermal/r2c_thermal.yaml"
AGING_MODEL = "ernestogym/ernesto/data/battery/models/aging/bolun.yaml"

WORLD = "ernestogym/envs/single_agent/world.yaml"

with open(WORLD, "r") as fin:
    world_settings = yaml.safe_load(fin)

ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)


def parameter_generator(battery_options: str = BATTERY_OPTIONS,
                        input_var: str = INPUT_VAR,
                        electrical_model: str = ELECTRICAL_MODEL,
                        thermal_model: str = THERMAL_MODEL,
                        aging_model: str = AGING_MODEL,
                        bypass_yaml_schema: bool = False,
                        is_continuous_action: bool = True,
                        ) -> dict:
    """
    Generates the parameters dict for `EnergyStorageEnv`.
    """
    # Battery options and models configuration settings retrieved with ErNESTO APIs.
    battery_params = read_yaml(battery_options, yaml_type='battery_options', bypass_check=bypass_yaml_schema)
    battery_params['battery']['params'] = validate_yaml_parameters(battery_params['battery']['params'])

    params = {'battery': battery_params['battery'],
              'input_var': input_var,
              'models_config': [read_yaml(electrical_model, yaml_type='model', bypass_check=bypass_yaml_schema),
                                read_yaml(thermal_model, yaml_type='model', bypass_check=bypass_yaml_schema)]}

    if 'soh' in world_settings['observations']:
        params['models_config'].append(read_yaml(aging_model, yaml_type='model', bypass_check=bypass_yaml_schema))
        params['aging'] = True

    # Exogenous variables data
    params['demand'] = {'data': read_csv(world_settings['demand']['path']),
                        'timestep': world_settings['demand']['timestep'],
                        'data_type': world_settings['demand']['data_type'],
                        'profile_id': world_settings['demand']['profile']}

    if 'generation' in world_settings['observations']:
        params['generation'] = {'data': read_csv(world_settings['generation']['path']),
                                'timestep': world_settings['generation']['timestep']}

    if 'market' in world_settings['observations']:
        params['market'] = {'data': read_csv(world_settings['market']['path']),
                            'timestep': world_settings['market']['timestep']}

    # Time info among observations
    params['day_sec_time'] = False
    params['rad_time'] = False

    if 'time_sec' in world_settings['observations']:
        params['day_sec_time'] = True
    if 'time_rad' in world_settings['observations']:
        params['rad_time'] = True

    params['is_continuous_action'] = is_continuous_action

    return params
