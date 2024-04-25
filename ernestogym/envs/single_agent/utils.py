"""
This module primarily implements the `parameter_generator()` function which
generates the parameters dict for `EnergyStorageEnv`.
"""
import pandas as pd
import yaml
import pint.util
from pint import UnitRegistry
from . import demand_satisfaction

BATTERY_OPTIONS = "ernestogym/ernesto/data/battery/battery.yaml"
INPUT_VAR = 'power'     # 'power'/'current'/'voltage'

ELECTRICAL_MODEL = "ernestogym/ernesto/data/battery/models/electrical/thevenin.yaml"
THERMAL_MODEL = "ernesto/data/battery/models/thermal/r2c_thermal.yaml"
AGING_MODEL = "ernesto/data/battery/models/aging/bolun.yaml"

ENERGY_GENERATION = "ernesto/data/profiles/generation.csv"
ENERGY_DEMAND = "ernesto/data/profiles/demand.csv"
MARKET_TREND = "ernesto/data/profiles/market_trend.csv"

ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)

# Dictionary of units internally used inside the simulator
internal_units = dict(
    current=['ampere', 'A', ureg.ampere],
    voltage=['volt', 'V', ureg.volt],
    power=['watt', 'W', ureg.watt],
    resistance=['ohm', '\u03A9', ureg.ohm],
    capacity=['faraday', 'F', ureg.faraday],
    temperature=['kelvin', 'K', ureg.kelvin],
    time=['seconds', 's', ureg.s],
    soc=[None, None, None],
    soh=[None, None, None]
)

# TODO: check input validation

def _validate_data_unit(data_list, var_name, unit):
    """
    Function to validate and adapt preprocessing unit to internal simulator units.

    Args:
        data_list (list): list with values of a preprocessing stream
        var_name (str): name of the variable
        unit (str): unit of the variable
    """
    # Unit employed is already compliant with internal simulator units
    if unit == internal_units[var_name][1]:
        return data_list

    try:
        tmp_data = data_list * ureg.parse_units(unit)
        transformed_data = tmp_data.to(internal_units[var_name][2])
        print("Ground variable '{}' has been converted from [{}] to [{}]"
              .format(var_name, unit, internal_units[var_name][1]))

    except pint.PintError as e:
        raise Exception("UnitError on '{}': ".format(var_name), e)

    return transformed_data.magnitude.tolist()


def _validate_parameters_unit(param_dict):
    """
    Function to validate and adapt units of provided parameters to internal simulator units.

    Args:
        param_dict (dict): dictionary of parameters (read by for example yaml config file)
    """
    transformed_dict = {}

    for key in param_dict.keys():
        param = param_dict[key]

        # Check if the parameter has a unit measure with a dictionary structure
        if isinstance(param, dict):
            # Parameter unit measure is not compliant with internal simulator units
            if param['unit'] != internal_units[param['var']][1]:
                try:
                    tmp_param = param['value'] * ureg.parse_units(param['unit'])
                    transformed_dict[key] = tmp_param.to(internal_units[param['var']][2]).magnitude

                except pint.PintError as e:
                    raise Exception("UnitError on '{}': ".format(param['var']), e)

            else:
                transformed_dict[key] = param['value']

        else:
            transformed_dict[key] = param_dict[key]

    return transformed_dict


##################################################################
def parameter_generator(battery_options: str = BATTERY_OPTIONS,
                        input_var: str = INPUT_VAR,
                        electrical_model: str = ELECTRICAL_MODEL,
                        thermal_model: str = THERMAL_MODEL,
                        aging_model: str = AGING_MODEL,
                        energy_generation: str = ENERGY_GENERATION,
                        energy_demand: str = ENERGY_DEMAND,
                        market_trend: str = MARKET_TREND,
                        use_aging_model: bool = True,
                        use_generation_profile: bool = True,
                        use_demand_profile: bool = True,
                        use_market_trend: bool = True,
                        is_continuous_action: bool = False,
                        ) -> dict:
    """
    Generates the parameters dict for `EnergyStorageEnv`.
    """

    # Battery options and models configuration settings
    params = {'battery_options': _read_yaml(battery_options), 'input_var': input_var,
              'models_config': [_read_yaml(electrical_model), _read_yaml(thermal_model)]}
    if use_aging_model:
        params['models_config'].append(_read_yaml(aging_model))

    # Exogenous variables data
    if use_generation_profile:
        params['generation'] = _read_yaml(energy_generation)
    if use_demand_profile:
        params['demand'] = _read_yaml(energy_demand)
    if use_market_trend:
        params['market'] = _read_csv(market_trend)

    params['is_continuous_action'] = is_continuous_action

    return params


def _read_yaml(file: str) -> dict:
    """
    Read configuration from yaml file
    :param file: configuration file
    """
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config


def _read_csv(file: str) -> dict:
    """
    Read data from csv files
    # TODO: fix return data
    """
    csv_data = pd.read_csv(file)
    return csv_data
