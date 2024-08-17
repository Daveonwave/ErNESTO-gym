import numpy as np

from ernestogym.ernesto.energy_storage.battery_models.generic_models import ElectricalModel
from .ecm_components import Resistor
from .ecm_components import ResistorCapacitorParallel
from .ecm_components import OCVGenerator
from ernestogym.ernesto.energy_storage.battery_models.parameters import instantiate_variables, Scalar, LookupTableFunction


def capacity_fading(c_n: float, q: float, alpha: float):
    """
    Capacity fading model depending on the exchange charge in the time interval.

    Parameters
    ----------------
    c_n: Capacity of the battery.
    q: Exchange charge in the time interval.
    alpha: Constant given in configuration file
    """
    return c_n * (1 - alpha * (q**0.5))


def resistance_fading(r_n: float, q: float, beta: float):
    """
    Resistor fading model depending on the exchange charge in the time interval.

    Parameters
    ----------------
    r_n: Resistance of the battery.
    q: Exchange charge in the time interval.
    beta: Constant given in configuration file
    """
    return r_n * (1 + beta * (q**0.5))


class TheveninFadingModel(ElectricalModel):
    """
    Thevenin Equivalent Circuit Model with fading functions for internal parameters.

    Reference:
    ----------------
    S. Barcellona, S. Colnago, E. Ferri and L. Piegari, "Evaluation of Dual-Chemistry Battery Storage System for
    Electric Vehicles Charging Stations," 2023 IEEE Vehicle Power and Propulsion Conference (VPPC), Milan, Italy, 2023,
    doi: 10.1109/VPPC60535.2023.10403281.
    """
    def __init__(self,
                 components_settings: dict,
                 sign_convention='active',
                 **kwargs
                 ):
        """

        """
        super().__init__(name='Thevenin with fading')
        self._sign_convention = sign_convention

        self._init_components = instantiate_variables(components_settings)

        self.r0 = Resistor(name='R0', resistance=self._init_components['r0'])
        self.rc = ResistorCapacitorParallel(name='RC',
                                            resistance=self._init_components['r1'],
                                            capacity=self._init_components['c'])
        self.ocv_gen = OCVGenerator(name='OCV', ocv_potential=self._init_components['v_ocv'])

        self._alpha_fading = kwargs['alpha_fading']
        self._beta_fading = kwargs['beta_fading']

        self._resistance_fading = np.vectorize(resistance_fading)
        self._capacity_fading = np.vectorize(capacity_fading)

    def reset_model(self, **kwargs):
        self._v_load_series = []
        self._i_load_series = []
        self._q_moved_charge_series = []
        self.r0.reset_data()
        self.rc.reset_data()
        self.ocv_gen.reset_data()

    def init_model(self, **kwargs):
        """
        Initialize the model at t=0
        """
        v = kwargs['voltage'] if kwargs['voltage'] else 0
        i = kwargs['current'] if kwargs['current'] else 0
        p = v * i
        q = kwargs['q_moved_charge']

        self.update_v_load(v)
        self.update_i_load(i)
        self.update_power(p)
        self.update_q_moved_charge(q)

        r0 = kwargs['r0'] if 'r0' in kwargs else None
        r1 = kwargs['r1'] if 'r1' in kwargs else None
        c = kwargs['c'] if 'c' in kwargs else None
        v_r0 = kwargs['v_r0'] if 'v_r0' in kwargs else None
        v_rc = kwargs['v_rc'] if 'v_rc' in kwargs else None
        v_ocv = kwargs['v_ocv'] if 'v_ocv' in kwargs else 0

        self.r0.init_component(r0=r0, v=v_r0)
        self.rc.init_component(r1=r1, c=c, v_rc=v_rc)
        self.ocv_gen.init_component(v=v_ocv)

    def load_battery_state(self, temp=None, soc=None, soh=None):
        """
        Update the SoC and SoH for the current simulation step
        """
        for component in [self.r0, self.rc, self.ocv_gen]:
            if temp is not None:
                component.temp = temp
            if soc is not None:
                component.soc = soc
            if soh is not None:
                component.soh = soh

    def step_voltage_driven(self, v_load, dt, k):
        """
        CV mode
        """
        # Solve the equation to get I
        r0 = self.r0.resistance
        r1 = self.rc.resistance
        c = self.rc.capacity
        v_ocv = self.ocv_gen.ocv_potential

        # Compute V_c with finite difference method
        term_1 = self.rc.get_v_series(k=k) / dt
        term_2 = (v_ocv - v_load) / (r0 * c)
        denominator = 1/dt + 1/(r0 * c) + 1/(r1 * c)

        v_rc = (term_1 + term_2) / denominator
        i = (v_ocv - v_rc - v_load) / r0

        if self._sign_convention == "passive":
            i = -i

        # Compute V_r0
        v_r0 = self.r0.compute_v(i=i)

        # Compute I_r1 and I_c for the RC parallel
        i_r1 = self.rc.compute_i_r1(v_rc=v_rc)
        i_c = self.rc.compute_i_c(i=i, i_r1=i_r1)

        # Compute power
        power = v_load * i

        # Moved charge
        q = self.get_q_moved_charge_series(k=-1) + abs(i) * dt / 3600

        # Update the collections of variables of ECM components
        self.r0.update_step_variables(r0=r0, v_r0=v_r0)
        self.rc.update_step_variables(r1=r1, c=c, v_rc=v_rc, i_r1=i_r1, i_c=i_c)
        self.ocv_gen.update_v(value=v_ocv)
        self.update_i_load(value=i)
        self.update_v_load(value=v_load)
        self.update_power(value=power)
        self.update_q_moved_charge(value=q)

        return v_load, i

    def step_current_driven(self, i_load, dt, k, p_load=None):
        """
        CC mode
        """
        # Solve the equation to get V
        r0 = self.r0.resistance
        r1 = self.rc.resistance
        c = self.rc.capacity
        v_ocv = self.ocv_gen.ocv_potential

        if self._sign_convention == 'passive':
            i_load = -i_load

        # Compute V_r0 and V_rc
        v_r0 = self.r0.compute_v(i=i_load)
        v_rc = (self.rc.get_v_series(k=k) / dt + i_load / c) / (1/dt + 1 / (c*r1))

        # Compute V
        v = v_ocv - v_r0 - v_rc

        # Compute I_r1 and I_c for the RC parallel
        i_r1 = self.rc.compute_i_r1(v_rc=v_rc)
        i_c = self.rc.compute_i_c(i=i_load, i_r1=i_r1)

        if p_load is not None:
            i_load = -i_load

        # Compute power
        if p_load is not None:
            power = p_load
        else:
            power = v * i_load
            if self._sign_convention == 'passive':
                power = -power

        # Moved charge
        q = self.get_q_moved_charge_series(k=-1) + abs(i_load) * dt / 3600

        # Update the collections of variables of ECM components
        self.r0.update_step_variables(r0=r0, v_r0=v_r0)
        self.rc.update_step_variables(r1=r1, c=c, v_rc=v_rc, i_r1=i_r1, i_c=i_c)
        self.ocv_gen.update_v(value=v_ocv)
        self.update_v_load(value=v)
        self.update_i_load(value=i_load)
        self.update_power(value=power)
        self.update_q_moved_charge(value=q)

        return v, i_load

    def step_power_driven(self, p_load, dt, k):
        """
        CP mode: to simplify the power driven case, we pose I = P / V(t-1), having a little shift in computed data
        """
        if self._sign_convention == 'passive':
            return self.step_current_driven(i_load=p_load / self._v_load_series[-1], dt=dt, k=k, p_load=p_load)
        else:
            return self.step_current_driven(i_load=p_load / self._v_load_series[-1], dt=dt, k=k, p_load=p_load)

    def compute_generated_heat(self, k=-1):
        """
        Compute the generated heat that can be used to feed the thermal model (when required).
        For Thevenin first order circuit it is: [P = V * I + V_rc * I_r1].

        Inputs:
        :param k: step for which compute the heat generation
        """
        # TODO: option about dissipated power computed with r0 only or r0 and r1
        return self.r0.get_r0_series(k=k) * self.get_i_series(k=k)**2 + \
            self.rc.get_r1_series(k=k) * self.rc.get_i_r1_series(k=k)**2
        # return self.r0.get_r0_series(k=k) * self.get_i_series(k=k) ** 2

    def compute_parameter_fading(self, c_n, k=-1):
        """
        Model of parameter fading hardcoded in this file within function 'capacity_fading' and 'resistor_fading'.
        """
        if isinstance(self.r0.resistor, Scalar):
            self.r0.resistance = resistance_fading(r_n=self.r0.nominal_resistance,
                                                   q=self.get_q_moved_charge_series(k=-1),
                                                   beta=self._beta_fading)
        elif isinstance(self.r0.resistor, LookupTableFunction):

            new_r0 = (self._resistance_fading(r_n=self.r0.nominal_resistance,
                                              q=self.get_q_moved_charge_series(k=-1),
                                              beta=self._beta_fading))
            fading_amount = new_r0 - self.r0.nominal_resistance
            self.r0.resistance = self.r0.resistor.get_y_values() + fading_amount

        else:
            raise TypeError("Unsupported type of resistance R0 for fading Thevenin model.")

        if isinstance(self.rc.resistor, Scalar):
            self.rc.resistance = resistance_fading(r_n=self.rc.nominal_resistance,
                                                   q=self.get_q_moved_charge_series(k=-1),
                                                   beta=self._beta_fading)
        elif isinstance(self.rc.resistor, LookupTableFunction):
            new_r1 = self._resistance_fading(r_n=self.rc.nominal_resistance,
                                             q=self.get_q_moved_charge_series(k=-1),
                                             beta=self._beta_fading)
            fading_amount = new_r1 - self.rc.nominal_resistance
            self.rc.resistance = self.rc.resistor.get_y_values() + fading_amount

        else:
            raise TypeError("Unsupported type of resistance R1 for fading Thevenin model.")

        # The fading applied to the capacity involves the capacity of the cell, not the one of the Thevenin model
        new_capacity = capacity_fading(c_n=c_n,
                                       q=self.get_q_moved_charge_series(k=-1),
                                       alpha=self._alpha_fading)

        #print('R0_n: ', self.r0.nominal_resistance, 'R1_n: ', self.rc.nominal_resistance, 'C: ', c_n)
        #print('R0: ', self.r0.resistance,  'R1: ', self.rc.resistance, 'C: ', new_capacity)
        return new_capacity

    def get_internal_resistance(self, nominal:bool=True):
        if nominal:
            return self.r0.nominal_resistance
        else:
            return self.r0.resistance

    def get_polarization_resistance(self, nominal:bool=True):
        if nominal:
            return self.rc.nominal_resistance
        else:
            return self.rc.resistance

    def get_internal_capacity(self):
        return self.rc.capacity

    def get_results(self, **kwargs):
        """
        Returns a dictionary with all final results
        TODO: selection of results by label from config file?
        """
        k = kwargs['k'] if 'k' in kwargs else None

        return {'voltage': self.get_v_series(k=k),
                'current': self.get_i_series(k=k),
                'power': self.get_p_series(k=k),
                'v_oc': self.ocv_gen.get_v_series(k=k),
                'r0': self.r0.get_r0_series(k=k),
                'r1': self.rc.get_r1_series(k=k),
                'c': self.rc.get_c_series(k=k),
                'q': self.get_q_moved_charge_series(k=k),
                'v_r0': self.r0.get_v_series(k=k),
                'v_rc': self.rc.get_v_series(k=k)
                }
