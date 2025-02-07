import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class EnergyDemand:
    """
    Energy demand
    """
    def __init__(self, data: pd.DataFrame, 
                 timestep: int, 
                 test_profiles: list[str],
                 data_usage: str = 'end'):
        assert data_usage in ['end', 'circular'], "'data_usage' of demand must be 'end' or 'circular'."

        self.timestep = timestep
        self._test_profiles = test_profiles
        cols_to_drop = ['timestamp', 'delta_time']
        cols_to_drop.extend(self._test_profiles)
        self._profile_labels = data.drop(columns=cols_to_drop).columns.to_list()

        # To track the current profile during training
        self._eval_profile = None
        self._current_profile = None
        self._locked_profile = False

        # Format YYYY-MM-DD hh:mm:ss
        self._timestamps = data['timestamp'].to_numpy()
        # self._timestamps = np.array([datetime.strptime(time, "%Y-%m-%d %H:%M:%S") for time in data['timestamp']])

        # Format total seconds
        self._times = data['delta_time'].to_numpy()
        # Format seconds from beginning of the timeseries
        self._delta_times = self._times - self._times[0]
        self._history = data.drop(columns=['timestamp', 'delta_time']).to_dict(orient='list')

        # Variables used to handle terminating conditions
        self._data_usage = data_usage
        self._first_idx = None
        self._last_idx = None

        # Max demand value
        self.max_demand = max([max(values) for key, values in self._history.items()])
        self.min_demand = min([min(values) for key, values in self._history.items()])

    @property
    def history(self):
        return self._history[self._current_profile]

    @property
    def times(self):
        return self._times

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def labels(self, is_training: bool = True):
        return self._profile_labels if is_training else self._test_profiles

    @property
    def profile(self):
        return self._current_profile

    @profile.setter
    def profile(self, profile_id: str):
        print("profile: ", profile_id)
        if self._locked_profile:
            print("Profile ID cannot be overwritten: the used profile is the one given in input.")
        else:
            assert str(profile_id) in self._profile_labels or str(profile_id) in self._test_profiles, \
                "'profile_id' of demand must be a label within the columns of the dataframe."
            self._current_profile = profile_id

    def __len__(self):
        return self._timestamps.shape[0]

    def __getitem__(self, idx):
        """
        Get demand by index
        :param idx:
        """
        assert 0 <= idx <= len(self), "k must be between 0 and the total length of the demand history!"
        return self._timestamps[idx], self._times[idx], self._history[self._current_profile][idx]

    def get_idx_from_times(self, time: int) -> int:
        """
        Get index of demand history for given time.
        :param time: time of demand history
        :return: index of demand history
        """
        if self._data_usage is None:
            assert self._delta_times[0] < time < self._delta_times[-1], \
                "Cannot be retrieve an index of the demand exceeding the time's range at the first iteration!"

        time = time % self._delta_times[-1]
        idx = int(time // self.timestep)

        #closer = [t for t in range(time + 1, time - self.timestep + 1, -1) if t in self._delta_times][0]
        #idx = np.where(self._delta_times == (time // self.timestep))[0][0]

        if self._first_idx is None:
            self._first_idx = idx

        self._last_idx = idx

        return idx

    def use_test_profile(self, test_profile: str = None):
        self._current_profile = test_profile if test_profile is None else np.random.choice(self._test_profiles)

    def is_run_out_of_data(self):
        """
        Check if demand history is out-of-data.
        """
        if self._data_usage == 'end':
            if self._last_idx == len(self) - 1:
                print("Demand history is run out-of-data: end of dataset reached.")
                return True
        else:
            if self._last_idx == self._first_idx - 1:
                print("Demand history is run out-of-data: circular termination reached.")
                return True

        return False



