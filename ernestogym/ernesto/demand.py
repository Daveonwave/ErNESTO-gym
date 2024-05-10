import pandas as pd


class EnergyDemand:
    """
    Energy demand
    """
    def __init__(self, data: pd.DataFrame, timestep: int, data_type: str = 'dataset', profile_id: int = 0):
        assert data_type in ['single', 'dataset'], "data_type of demand must be 'single' or 'dataset'"
        assert str(profile_id) in data.columns, ("profile_id of demand must be a label within the columns of the "
                                                 "dataframe")

        self.timestep = timestep
        self._data_type = data_type
        self._profile_id = profile_id

        self._times = data['timestamp'].tolist()
        self._delta_times = data['delta_time'].tolist()
        self._history = data.drop(columns=['timestamp', 'delta_time']).to_dict(orient='list')

    @property
    def history(self):
        return self._history

    @property
    def delta_times(self):
        return self._delta_times

    def __len__(self):
        return len(self._times)

    def __getitem__(self, idx):
        assert 0 <= idx <= self.__len__(), "k must be between 0 and the total length of the demand history"
        return self._times[idx], self._delta_times[idx], {key: self._history[key][idx] for key in self._history.keys()}

    def get_idx_from_time(self, time: int) -> int:
        """
        Get index of demand history for given time.
        :param time: time of demand history
        :return: index of demand history
        """
        assert self._delta_times[0] < time < self._delta_times[-1], ("Cannot be retrieve an index of a time exceeding "
                                                                     "the time range of the demand history")
        # Value already in the list
        if time in self._delta_times:
            idx = self._delta_times.index(time)
        # If the value doesn't belong to the list, then take the next bigger value
        else:
            closer = min(self._delta_times, key=lambda x: abs(x - time))
            idx = self._delta_times.index(closer)

        return idx


