import pandas as pd


class EnergyGeneration:
    def __init__(self, data: pd.DataFrame, timestep: int):
        self.timestep = timestep

        self._times = data['timestamp'].tolist()
        self._delta_times = data['delta_time'].tolist()
        self._history = data['PV'].tolist()

    @property
    def history(self):
        return self._history

    def __len__(self):
        return len(self._history)

    def __getitem__(self, idx):
        assert 0 <= idx < len(self._history), f"Index {idx} out of range for energy generation"
        return self._times[idx], self._delta_times[idx], self._history[idx]

    def get_idx_from_time(self, time: int):
        """
        Get index of generation history for given time.
        :param time: time of generation history
        """
        assert self._delta_times[0] < time < self._delta_times[-1], ("Cannot be retrieve an index of a time exceeding "
                                                                     "the time range of the generation history")
        # Value already in the list
        if time in self._delta_times:
            idx = self._delta_times.index(time)
        # If the value doesn't belong to the list, then take the next bigger value
        else:
            closer = min(self._delta_times, key=lambda x: abs(x - time))
            idx = self._times.index(closer)

        return idx
