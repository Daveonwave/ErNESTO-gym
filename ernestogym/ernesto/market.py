import pandas as pd


class EnergyMarket:
    """
    Class to represent an energy market.
    """
    def __init__(self, data: pd.DataFrame, timestep: int):
        self.timestep = timestep

        self._ask = data['ask'].tolist()
        self._bid = data['bid'].tolist()
        self._times = data['timestamp'].tolist()
        self._delta_times = data['delta_time'].tolist()

    @property
    def ask(self):
        return self._ask

    @property
    def bid(self):
        return self._bid

    def __len__(self):
        return len(self._times)

    def __getitem__(self, idx):
        return self._times[idx], self._delta_times[idx], self._ask[idx], self._bid[idx]

    def get_idx_from_time(self, time: int) -> int:
        """
        Get index of market history for given time.
        :param time: time of market history
        """
        assert self._delta_times[0] < time < self._delta_times[-1], ("Cannot be retrieve an index of a time exceeding "
                                                                     "the time range of the market history")
        # Value already in the list
        if time in self._delta_times:
            idx = self._delta_times.index(time)
        # If the value doesn't belong to the list, then take the next bigger value
        else:
            closer = min(self._delta_times, key=lambda x: abs(x - time))
            idx = self._times.index(closer)

        return idx
