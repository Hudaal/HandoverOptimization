import re

# Data structure to keep all the imported time series data, either related to
# a particular UE or cell. This data structure is a nested dictionary keyed on
# data[obj_type][obj_id][timeseries_name], each entry containing a sorted list
# of (timestep, value) tuples, with obj_type as either 'ue' or 'cell'

class Data:
    def __init__(self):
        self.clear()

    def clear(self):
        self.data = {
            'ue': {},
            'cell': {},
        }

    def add_data(self, timestep, obj_type, obj_id, **d):
        if obj_id not in self.data[obj_type]:
            self.data[obj_type][obj_id] = {}
        for key, value in d.items():
            if key not in self.data[obj_type][obj_id]:
                self.data[obj_type][obj_id][key] = []
            # Append (timestep, value) tuples to a list such as data['ue'][2]['bytes_rx']
            self.data[obj_type][obj_id][key].append((timestep, value))

    def get_objects(self, obj_type):
        return sorted(self.data[obj_type].keys())

    def timeseries_valid(self, obj_type, obj_id, key):
        return (obj_type in self.data) and (obj_id in self.data[obj_type]) and (key in self.data[obj_type][obj_id])

    def get_timeseries(self, until_timestep, obj_type, obj_id, key):
        if not self.timeseries_valid(obj_type, obj_id, key):
            return None
        return list(filter(lambda t: t[0] <= until_timestep, self.data[obj_type][obj_id][key]))

    def get_full_timeseries(self, obj_type, obj_id, key):
        if not self.timeseries_valid(obj_type, obj_id, key):
            return None
        return self.data[obj_type][obj_id][key]

    def get_value(self, until_timestep, obj_type, obj_id, key):
        if not self.timeseries_valid(obj_type, obj_id, key):
            return None
        timeseries = self.get_timeseries(until_timestep, obj_type, obj_id, key)
        # Take the latest (timestep, value) tuple and return only the value
        return timeseries[-1][1] if len(timeseries) > 0 else None

    def get_latest_value(self, obj_type, obj_id, key, else_val=None):
        if not self.timeseries_valid(obj_type, obj_id, key):
            return None
        timeseries = self.get_full_timeseries(obj_type, obj_id, key)
        return timeseries[-1][1] if len(timeseries) > 0 else else_val

    def get_last_n_values(self, obj_type, obj_id, key, n):
        if not self.timeseries_valid(obj_type, obj_id, key):
            return None
        timeseries = self.get_full_timeseries(obj_type, obj_id, key)
        return [ t[1] for t in timeseries[-n:] ]

    def get_matching_timeseries(self, until_timestep, obj_type, obj_id, pattern):
        if obj_id not in self.data[obj_type]:
            return None
        matching_timeseries = []
        for key in sorted(self.data[obj_type][obj_id].keys()):
            match = re.match(pattern, key)
            if match is not None:
                extracted_key = match.group(1)
                timeseries = self.get_timeseries(until_timestep, obj_type, obj_id, key)
                if len(timeseries) > 0:
                    matching_timeseries.append((extracted_key, timeseries))
        return matching_timeseries
