from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import re
import datatracker
import math
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tensorflow.python.ops.numpy_ops import np_config

from Global_parameters import gp

np_config.enable_numpy_behavior()


class Environment_parser:
    def __init__(self) -> None:
        pass

    def parse_document(self, data, file_name):
        UE = 'ue'
        CELL = 'cell'
        f = open(file_name, "r")
        for line in f:
            match = re.match(r'^(.*) ms: Cell state: Cell (.*) at (.*) (.*) direction (.*)$', line)
            if match:
                data.add_data(int(match.group(1)), CELL, int(match.group(2)),
                    coords=(float(match.group(3)), float(match.group(4))),
                    direction=int(match.group(5)))
                # print("cords, direction", (float(match.group(3)), float(match.group(4))), int(match.group(5)))
            match = re.match(r'^(.*) ms: UE state: IMSI (.*) at (.*) (.*) with (.*) received bytes$', line)
            if match:
                data.add_data(int(match.group(1)), UE, int(match.group(2)),
                    coords=(float(match.group(3)), float(match.group(4))),
                    bytes_rx=int(match.group(5)))
            match = re.match(r'^(.*) ms: UE seen at cell: Cell (.*) saw IMSI (.*) \(context: .*\)$', line)
            if match:
                data.add_data(int(match.group(1)), UE, int(match.group(3)),
                    cell_associated=int(match.group(2)))
            match = re.match(r'^(.*) ms: Measurement report: Cell .* got measurements from IMSI (.*) \(ID .*, cell:RSRP/RSRQ (.*)\)$', line)
            if match:
                timestep, imsi = int(match.group(1)), int(match.group(2))
                measurements = match.group(3).split(' ')
                for measurement in measurements:
                    match = re.match(r'^(.*):(.*)/(.*)$', measurement)
                    cell, rsrp, rsrq = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    # print("cell, rsrp, rsrq", cell, rsrp, rsrq)
                    data.add_data(timestep, UE, imsi, **{
                        'rsrp_for_%d' % cell: rsrp,
                        'rsrq_for_%d' % cell: rsrq,
                    })
                            
    def find_cell_ue_dicts(self, data, durration_in_ms):
        ue_cell_connection = {}
        cell_connected_ue = {}
        for ue in data['ue']:
            ue_cell_connection[ue] = {}
            # print('ue: ', ue, 'cells: ', data['ue'][ue]['cell_associated'])
            for index, (time, cell) in enumerate(data['ue'][ue]['cell_associated']):
                if cell not in ue_cell_connection[ue].keys():
                    ue_cell_connection[ue][cell] = []
                # else:
                    # print('cell ' + str(cell) + ' in ue ' + str(ue) + ' is repeated!')
                ue_cell_connection[ue][cell].append([round(time, -2), durration_in_ms-1])
                if cell not in cell_connected_ue.keys():
                    cell_connected_ue[cell] = {}
                cell_connected_ue[cell][ue] = ue_cell_connection[ue][cell]
                if index > 0:
                    ue_cell_connection[ue][data['ue'][ue]['cell_associated'][index-1][1]][-1][1] = round(time, -2)
                    cell_connected_ue[data['ue'][ue]['cell_associated'][index-1][1]][ue] = ue_cell_connection[ue][data['ue'][ue]['cell_associated'][index-1][1]]


        return ue_cell_connection, cell_connected_ue

    def Add_more_cell_info(self, data, cell_connected_ue):
        cell_info = {}
        all_cells = list(range(1, gp.ENB_Count+1))
        for index, (cell, ue_info) in enumerate(cell_connected_ue.items()):
            # print(index, cell, ue_info)
            all_cells.remove(int(cell))
            all_ues_durations = []
            all_ues_distances = []
            all_ues_throughputs = []
            all_ues_rsrq = []
            all_ues_rsrp = []
            handovers = 0
            for ue, time_intevals in ue_info.items():
                # print('ue', ue, ' in cell', cell)
                throughputs = []
                rsrqs = []
                rsrps = []
                distances = []
                duration_ue = 0
                for time_inteval in time_intevals:
                    handovers += 1

                    duration_ue += time_inteval[1]-time_inteval[0]

                    if 'rsrq_for_'+str(cell) not in data['ue'][ue].keys():
                        rsrq = [(0,0)]
                    else: 
                        rsrq = list(filter(lambda x: (x[0] >= time_inteval[0] and x[0] <= time_inteval[1]), data['ue'][ue]['rsrq_for_'+str(cell)]))
                    rsrqs.extend([row[1] for row in rsrq])
                    all_ues_rsrq.extend([row[1] for row in rsrq])

                    if 'rsrp_for_'+str(cell) not in data['ue'][ue].keys():
                        rsrp = [(0,0)]
                    else: 
                        rsrp = list(filter(lambda x: (x[0] >= time_inteval[0] and x[0] <= time_inteval[1]), data['ue'][ue]['rsrp_for_'+str(cell)]))
                    rsrps.extend([row[1] for row in rsrp])
                    all_ues_rsrp.extend([row[1] for row in rsrp])

                    throughput = list(filter(lambda x: (x[0] >= time_inteval[0] and x[0] <= time_inteval[1]), data['ue'][ue]['bytes_rx']))
                    throughput_values = [row[1] for row in throughput]
                    throughput_times = [row[0] for row in throughput]


                    for time_step in range(time_inteval[0], time_inteval[1]+1, 100):
                        coords = list(filter(lambda x: (x[0] == time_step), data['ue'][ue]['coords']))
                        # print('time_step:', time_step, 'coords: ', coords)
                        if len(coords) > 0:
                            coords = coords[0]
                            distance = math.sqrt((coords[1][0]- data['cell'][cell]['coords'][0][1][0])**2 +
                                            (coords[1][1] - data['cell'][cell]['coords'][0][1][1])**2)
                            distances.append(distance)
                        all_ues_distances.append(distance)

                        if round(time_step, -2) in throughput_times:
                            throughputs_index = throughput_times.index(round(time_step, -2))
                            throughputs.append(throughput_values[throughputs_index])
                            all_ues_throughputs.append(throughput_values[throughputs_index])


                all_ues_durations.append(duration_ue)
                info_dict = {}
                info_dict['distances'] = np.array(distances)
                info_dict['rsrq'] = np.array(rsrqs)
                info_dict['rsrp'] = np.array(rsrps)
                info_dict['throughputs'] = np.array(throughputs)
                cell_connected_ue[cell][ue].append(info_dict)
                myoutput = open(gp.rsrq_throughput_file, 'a')
                if not (len(info_dict['rsrp']) == 0 or len(info_dict['rsrp']) == 0 or len(info_dict['rsrp']) == 0):
                    subprocess.run(["echo", 'cell = {0}: ue = {1}: avg_throughput = {2}: avg_rsrq  = {3}: avg_rsrp  = {4}\n'.format(cell, ue, sum(info_dict['throughputs'])/len(info_dict['throughputs']), sum(info_dict['rsrq'])/len(info_dict['rsrq']), sum(info_dict['rsrp'])/len(info_dict['rsrp']))], stdout=myoutput)

            # print('duration', all_ues_durations)
            all_ues_durations = np.array(all_ues_durations)
            all_ues_distances = np.array(all_ues_distances)
            all_ues_throughputs = np.array(all_ues_throughputs)
            # print('cell', cell, 'rsrq', all_ues_rsrq)
            all_ues_rsrq = np.array(all_ues_rsrq)
            all_ues_rsrp = np.array(all_ues_rsrp)
            
            cell_info[cell] = {'durations': all_ues_durations, 'distances': all_ues_distances, 'rsrqs': all_ues_rsrq, 'rsrps': all_ues_rsrp, 
                            'throughputs': all_ues_throughputs, 'handovers':  np.array([handovers])[0],
                            'ue_connected': np.array([len(cell_connected_ue[cell].keys())])[0]}
            
        for cell in all_cells:
            cell_info[cell] = {'durations': [0], 'distances': [0], 'rsrqs': [0], 'rsrps': [0], 
                            'throughputs': [0], 'handovers': 0, 'ue_connected': 0}
            
        # print(cell_info)
        subprocess.run(["echo", '\n'], stdout=myoutput)
        return cell_connected_ue, cell_info

    def find_max(self, cell_info, key, eval1, eval2):
        max_arr = []
        for cell, value in cell_info.items():
            if len(value[key]) == 0:
                max_arr.append(0)
            else: max_arr.append(max(value[key]))
        return max(max_arr)

    def find_state(self, cell_info, ue_cell_connection, UE_Count, duration, eval1, eval2, max_speed, min_speed):
        state = []
        if eval1:
            gp.eval_handovers_count = 0
        elif eval2:
            gp.eval2_handovers_count = 0
        else:
            gp.handovers_count = 0
        for cell, info in cell_info.items():
            cell_state = np.zeros([gp.all_count])
            for key, value in info.items():
                if key == 'durations' or key == 'distances' or key == 'rsrqs' or key == 'throughputs' or key == 'rsrps':
                    max_value = self.find_max(cell_info, key, eval1, eval2)
                    # print(max_value, key)
                    if len(value) == 0:
                        avg = 0
                        min_value = 0
                        max_value = 0
                    else:
                        values = value/max_value
                        avg = sum(values) / len(values)
                        min_value = min(values)
                        max_value = max(values)
                    if key == 'durations':
                        cell_state[0] = avg
                        cell_state[1] = min_value
                        cell_state[2] = max_value
                    elif key == 'distances':
                        cell_state[3] = avg
                        cell_state[4] = min_value
                        cell_state[5] = max_value
                    elif key == 'rsrqs':
                        cell_state[6] = avg
                        if eval1:
                            gp.eval_cell_rsrqs.append(max_value)
                        elif eval2:
                            gp.eval2_cell_rsrqs.append(max_value)
                        else:
                            gp.cell_rsrqs.append(max_value)
                    elif key == 'rsrps':
                        cell_state[7] = avg
                    elif key == 'throughputs':
                        cell_state[8] = avg
                        if eval1:
                            gp.eval_cell_throughputs.append(max_value)
                        elif eval2:
                            gp.eval2_cell_throughputs.append(max_value)
                        else:
                            gp.cell_throughputs.append(max_value)
                if key == 'handovers':
                    cell_state[9] = value/100
                    if eval1:
                        gp.eval_handovers_count += value
                    elif eval2:
                        gp.eval2_handovers_count += value
                    else:
                        gp.handovers_count += value
                if key == 'ue_connected':
                    cell_state[10] = value/10
            state.extend(cell_state)
        print('handovers', gp.handovers_count)
        state.append(UE_Count/gp.UE_upper_count)
        state.append(max_speed)
        state.append(min_speed)
        state.append(duration)
        return state

    def find_reward(self, data, cell_connected_ue, duration, UE_Count, eval1, eval2): #total time in 100 ms
        HO_total = 0
        throughput_total = 0
        count = 0
        rsrq_sum = 0

        for cell in data.data['cell'].keys():
            throughput_sum = 0
            if(cell in cell_connected_ue.keys()):
                for ue, info in cell_connected_ue[cell].items():
                    throughput_sum += sum(info[-1]['throughputs'])
                    rsrq_sum += sum(info[-1]['rsrq'])
                    count += 1
                    HO_total += len(info) - 1
                cell_throughput = throughput_sum / (duration * 10)
                throughput_total += cell_throughput

        # print('handovers from reward', HO_total)
        ANOH = HO_total / (UE_Count * duration * 10)
        optimize_ratio = throughput_total / ANOH
        rsrq_avg = rsrq_sum / count

        if eval1:
            gp.eval_throughput_to_save.append(throughput_total)
            gp.eval_rsrq_to_save.append(rsrq_sum)
        elif eval2:
            gp.eval2_throughput_to_save.append(throughput_total)
            gp.eval2_rsrq_to_save.append(rsrq_sum)
        else:
            gp.throughput_to_save.append(throughput_total)
            gp.rsrq_to_save.append(rsrq_sum)

        rsrq = rsrq_sum / duration * 10
        return throughput_total / 50000
