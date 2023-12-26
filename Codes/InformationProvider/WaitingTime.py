from traci import vehicle, lane, trafficlight
import numpy as np
class WaitingTime:
    def __init__(self, Agent_ids):
        self.Agent_ids = Agent_ids
        self.inti_vehicles_waiting_time = {}
        self.vehicles_waiting_time = {}
        for agent_id in self.Agent_ids:
            vehicles = self.get_vehicles(agent_id)
            waiting_time = self.Waiting_Time_Vehicles(agent_id, vehicles)
            self.inti_vehicles_waiting_time[agent_id] = [vehicles, waiting_time]
    def get_vehicles(self, junction_id):
        lane_ids = trafficlight.getControlledLanes(junction_id)
        vehicles = []
        for lane_id in lane_ids:
            lane_vehicles = list(lane.getLastStepVehicleIDs(lane_id))
            if len(lane_vehicles) <= 0:continue
            vehicles += lane_vehicles
        return vehicles
    def Waiting_Time_Vehicles(self, junction_id, vehicles=None):
        vehicles = self.get_vehicles(junction_id) if vehicles is None else vehicles
        if len(vehicles) > 0:
            waiting_time_vehicles = list(map(lambda veh: vehicle.getWaitingTime(veh), vehicles))
            return waiting_time_vehicles
        return [0]
    def Actual_Waiting(self, junction_id, vehicles=None):
        vehicles = self.get_vehicles(junction_id) if vehicles is None else vehicles
        waiting_time = self.Waiting_Time_Vehicles(junction_id, vehicles)
        self.vehicles_waiting_time[junction_id] = [vehicles, waiting_time]
        vehicles = list(set(self.inti_vehicles_waiting_time[junction_id][0]) - set(self.vehicles_waiting_time[junction_id][0]))
        waiting_times = []
        for id in vehicles:
            waiting_times.append(self.inti_vehicles_waiting_time[junction_id][1][self.inti_vehicles_waiting_time[junction_id][0].index(id)])
        self.inti_vehicles_waiting_time[junction_id] = self.vehicles_waiting_time[junction_id]
        return waiting_times
    def Average_Waiting_Time_Vehicles(self, junction_id, vehicles=None):
        waiting_time_vehicles = self.Actual_Waiting(junction_id, vehicles)
        if len(waiting_time_vehicles) > 0:
            average_waiting_time = np.average(waiting_time_vehicles)
            return average_waiting_time
        return 0
    def Standard_Deviation_Waiting_Time_Vehicles(self, junction_id, vehicles=None):
        waiting_time_vehicles = self.Actual_Waiting(junction_id, vehicles)
        if len(waiting_time_vehicles) > 0:
            std_dev_waiting_time = np.std(waiting_time_vehicles)
            return std_dev_waiting_time
        return 0
