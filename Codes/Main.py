import os
import sys
sys.path.append(os.path.abspath("."))
# code_path = '/content/drive/MyDrive/Colab_Notebooks/Muawiya/TCC_MI/TCC-MI/'
# sys.path.insert(0,code_path)
from Codes.configuration import Network_Path, Methods, Result_Path, save_object, load_object
from Codes.Controller import Controller
import matplotlib.pyplot as plt # type: ignore
# from traci import trafficlight
# from sumolib import checkBinary
# from traci import start
# import optparse
# import traci

class SUMO_ENV:

    def __init__(self):
        self.intersections = None
        self.is_Resumption = False

    # def get_Options(self):
    #     opt_parser = optparse.OptionParser()
    #     opt_parser.add_option("--nogui", action="store_true",
    #                           default=False, help="run the commandline version of sumo")
    #     options, _ = opt_parser.parse_args()
    #     return options

    # def Starting(self):
    #     if self.get_Options().nogui:sumoBinary = checkBinary('sumo')
    #     else: sumoBinary = checkBinary('sumo-gui')
    #     start([sumoBinary, "-c", Network_Path])

    # def exit(self):
    #     traci.close()
    #     sys.stdout.flush()

    def Random_Methode(self):
        controller = Controller(self.intersections, self.is_Resumption)
        controller.Run(method_name=Methods.Random)

    def Fixed_Methode(self):
        controller = Controller(self.intersections, self.is_Resumption)
        controller.Run(method_name=Methods.Fixed)

    def Kalman_Methode_R1(self):
        controller = Controller(self.intersections, self.is_Resumption)
        controller.Run(method_name=Methods.Kalman_R1) # 

    def Kalman_Methode_R2(self):
        controller = Controller(self.intersections, self.is_Resumption)
        controller.Run(method_name=Methods.Kalman_R2)

    def Kalman_Methode_R3(self):
        controller = Controller(self.intersections, self.is_Resumption)
        controller.Run(method_name=Methods.Kalman_R3)

    def Traditional_RL_Methode_R1(self):
        controller = Controller(self.intersections, self.is_Resumption)
        controller.Run(method_name=Methods.Traditional_R1)

    def Traditional_RL_Methode_R2(self):
        controller = Controller(self.intersections, self.is_Resumption)
        controller.Run(method_name=Methods.Traditional_R2)

    def Traditional_RL_Methode_R3(self):
        controller = Controller(self.intersections, self.is_Resumption)
        controller.Run(method_name=Methods.Traditional_R3)

    def Default_Case(self):
        print("Error running method")

    def Run_Methodes(self):
        os.makedirs(Result_Path, exist_ok=True)
        switch_dict = {
            Methods.Kalman_R1: self.Kalman_Methode_R1,
            # Methods.Fixed: self.Fixed_Methode,
            # Methods.Random: self.Random_Methode,
            # Methods.Traditional_R1: self.Traditional_RL_Methode_R1,
            # Methods.Kalman_R2: self.Kalman_Methode_R2,
            # Methods.Kalman_R3: self.Kalman_Methode_R3,
            # Methods.Traditional_R2: self.Traditional_RL_Methode_R2,
            # Methods.Traditional_R3: self.Traditional_RL_Methode_R3,
            }
        method_list = list(switch_dict.keys())
        method_name = method_list[0]
        if os.path.exists(os.path.join(Result_Path, "method.pkl")):
            method_name = load_object("method", Result_Path)
        for i, method in enumerate(method_list[method_list.index(method_name):]):
            if i==0 and os.path.exists(os.path.join(Result_Path, "method.pkl")):self.is_Resumption = True  
            else: self.is_Resumption = False
            # self.Starting()
            # self.intersections = trafficlight.getIDList()
            case_function = switch_dict.get(method, self.Default_Case)
            case_function()
            # self.exit()
if __name__ == "__main__":
    # try:
    env = SUMO_ENV()
    env.Run_Methodes()
    # except Exception as e:
    #     print(f"An exception of type {type(e).__name__} occurred.")

