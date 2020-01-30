import pickle
import pandas as pd
from utils import Location, Map

def generate_maps():

    maps = []

    map_name_1 = 'map_1'
    bridge_locations_1 = [(11, 26)]
    bridge_probabilities_1 = [0.4980]
    seed_1 = pd.read_csv('map_1.csv', header=None)
    start_1 = Location(5, 5)
    goal_1 = Location(35, 5)
    map_1 = Map(map_name_1, bridge_locations_1, bridge_probabilities_1, seed_1.values, start_1, goal_1)
    maps.append(map_1)

    map_name_2 = 'map_2'
    bridge_locations_2 = [(29, 37), (2, 38), (11, 43)]
    bridge_probabilities_2 = [0.698039215686275, 0.803921568627451, 0.698039215686275]
    seed_2 = pd.read_csv('map_2.csv', header=None)
    start_2 = Location(5, 5)
    goal_2 = Location(45, 5)
    map_2 = Map(map_name_2, bridge_locations_2, bridge_probabilities_2, seed_2.values, start_2, goal_2, None, 20)
    maps.append(map_2)

    map_name_3 = 'map_3'
    bridge_locations_3 = [(15, 25), (15, 26), (15, 27), (5, 31), (5, 32), (5, 33), (5, 34), (15, 34),
                          (5, 35), (5, 36), (5, 37), (5, 38), (5, 39), (15, 39), (5, 40), (25, 41)]
    bridge_probabilities_3 = [0.901960784313726, 0.901960784313726, 0.901960784313726, 0.952941176470588,
                              0.952941176470588, 0.952941176470588, 0.952941176470588, 0.901960784313726,
                              0.952941176470588, 0.952941176470588, 0.952941176470588, 0.952941176470588,
                              0.952941176470588, 0.901960784313726, 0.952941176470588, 0.498039215686275]
    seed_3 = pd.read_csv('map_3.csv', header=None)
    start_3 = Location(10, 8)
    goal_3 = Location(47, 3)
    map_3 = Map(map_name_3, bridge_locations_3, bridge_probabilities_3, seed_3.values, start_3, goal_3, None, 40)
    maps.append(map_3)

    return maps


def save_maps(maps, map_names):
    for map_, map_name in zip(maps, map_names):
        with open('{}.pickle'.format(map_name), 'wb') as f:
            pickle.dump(map_, f)
        

def load_map(map_name):
    with open(map_name, 'rb') as f:
        map_ = pickle.load(f)
    return map_


if __name__ == '__main__':
    maps = generate_maps()
    map_names = ['map_1', 'map_2', 'map_3']
    save_maps(maps, map_names)