import pandas as pd
import pickle


class DataLoader:
    
    def __init__(self, path, root_dir="./data/keypoints", keys=None):
        
        self.data = pd.read_csv(path)
        self.keypoints = {}
        if keys is None:
            keys = self.data['ID'].tolist()
        self.keys = keys
        
        for key in keys:
            try:
                keypoints_path = self.data[self.data['ID'] == key]['DataPath'].tolist()[0]
                with open(f"{root_dir}/{keypoints_path}", 'rb') as p:
                    keypoints = pickle.load(p)
                self.keypoints[key] = keypoints['keypoints3d']
            except:
                print(f"couldn't load data from {root_dir}/{keypoints_path}")
                
        return
    
    def load_keypoints3d(self, key, start_frame=0, end_frame=-1):
        return self.keypoints[key][start_frame:end_frame]
    
    def load_score(self, key, item):
        return int(self.data[self.data['ID'] == key][item])
