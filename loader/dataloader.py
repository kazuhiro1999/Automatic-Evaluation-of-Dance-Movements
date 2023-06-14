import pandas as pd
import pickle


class DataLoader:
    
    def __init__(self, path, root_dir="./data/keypoints"):
        
        self.data = pd.read_csv(path)
        self.keypoints = {}
        
        keys = self.data['ID'].tolist()
        for key in keys:
            try:
                keypoints_path = self.data[self.data['ID'] == key]['DataPath'].tolist()[0]
                with open(f"{root_dir}/{keypoints_path}", 'rb') as p:
                    keypoints = pickle.load(p)
                self.keypoints[key] = keypoints['keypoints3d']
            except:
                idx = self.data.index[self.data["ID"] == key]
                self.data = self.data.drop(idx, axis=0)
                print(f"couldn't load data from {root_dir}/{keypoints_path}")
        
        self.keys = self.data['ID'].tolist()
                
        return
    
    def load_keypoints3d(self, key, start_frame=0, end_frame=-1):
        return self.keypoints[key][start_frame:end_frame]
    
    def load_score(self, key, item):
        return int(self.data[self.data['ID'] == key][item])
