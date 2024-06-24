import numpy as np
import pandas as pd
import pickle


class DataLoader:
    
    def __init__(self, path, root_dir="./data/keypoints"):
        
        self.data = pd.read_csv(path)
        self.items = ['Dynamics', 'Sharpness', 'Scalability', 'Timing', 'Accuracy', 'Stability']
        self.keypoints = {}
        self.rotations = {}
        self.scores = {}
        self.standardized_scores = {}
        
        self.keys = self.data['ID'].tolist()
        for key in self.keys:
            try:
                keypoints_path = self.data[self.data['ID'] == key]['DataPath'].tolist()[0]
                with open(f"{root_dir}/{keypoints_path}", 'rb') as p:
                    keypoints = pickle.load(p)
                self.keypoints[key] = keypoints['position18']
                self.rotations[key] = keypoints['rotation18']
            except:
                idx = self.data.index[self.data["ID"] == key]
                self.data = self.data.drop(idx, axis=0)
                print(f"couldn't load data from {root_dir}/{keypoints_path}")

        scores_np = self.data[self.items].to_numpy()
        mask = np.any(scores_np > 0, axis=-1)
        valid_scores = scores_np[mask]
        means = valid_scores.mean(axis=0)
        stds = valid_scores.std(axis=0)
        
        scores_np[self.data['IsReference'].tolist()] = 10
        standardized_scores_np = np.where(scores_np>0, (scores_np - means) / stds, -1)

        for i, key in enumerate(self.keys):
            self.scores[key] = {}
            self.standardized_scores[key] = {}
            for j, item in enumerate(self.items):
                self.scores[key][item] = scores_np[i,j]
                self.standardized_scores[key][item] = standardized_scores_np[i,j]
                
        return
    
    def load_keypoints3d(self, key, start_frame=0, end_frame=-1):
        return self.keypoints[key][start_frame:end_frame]

    def load_rotations(self, key, start_frame=0, end_frame=-1):
        return self.rotations[key][start_frame:end_frame]
    
    def load_score(self, key, item, standard=False):
        if standard:
            return self.standardized_scores[key][item]
        else:
            return self.scores[key][item]