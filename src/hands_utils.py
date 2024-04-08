from cv2.typing import MatLike
import cv2 as cv

from mediapipe.python.solutions import hands as h
from mediapipe.python.solutions import drawing_utils
import mediapipe as mp

from typing import List, Tuple

from aritmethic import Point2D, Line2D
from exceptions import InvalidPathException, VideoReadingException

import pandas as pd
import time

import pickle
import os
import numpy as np

points_positions: dict ={
    'WRIST' : '0',
    'THUMB_CMC' : '1',
    'THUMB_MCP' : '2',
    'THUMB_IP' : '3',
    'THUMB_TIP' : '4',
    'INDEX_FINGER_MCP' : '5',
    'INDEX_FINGER_PIP' : '6',
    'INDEX_FINGER_DIP' : '7',
    'INDEX_FINGER_TIP' : '8',
    'MIDDLE_FINGER_MCP' : '9',
    'MIDDLE_FINGER_PIP' : '10',
    'MIDDLE_FINGER_DIP' : '11',
    'MIDDLE_FINGER_TIP' : '12',
    'RING_FINGER_MCP' : '13',
    'RING_FINGER_PIP' : '14',
    'RING_FINGER_DIP' : '15',
    'RING_FINGER_TIP' : '16',
    'PINKY_MCP' : '17',
    'PINKY_PIP' : '18',
    'PINKY_DIP' : '19',
    'PINKY_TIP' : '20',
}

# TS represents the strak of the fingers lines
T1: list = [x for x in range(1, 5)]
T2: list = [x for x in range(5, 9)]
T3: list = [x for x in range(9, 13)]
T4: list = [x for x in range(13, 17)]
T5: list = [x for x in range(17, 21)]
TS: list = [T1, T2, T3, T4, T5]

class HandDetector():
    def __init__(self, static_image_mode: bool = True,
                        max_hands: int = 2,
                        min_det_confidence: int = 0.1,
                        min_track_confidence: int = 0.1) -> None:
        self.static_image_mode = static_image_mode # If true the mp will always make a detection proccess, that slow the processing
        self.max_hands = max_hands
        self.min_det_confidence = min_det_confidence
        self.min_track_confidence = min_track_confidence
        
        self.fps: int = 0
        # Hands() uses RGB images
        self.hands = h.Hands(static_image_mode=self.static_image_mode,
                                    max_num_hands= self.max_hands,
                                    min_detection_confidence=self.min_det_confidence,
                                    min_tracking_confidence=self.min_track_confidence)
    
    """
    land_marks return a mediapipe object
    """
    def land_marks(self, image: MatLike) -> object:
        previousTime: int = time.time()
        result: dict = self.hands.process(image).multi_hand_landmarks
        self.fps = round(1/(time.time()-previousTime))
        
        return result
    
    def drawer(self, hands_land_marks: dict, img: MatLike) -> None:
        if hands_land_marks:
            for handMark in hands_land_marks:
                drawing_utils.draw_landmarks(img, handMark, h.HAND_CONNECTIONS)
    
    """
    parser creates a readable dict
    """        
    def parser(self,
               hands_land_marks: dict,
               shape: Tuple[int, int, int] = None,
               draw: bool = True) -> dict:
        data: dict = {}
        if hands_land_marks:
            for hand_n, handMark in enumerate(hands_land_marks):
                lm_dict: dict = {}
                for id, lm in enumerate(handMark.landmark):     
                    h, w, c = shape if shape != None else (1, 1, 1)
                    cx, cy = (int(lm.x*w), int(lm.y*h)) if shape != None else (float(lm.x*w), float(lm.y*h))
                    
                    lm_dict[f'{id}'] = {
                        'x': cx,
                        'y': cy
                    }
                    
                data[f'hand_{hand_n}'] = lm_dict
                
        return data 
    
    """
    hand_stats receive a dict from the parser
    """                
    def hand_stats(self,
                   hands_positions: dict) -> dict:
        kk = time.time()
        detected_hands: list = hands_positions.keys()
        stats: dict = {}
        for key in detected_hands:
            stats[key] = {}
        # Now I'm going to calculate the distance between all points.
        for key in detected_hands:
            # Distance beetwen each finger point
            hand = hands_positions[key]
            hand_points: list[Point2D] = []
            # Create an instance of each point
            for key_1 in hand.keys():
                p = Point2D(hand[key_1]['x'], hand[key_1]['y'])
                hand_points.append(p)
                
            for id_0, point_0 in enumerate(hand_points):
                for id_1, point_1 in enumerate(hand_points):
                    if not point_0.equal(point_1):
                        distance: int = point_0.distance(point_1)
                        stats[key][f'D({id_0}-{id_1})'] = distance
                        
        # Now i'm going to calculate the angle between neighbours
        for key in detected_hands:
            hand = hands_positions[key]
            for id_0, t in enumerate(TS):
                i: int = t[0]
                i_1: int = i+1
                max_: int = t[-1]
                # i and i_1 represents the cursor wich will move over the list
                while i_1 != max_:
                    i_f = i_1 + 1 
                    p1 = Point2D(hand[f'{i}']['x'], hand[f'{i}']['y'])
                    pivot = Point2D(hand[f'{i_1}']['x'], hand[f'{i_1}']['y'])
                    p2 = Point2D(hand[f'{i_f}']['x'], hand[f'{i_f}']['y'])
                    l1 = Line2D(p1, pivot)
                    l2 = Line2D(pivot, p2)
                    alpha: float = l1.angle_with(l2)
                    i += 1
                    i_1 += 1
                    stats[key][f'A({i}_{i_1}-{i_1}_{i_f})'] = alpha
        
        return stats           
            
    def predict_hands_position(self,
                               rgb_img: MatLike,
                               draw: bool = True) -> dict:
        res: dict = self.land_marks(rgb_img)
    
        self.drawer(res, rgb_img) if draw else ...
    
        parse = self.parser(res)#, img.shape)
        data: dict = self.hand_stats(parse)
        
        detections: dict = {}
        
        with open('model/hands_model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        
            for k in data.keys():
                data_ = data[k]
                    
                if data != {}:
                    pred: str = loaded_model.predict(np.array([val for val in data_.values()]).reshape(1, -1))
                    detections[k] = pred[0]
            
        return detections
        
class HandsDataset:
    def __init__(self, data_frame: pd.DataFrame, position_name: str) -> None:
        self.data_frame = data_frame.reset_index(drop=True)
        self.position_name = position_name
        
    def export_dataset_to_csv(self, path: str) -> None:
        if type(path) != str: raise InvalidPathException
        if not os.path.exists(path): raise InvalidPathException
        
        self.data_frame.to_csv(path)
    
class RandomForestClassifierConfig:
    def __init__(self, n_estimators: int,
                 random_state: int,
                 test_size: float,
                 train_split_random_state: int) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.test_size = test_size
        self.train_split_random_state = train_split_random_state
        
class HandsTrainingResult:
    def __init__(self, accuracy: float, report: str | dict) -> None:
        self.accuracy = round(accuracy, 2)
        self.report = report
        
    def print_all(self) -> None:
        print(f"Accuracy: {self.accuracy}")
        print("Classification Report:\n", self.report)

class HandPositionTraining:
    @staticmethod
    def create_dataset_on_videocapture(position_name: str) -> HandsDataset:         
        df: pd.DataFrame = pd.DataFrame()
        
        video = cv.VideoCapture(1)
        hands = HandDetector(static_image_mode=False ,max_hands=2, min_det_confidence=.1)

        while True:
            _, img = video.read()
            rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            
            res: dict = hands.land_marks(rgb)
            hands.drawer(res, img)
        
            parse = hands.parser(res)#, img.shape)
            data: dict = hands.hand_stats(parse)
            hands_in_data: list = data.keys()
            
            for hand in hands_in_data:
                data_hand: dict = data[hand]
                data_hand['POS'] = position_name
                if data_hand != {'POS': position_name}:
                    df = pd.concat([pd.DataFrame([data_hand]), df])
            
            cv.imshow("video", img)
            if cv.waitKey(1) != -1:
                cv.destroyAllWindows()
                break
            
        return HandsDataset(df)

    @staticmethod
    def create_dataset_on_video(position_name: str,
                                video_path: str) -> HandsDataset:         
        if not os.path.exists(video_path): raise InvalidPathException
        
        df: pd.DataFrame = pd.DataFrame()
        
        video = cv.VideoCapture(video_path)
        hands = HandDetector(static_image_mode=False ,max_hands=2, min_det_confidence=.1)

        if not video.isOpened(): raise VideoReadingException

        while video.isOpend():
            _, img = video.read()
            rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            
            res: dict = hands.land_marks(rgb)
            hands.drawer(res, img)
        
            parse = hands.parser(res)#, img.shape)
            data: dict = hands.hand_stats(parse)
            hands_in_data: list = data.keys()
            
            for hand in hands_in_data:
                data_hand: dict = data[hand]
                data_hand['POS'] = position_name
                if data_hand != {'POS': position_name}:
                    df = pd.concat([pd.DataFrame([data_hand]), df])
            
        return HandsDataset(df)
    
    @staticmethod
    def train_model(datasets: List[HandsDataset],
                    output_path: str,
                    random_forest_config: RandomForestClassifierConfig = None) -> HandsTrainingResult:
        if not os.path.exists(output_path): raise InvalidPathException
        
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import pickle
        
        dataframe: pd.DataFrame = pd.DataFrame()

        if random_forest_config is None:
            random_forest_config = RandomForestClassifierConfig(n_estimators=100,
                                                                random_state=42,
                                                                test_size=0.2,
                                                                train_split_random_state=42)

        for temp_dataframe in datasets:
            dataframe = pd.concat([dataframe,
                                   temp_dataframe.data_frame])
        
        try:
            dataframe = dataframe.drop('Unnamed: 0', axis=1);
        except:
            pass
        
        X = dataframe.drop('POS', axis=1)  # Features
        y = dataframe['POS']  # Labels
        
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            random_forest_config.test_size,
                                                            random_forest_config.train_split_random_state)
        
        rf_classifier = RandomForestClassifier(n_estimators=random_forest_config.n_estimators,
                                               random_state=random_forest_config.random_state)

        # Train the classifier on the training data
        rf_classifier.fit(X_train.values, y_train.values)
        
        y_pred = rf_classifier.predict(X_test)
        
        with open('hands_model.pkl', 'wb') as model_file:
            pickle.dump(rf_classifier, model_file)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return HandsTrainingResult(accuracy, report)