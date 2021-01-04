import numpy as np
import sqlite3
from collections import namedtuple

from database import COLMAPDatabase
from data_structure import MatchesList
from compute_tracks import ComputeTracks
from summarize_scene import SummarizeScene
from regenerate_matches import ConstructPathNetwork, RewriteDatabese

DisambiguateParameters = namedtuple('DisambiguateParameters', 
                                    'track_degree coverage_thres score_thres alpha minimal_views',
                                    defaults=[3, 0.6, 0.1, 0, 5])



def Disambiguate(db_path, new_db_path, params):
    db = COLMAPDatabase.connect(db_path)
    num_images = next(db.execute("SELECT COUNT(*) FROM images"))[0]
    print("{} images in the database".format(num_images))
    keypoints_rows = db.execute("SELECT rows FROM keypoints")
    num_keypoints_list = [row[0] for row in keypoints_rows]
    # for keypoints_result in keypoints_results:
    #     image_id, rows, cols, keypoints = keypoints_result
    #     num_keypoints_list.append(rows)
    #     print(image_id, rows)
    print(num_keypoints_list)
    matches_list = MatchesList(num_images, database=db) 
    tracks, visible_tracks, visible_keypoints = ComputeTracks(
            num_images, num_keypoints_list, matches_list, params.track_degree)
    confusing_tracks, unique_tracks, img_included = SummarizeScene(
            tracks, visible_tracks, visible_keypoints, params.coverage_thres, params.alpha) 
    path_graph, is_geo_neighbors = ConstructPathNetwork(
            num_images, matches_list, img_included, unique_tracks, visible_tracks, params.minimal_views, params.score_thres)
    RewriteDatabese(db, new_db_path, is_geo_neighbors)

    db.close()
    
        



if __name__ == "__main__":
    params = DisambiguateParameters()
    Disambiguate("data/yan2017/colmap_street/match.db", "data/yan2017/colmap_street/new_match.db", params)