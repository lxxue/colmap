import numpy as np
import sqlite3
from collections import namedtuple
import argparse
import time

from database import COLMAPDatabase
from matches_list import MatchesList
from compute_tracks import ComputeTracks
from summarize_scene import SummarizeScene
from regenerate_matches import ConstructPathNetwork, RewriteDatabese, CompareDatabase


# DisambiguateParameters = namedtuple('DisambiguateParameters', 
#                                     'track_degree \
#                                      coverage_thres \
#                                      score_thres \
#                                      alpha \
#                                      minimal_views',
#                                      defaults=[3, 0.6, 0.1, 0, 5])


def Disambiguate(params):
    print(params.db_path)
    db = COLMAPDatabase.connect(params.db_path)
    num_images = next(db.execute("SELECT COUNT(*) FROM images"))[0]
    print("{} images in the database".format(num_images))
    keypoints_rows = db.execute("SELECT rows FROM keypoints")
    num_keypoints_list = [row[0] for row in keypoints_rows]
    # for keypoints_result in keypoints_results:
    #     image_id, rows, cols, keypoints = keypoints_result
    #     num_keypoints_list.append(rows)
    #     print(image_id, rows)
    # print(num_keypoints_list)
    t0 = time.time()
    matches_list = MatchesList(num_images, database=db) 
    t1 = time.time()
    print("matches sorted")
    tracks, visible_tracks, visible_keypoints = ComputeTracks(
            num_images, num_keypoints_list, matches_list, params.track_degree)
    t2 = time.time()
    unique_tracks, img_included = SummarizeScene(
            tracks, visible_tracks, visible_keypoints, params.coverage_thres, params.alpha) 
    t3 = time.time()
    is_geo_neighbors = ConstructPathNetwork(
            num_images, matches_list, img_included, unique_tracks, visible_tracks, params.minimal_views, params.score_thres)
    t4 = time.time()
    RewriteDatabese(db, params.new_db_path, is_geo_neighbors)
    db.close()
    t5 = time.time()
    CompareDatabase(params.db_path, params.new_db_path)
    t6 = time.time()

    print("----------------Time Analysis---------------------")
    print("Sort Matches: {:.2f} minutes".format((t1-t0)/60))
    print("Compute Tracks: {:.2f} minutes".format((t2-t1)/60))
    print("Summarize Scene: {:.2f} minutes".format((t3-t2)/60))
    print("Construct Path Network: {:.2f} minutes".format((t4-t3)/60))
    print("Rewrite Database: {:.2f} minutes".format((t5-t4)/60))
    print("Compare Database: {:.2f} minutes".format((t6-t5)/60))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keep matches that are geodescially close")
    parser.add_argument("db_path", type=str, 
                        help="path to the original colmap database")
    parser.add_argument("new_db_path", type=str,
                        help="path to the new colmap database after disambiguation. should be different from db_path")
    parser.add_argument("--track_degree", type=int, default=3, 
                        help="minimal #views in different images for a track to be valid")
    parser.add_argument("--coverage_thres", type=float, default=0.6, 
                        help="minimal percentage for the tracks covered by the iconic set")
    parser.add_argument("--score_thres", type=float, default=0.1,
                        help="minimal persentage of shared unique points between two images for them to be regarded as geometrically consistent")
    parser.add_argument("--alpha", type=float, default=0., 
                        help="the weight for distinctiveness term in the objective function of choosing the next image to be added to the iconic set")
    parser.add_argument("--minimal_views", type=int, default=5,
                        help="minimal unique tracks two images must share to be regarded as geodesically consistent")
    params = parser.parse_args()
    Disambiguate(params)
    
    # example
    # python data/yan2017/colmap_street/match.db data/yan2017/colmap_street/new_match.db
    # params = DisambiguateParameters()
    # Disambiguate("data/yan2017/colmap_street/match.db", "data/yan2017/colmap_street/new_match.db", params)