import numpy as np
import os

from utils import IntersectList, image_ids_to_pair_id, pair_id_to_image_ids, blob_to_array
from matches_list import MatchesList
from database import COLMAPDatabase

def ConstructPathNetwork(num_images, matches_list, img_included, unique_tracks, visible_tracks, minimal_views, score_thres):
    path_graph = [None] * num_images
    # we can make this sparse matrix and save space later on
    is_geo_neighbors = np.zeros((num_images, num_images), dtype=np.bool)
    for i in range(num_images):
        path = []
        if img_included[i]:
            print("---iconic image {} connects:".format(i+1))
            continue
        
        print("---non-iconic image {} connects:".format(i+1))
        unique_i = IntersectList(unique_tracks, visible_tracks[i], need_diff=False)
        for j, matches in matches_list[i]:
            if img_included[j-1]:
                # skip iconic images
                # was commented in the original code
                # continue
                pass
            common_ij = IntersectList(visible_tracks[i], visible_tracks[j-1], need_diff=False)
            unique_j = IntersectList(unique_tracks, visible_tracks[j-1], need_diff=False)
            unique_ij = IntersectList(unique_i, unique_j, need_diff=False)
            if len(unique_ij) > minimal_views:
                score = len(unique_ij) / max(len(unique_i), len(unique_j))
                if score > score_thres:
                    path.append(j)
                    print("{} {} {} {}".format(j, len(unique_ij), max(len(unique_i), len(unique_j)), score))
                    is_geo_neighbors[i, j-1] = True
                    is_geo_neighbors[j-1, i] = True
        path_graph[i] = path
        print("")
    
    return path_graph, is_geo_neighbors

# def RegenerateMatches(num_images, is_geo_neighbors, matches_list):
#     # subject to simplications. For now we mimic the original code
#     # Actually I feel like do this in my way is way more convenient and straightforaward
#     new_matches_list = MatchesList(num_images) 
#     for i in range(num_images):
#         for j, matches in matches_list[i]:
#             # j is stored as 1-based
#             if is_geo_neighbors[i][j-1]:
#                 new_matches_list.append((j, matches))
# 
#     return new_matches_list


    # for i in range(num_images):
    #     for j in range(num_images):
    #         existed = False
    #         for k in path_graph[i]:
    #             if k == j:
    #                 existed = True
    #                 break
    #         if not existed:
    #             for k in path_graph[j]:
    #                 if k == i:
    #                     existed = True
    #                 break
    #
    #         if existed:
    #             pair_id = image_ids_to_pair_id(i, j)

def RewriteDatabese(old_db, new_db_path, is_geo_neighbors):
    if os.path.exists(new_db_path):
        os.system("rm {}".format(new_db_path))
        # print("ERROR: database path already exists -- will not modify it.")
        print("WARNING: database path already exists -- will overwrite it.")
        return
    new_db = COLMAPDatabase.connect(new_db_path) 
    new_db.create_tables()

    # TODO: copy the databse and only modify table matches and two_view_geometries 
    # instead of creating one from scratch
    # TODO: this is really stupid to first retrive binary data, convert it to non-binary
    # and then write it as binary... Should skip the conversion by changing COLMAPDatase's API

    # table camearas unchanged
    camera_results = old_db.execute("SELECT * FROM cameras")
    for camera_result in camera_results:
        camera_id, model, width, height, params, prior = camera_result
        params = blob_to_array(params, np.float64)
        new_db.add_camera(model=model,
                          width=width, 
                          height=height, 
                          params=params, 
                          prior_focal_length=prior, 
                          camera_id=camera_id)
    
    # table descriptors unchanged
    descriptors_results = old_db.execute("SELECT * FROM descriptors")
    for descriptors_result in descriptors_results:
        image_id, rows, cols, descriptors = descriptors_result
        descriptors = blob_to_array(descriptors, np.uint8, (rows, cols))
        new_db.add_descriptors(image_id=image_id, descriptors=descriptors)

    # table images unchanged
    image_results = old_db.execute("SELECT * FROM images")
    for image_result in image_results:
        image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz = image_result
        new_db.add_image(name=name, 
                         camera_id=camera_id, 
                         prior_q=np.array([prior_qw, prior_qx, prior_qy, prior_qz]),
                         prior_t=np.array([prior_tx, prior_ty, prior_tz]),
                         image_id=image_id)

    # table keypoints unchanged
    keypoints_results = old_db.execute("SELECT * FROM keypoints")
    for keypoints_result in keypoints_results:
        image_id, rows, cols, keypoints = keypoints_result
        keypoints = blob_to_array(keypoints, np.float32, (rows, cols))
        new_db.add_keypoints(image_id=image_id, 
                             keypoints=keypoints)

    # table matches filtered
    matches_results = old_db.execute("SELECT * FROM matches")
    # write match_list.txt for matches_importer
    match_list_path = os.path.join(os.path.dirname(new_db_path), "match_list.txt")
    match_list_file = open(match_list_path, 'wt')
    # Get a mapping between image ids and image names
    image_id_to_name = dict()
    cursor = old_db.execute('SELECT image_id, name FROM images;')
    for row in cursor:
        image_id = row[0]
        name = row[1]
        image_id_to_name[image_id] = name

    for matches_result in matches_results:
        pair_id, rows, cols, matches = matches_result
        image_id1, image_id2 = pair_id_to_image_ids(pair_id)
        if rows > 0 and is_geo_neighbors[image_id1-1][image_id2-1]:
            assert(is_geo_neighbors[image_id2-1][image_id1-1])
            matches = blob_to_array(matches, np.uint32, (rows, cols))
            new_db.add_matches(image_id1=image_id1,
                               image_id2=image_id2,
                               matches=matches)
            image_name1 = image_id_to_name[image_id1]
            image_name2 = image_id_to_name[image_id2]
            match_list_file.write("{} {}\n".format(image_name1, image_name2))

    match_list_file.close()
    new_db.commit()
    new_db.close()
    # table two_view_geometry left empty and computed by colmap
    # TODO: any other parameters to specify here?
    os.system("colmap matches_importer --database_path {} --match_list_path {}".format(new_db_path, match_list_path))








