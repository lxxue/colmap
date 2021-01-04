import numpy as np
import os

from utils import IntersectList, image_ids_to_pair_id, pair_id_to_image_ids, blob_to_array
from matches_list import MatchesList
from database import COLMAPDatabase

def ConstructPathNetwork(num_images, matches_list, img_included, unique_tracks, visible_tracks, minimal_views, score_thres):
    # path_graph = [None] * num_images
    # TODO: use sparse matrix to save memory
    is_geo_neighbors = np.zeros((num_images, num_images), dtype=np.bool)
    for i in range(num_images):
        if img_included[i]:
            print("---iconic image {} connects:".format(i+1))
            # TODO: if we use continue here, then non-iconic images will also be used as backbone connection
            # continue
        else:
            print("---non-iconic image {} connects:".format(i+1))
        unique_i = IntersectList(unique_tracks, visible_tracks[i], need_diff=False)
        for j, _ in matches_list[i]:
            # skip non-iconic images
            # was commented in the original code
            # if not img_included[j-1]:
            #     continue
            # i is 0-based and j is 1-based
            # common_ij = IntersectList(visible_tracks[i], visible_tracks[j-1], need_diff=False)
            unique_j = IntersectList(unique_tracks, visible_tracks[j-1], need_diff=False)
            unique_ij = IntersectList(unique_i, unique_j, need_diff=False)
            if len(unique_ij) > minimal_views:
                score = len(unique_ij) / max(len(unique_i), len(unique_j))
                if score > score_thres:
                    print("image {}(common unique points {}, max unique points {}, score {:.2f})".format(j, len(unique_ij), max(len(unique_i), len(unique_j)), score))
                    is_geo_neighbors[i, j-1] = True
                    is_geo_neighbors[j-1, i] = True
    
    return is_geo_neighbors

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
        # return
        print("WARNING: database path already exists -- will overwrite it.")
    new_db = COLMAPDatabase.connect(new_db_path) 
    new_db.create_tables()

    # TODO: copy the databse and only modify table matches and two_view_geometries 
    # instead of creating one from scratch
    # TODO: this is really stupid to first retrive binary data, convert it to non-binary
    # and then write it as binary... Should skip the conversion by changing COLMAPDatabase's API

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
    # Get a mapping between image ids and image names
    image_id_to_name = dict()
    for image_result in image_results:
        image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz = image_result
        image_id_to_name[image_id] = name
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

    for matches_result in matches_results:
        pair_id, rows, cols, matches = matches_result
        image_id1, image_id2 = pair_id_to_image_ids(pair_id)
        if rows > 0 and not is_geo_neighbors[image_id1-1][image_id2-1]:
            print("matches between image {} and image {} are discarded".format(image_id1, image_id2))
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
    # table two_view_geometries left empty and computed by colmap
    # TODO: any other parameters to specify here?
    os.system("colmap matches_importer --database_path {} --match_list_path {}".format(new_db_path, match_list_path))


def CompareDatabase(old_db_path, new_db_path):
    old_db = COLMAPDatabase.connect(old_db_path)
    new_db = COLMAPDatabase.connect(new_db_path)

    results1 = old_db.execute("SELECT * FROM cameras")
    results2 = new_db.execute("SELECT * FROM cameras")
    for result1, result2 in zip(results1, results2):
        camera_id1, model1, width1, height1, params1, prior1 = result1
        camera_id2, model2, width2, height2, params2, prior2 = result1
        params1 = blob_to_array(params1, np.float64)
        params2 = blob_to_array(params2, np.float64)
        assert(camera_id1 == camera_id2)
        assert(model1 == model2)
        assert(width1 == width2)
        assert(height1 == height2)
        assert(np.all(params1 == params2))
        assert(prior1 == prior2)

    results1 = old_db.execute("SELECT * FROM descriptors")
    results2 = new_db.execute("SELECT * FROM descriptors")
    for result1, result2 in zip(results1, results2):
        image_id1, rows1, cols1, data1 = result1
        image_id2, rows2, cols2, data2 = result2
        data1 = blob_to_array(data1, np.uint8, (rows1, cols1))
        data2 = blob_to_array(data2, np.uint8, (rows2, cols2))
        assert(image_id1 == image_id2)
        assert(np.all(data1 == data2))

    results1 = old_db.execute("SELECT * FROM images")
    results2 = new_db.execute("SELECT * FROM images")
    for result1, result2 in zip(results1, results2):
        image_id1, name1, camera_id1, prior_qw1, prior_qx1, prior_qy1, prior_qz1, prior_tx1, prior_ty1, prior_tz1 = result1
        image_id2, name2, camera_id2, prior_qw2, prior_qx2, prior_qy2, prior_qz2, prior_tx2, prior_ty2, prior_tz2 = result2
        assert(image_id1 == image_id2)
        assert(name1 == name2)
        assert(camera_id1 == camera_id2)
        assert(prior_qw1 == prior_qw2)
        assert(prior_qx1 == prior_qx2)
        assert(prior_qy1 == prior_qy2)
        assert(prior_qz1 == prior_qz2)
        assert(prior_tx1 == prior_tx2)
        assert(prior_ty1 == prior_ty2)
        assert(prior_tz1 == prior_tz2)

    results1 = old_db.execute("SELECT * FROM keypoints")
    results2 = new_db.execute("SELECT * FROM keypoints")
    for result1, result2 in zip(results1, results2):
        image_id1, rows1, cols1, keypoints1 = result1
        image_id2, rows2, cols2, keypoints2 = result2
        keypoints1 = blob_to_array(keypoints1, np.float32, (rows1, cols1))
        keypoints2 = blob_to_array(keypoints2, np.float32, (rows2, cols2))
        assert(image_id1 == image_id2)
        assert(np.all(keypoints1 == keypoints2))

    print("Table cameras|descriptors|images|keypoints in the new database are the same as those in the original databse")

    # matches and two_view_geometries in the new database should be the subset of that in the old database
    # assuming the geometry verification is done under the same condition
    results2 = new_db.execute("SELECT * FROM matches")
    for result2 in results2:
        pair_id2, rows2, cols2, matches2 = result2
        matches2 = blob_to_array(matches2, np.uint32, (rows2, cols2))
        result1 = next(old_db.execute("SELECT * FROM matches WHERE pair_id = {}".format(pair_id2)))
        pair_id1, rows1, cols1, matches1 = result1
        matches1 = blob_to_array(matches1, np.uint32, (rows1, cols1))
        assert(pair_id1 == pair_id2)
        assert(np.all(matches1 == matches2))

    print("Table matches in the new database is a subset of the that in the original databse")
    # print("Table matches|two_view_geometries in the new database is a subset of the that in the original databse")

    # results2 = new_db.execute("SELECT * FROM two_view_geometries")
    # for result2 in results2:
    #     pair_id2, rows2, cols2, matches2, config2, F2, E2, H2 = result2
    #     matches2 = blob_to_array(matches2, np.uint32, (rows2, cols2))            
    #     F2 = blob_to_array(F2, np.float64, (3, 3))
    #     E2 = blob_to_array(E2, np.float64, (3, 3))
    #     H2 = blob_to_array(H2, np.float64, (3, 3))
    #     result1 = next(old_db.execute("SELECT * FROM two_view_geometries WHERE pair_id = {}".format(pair_id2)))
    #     pair_id1, rows1, cols1, matches1, config1, F1, E1, H1 = result1
    #     matches1 = blob_to_array(matches1, np.uint32, (rows1, cols1))            
    #     F1 = blob_to_array(F1, np.float64, (3, 3))
    #     E1 = blob_to_array(E1, np.float64, (3, 3))
    #     H1 = blob_to_array(H1, np.float64, (3, 3))
    #     assert(pair_id1 == pair_id2)
    #     assert(np.all(matches1 == matches2))
    #     assert(config1 == config2)
    #     # assert(np.allclose(F1, F2))
    #     assert(F1 is not None and F2 is not None or F1 is None and F2 is None)
    #     assert(F1 is not None and E2 is not None or E1 is None and E2 is None)
    #     assert(F1 is not None and H2 is not None or H1 is None and H2 is None)
    #     if F1 is not None and F2 is not None and not np.allclose(F1, F2):
    #         print(F1.ravel())
    #         print(F2.ravel())
    #     # assert(np.allclose(E1, E2))
    #     if E1 is not None and E2 is not None and not np.allclose(E1, E2):
    #         print(E1.ravel())
    #         print(E2.ravel())
    #     # assert(np.allclose(H1, H2))
    #     if H1 is not None and H2 is not None and not np.allclose(H1, H2):
    #         print(H1.ravel())
    #         print(H2.ravel())


