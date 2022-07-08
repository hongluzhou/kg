import os
import pdb
import numpy as np
from tqdm import tqdm
import glob
import time
import datetime
import pickle
import json
from collections import defaultdict
import pandas as pd
pd.options.display.max_colwidth = 500

import torch

from sentence_transformers import util
# from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


def load_step_MPNet_embeddings(step_embeddings_path):
    return np.load(step_embeddings_path)


def load_step_S3D_text_embeddings(step_embeddings_path):
    with open(step_embeddings_path, 'rb') as f:
        step_embeddings = pickle.load(f)
    return step_embeddings


def get_step_des_feats(args, logger, language_model='MPNet'):
    
    if language_model == 'MPNet':
        step_des_feats = load_step_MPNet_embeddings(args.step_des_feats_mpnet_path)
    elif language_model == 'S3D':
        step_des_feats = load_step_S3D_text_embeddings(args.step_des_feats_s3d_path)
    logger.info("step descriptons features loaded! shape: {}".format(step_des_feats.shape))
    
    return step_des_feats


def get_all_video_ids(args, logger):
    start_time = time.time()
    if os.path.exists(args.video_ID_path):
        videos = np.load(args.video_ID_path)
    else:
        videos = []
        for f in tqdm(os.listdir(args.segment_feat_dir)):
            if os.path.isdir(os.path.join(args.segment_feat_dir, f)):
                videos.append(f)
        logger.info("number of videos: {}".format(len(videos)))
        np.save(args.video_ID_path, videos)
    logger.info("getting all video IDs took {} s".format(round(time.time()-start_time), 2))
    return videos


def load_frame_S3D_embeddings(args, logger):
    # get all video IDs
    videos = get_all_video_ids(args, logger)
    
    frame_embeddings = []
    frame_lookup_table = []
    
    videos_missing_features = set()
    for v in tqdm(videos):
        try:
            video_s3d = np.load(os.path.join(args.segment_feat_dir, v, 'video.npy'))
            # video_s3d shape: (num_clips, num_subclips, 512)
            
            for c_idx in range(video_s3d.shape[0]):
                frame_embeddings.append(np.float64(np.mean(video_s3d[c_idx], axis=0)))
                frame_lookup_table.append((v, c_idx))

        except FileNotFoundError:
            videos_missing_features.add(v)

    logger.info("number of videos missing visual S3D features: {}".format(
        len(videos_missing_features)))
   
    frame_embeddings = np.array(frame_embeddings)
    
    return frame_embeddings, frame_lookup_table


def load_narration_MPNet_embeddings(args, logger):
    
    # get all video IDs
    videos = get_all_video_ids(args, logger)
    
    narration_embeddings = []
    narration_lookup_table = []
    
    videos_missing_features = set()
    for v in tqdm(videos):
        try:
            text_mpnet = np.load(os.path.join(args.segment_feat_dir, v, 'text_mpnet.npy'))
            # text_mpnet shape: (num_clips, num_subclips, 768)

            for c_idx in range(text_mpnet.shape[0]):
                narration_embeddings.append(np.mean(text_mpnet[c_idx], axis=0))
                narration_lookup_table.append((v, c_idx))

        except FileNotFoundError:
            videos_missing_features.add(v)

    logger.info("number of videos missing narration MPNet features: {}".format(
        len(videos_missing_features)))
    
    narration_embeddings = np.array(narration_embeddings)
    
    return narration_embeddings, narration_lookup_table


def get_segment_video_feats(args, logger):
    # load segment video embeddings
    start_time = time.time()
    if os.path.exists(args.frame_embeddings_path) and os.path.exists(args.frame_embeddings_lookup_table_path):
        frame_embeddings = np.load(args.frame_embeddings_path)
        frame_lookup_table = pickle.load(open(args.frame_embeddings_lookup_table_path, "rb"))
    else:
        frame_embeddings, frame_lookup_table = load_frame_S3D_embeddings(args, logger)
        np.save(args.frame_embeddings_path, frame_embeddings)    
        with open(args.frame_embeddings_lookup_table_path, 'wb') as f:
            pickle.dump(frame_lookup_table, f)
    
    logger.info("segment video embeddings shape: {}".format(frame_embeddings.shape))
    # segment video embeddings shape: (3741608, 512)
    logger.info("getting all segment video embeddings took {} s".format(round(time.time()-start_time), 2))
    return frame_embeddings, frame_lookup_table
    
    
def get_segment_narration_feats(args, logger):
    # load segment narration embeddings
    start_time = time.time()
    if os.path.exists(args.narration_embeddings_path) and os.path.exists(args.narration_lookup_table_path):
        narration_embeddings = np.load(args.narration_embeddings_path)
        narration_lookup_table = pickle.load(open(args.narration_lookup_table_path, "rb"))
    else:
        narration_embeddings, narration_lookup_table = load_narration_MPNet_embeddings(args, logger)
        np.save(args.narration_embeddings_path, narration_embeddings)    
        with open(args.narration_lookup_table_path, 'wb') as f:
            pickle.dump(narration_lookup_table, f)
    
    logger.info("segment narration embeddings shape: {}".format(narration_embeddings.shape))
    # segment narration embeddings shape: (3741608, 768)
    logger.info("getting all segment narration embeddings took {} s".format(round(time.time()-start_time), 2))
    return narration_embeddings, narration_lookup_table


def get_nodes_by_removing_step_duplicates(args, logger, step_des_feats):
    start_time = time.time()
    if os.path.exists(args.node2step_path) and os.path.exists(args.step2node_path):
        node2step = np.load(args.node2step_path)
        step2node = np.load(args.step2node_path)
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            linkage=args.step_clustering_linkage, 
            distance_threshold=args.step_clustering_distance_thresh, 
            affinity=args.step_clustering_affinity).fit(step_des_feats)
            # distance_threshold : The linkage distance threshold above which, clusters will not be merged. 
        num_nodes = clustering.n_clusters_

        node2step, step2node = defaultdict(), defaultdict()
        for cluster_id in range(num_nodes):
            cluster_members = np.where(clustering.labels_ == cluster_id)[0]
            node2step[cluster_id] = cluster_members
            for step_id in cluster_members:
                step2node[step_id] = cluster_id
        np.save(args.node2step_path, node2step)    
        np.save(args.step2node_path, step2node)    

    logger.info("from steps to nodes took {} s".format(round(time.time()-start_time), 2))
    return node2step, step2node


def find_matched_segments_for_steps_using_narration(
    step_embeddings, narration_embeddings, threshold=0.7, topK=3):
    
    segments_of_steps = defaultdict()
    for step_id in tqdm(range(step_embeddings.shape[0])):
        
        cos_scores = util.cos_sim(step_embeddings[step_id], narration_embeddings)

        sorted_values, sorted_indices = torch.sort(cos_scores[0], descending=True)
        
        segments_of_steps[step_id] = list()
        for i in range(len(sorted_values)):
            if sorted_values[i] > threshold and i < topK:
                segments_of_steps[step_id].append(sorted_indices[i])
            else:
                break
                
    return segments_of_steps


def find_matched_segments_for_steps_using_video(
    step_embeddings, frame_embeddings, threshold=9, topK=3):
    
    segments_of_steps = defaultdict()
    for step_id in tqdm(range(step_embeddings.shape[0])):
        
        # dot product as similarity score
        sim_scores = np.einsum('ij,ij->i',
                               step_embeddings[step_id][np.newaxis, ...], 
                               frame_embeddings)
        sorted_values = np.sort(sim_scores)[::-1]  # sort in descending order
        sorted_indices = np.argsort(-sim_scores)  # indices of sorting in descending order
    
        segments_of_steps[step_id] = list()
        for i in range(len(sorted_values)):
            if sorted_values[i] > threshold and i < topK:
                segments_of_steps[step_id].append(sorted_indices[i])
            else:
                break
    
    return segments_of_steps


def find_matched_steps_for_segments_using_narration(
    step_embeddings, narration_embeddings, threshold=9, topK=3):
    
    steps_of_segments = defaultdict(list)
    steps_of_segments_rev = defaultdict(list)
    
    for segment_id in tqdm(range(narration_embeddings.shape[0])):
        
        cos_scores = util.cos_sim(step_embeddings, narration_embeddings[segment_id])

        sorted_values, sorted_indices = torch.sort(cos_scores[:,0], descending=True)
        
        for i in range(len(sorted_values)):
            if sorted_values[i] > threshold and i < topK:
                step_id = sorted_indices[i]
                
                steps_of_segments[segment_id].append(step_id)
                steps_of_segments_rev[step_id].append(segment_id)
            else:
                break
                
    return steps_of_segments, steps_of_segments_rev



def find_step_similarities_for_segments_using_video(
    args, logger,
    step_des_feats, segment_video_embeddings, segment_video_lookup_table):
    
    
    for segment_id in tqdm(range(segment_video_embeddings.shape[0])):
        v, cidx = segment_video_lookup_table[segment_id]
        save_path = os.path.join(args.segment_wikistep_sim_scores_v_path, v, 'segment_{}.npy'.format(cidx))
        if not os.path.exists(save_path):
            
            # dot product as similarity score
            sim_scores = np.einsum('ij,ij->i',
                                   step_des_feats, 
                                   segment_video_embeddings[segment_id][np.newaxis, ...])
            
            os.makedirs(os.path.join(args.segment_wikistep_sim_scores_v_path, v), exist_ok=True)
            np.save(save_path, sim_scores)
    return 


def find_step_similarities_for_segments_using_narration(
    args, logger,
    step_des_feats, segment_narration_embeddings, segment_narration_lookup_table):
    
    
    for segment_id in tqdm(range(segment_narration_embeddings.shape[0])):
        v, cidx = segment_narration_lookup_table[segment_id]
        save_path = os.path.join(args.segment_wikistep_sim_scores_n_path, v, 'segment_{}.npy'.format(cidx))
        if not os.path.exists(save_path):
            
            cos_scores = util.cos_sim(
                step_des_feats, 
                segment_narration_embeddings[segment_id])
            
            os.makedirs(os.path.join(args.segment_wikistep_sim_scores_n_path, v), exist_ok=True)
            np.save(save_path, cos_scores[:, 0].numpy())
    return 


    
def obtain_external_knowledge(args, logger):
    
    if not args.step_nodes_formed:
        # -- step_des_feats: step language description features
        step_des_feats = get_step_des_feats(args, logger, language_model="MPNet")

        # get nodes by removing step duplicates
        node2step, step2node = get_nodes_by_removing_step_duplicates(args, logger, step_des_feats)
    
    
    if not args.segment_wikistep_sim_scores_n_ready:
        
        # -- step_des_feats: step language description features
        step_des_feats = get_step_des_feats(args, logger, language_model="MPNet")

        segment_narration_embeddings, segment_narration_lookup_table = \
            get_segment_narration_feats(args, logger)

        find_step_similarities_for_segments_using_narration(
            args, logger, 
            step_des_feats, segment_narration_embeddings, segment_narration_lookup_table)


    if not args.segment_wikistep_sim_scores_v_ready:
        
        # -- step_des_feats: step language description features
        step_des_feats = get_step_des_feats(args, logger, language_model="S3D")
        
        segment_video_embeddings, segment_video_lookup_table = \
            get_segment_video_feats(args, logger)
        
        find_step_similarities_for_segments_using_video(
                args, logger, 
                step_des_feats, segment_video_embeddings, segment_video_lookup_table)
        
    return