import json
import random
import torch
import numpy as np
import logging

from pathlib import Path
from collections import defaultdict
from modules.utils import get_project_root


def kg_default_root():
    kg_root_path = Path(get_project_root()) / "dataset/data/kg/wikidata-graph"
    if not kg_root_path.exists():
        raise FileExistsError("Default directory is not found, please specify kg root")
    return kg_root_path


def construct_adj(**kwargs):     # graph is a triple
    logger = logging.getLogger("Construct adj graph")
    logger.info("construct adj graph begin")
    kg_root_path = kwargs.get("kg_root_path", kg_default_root())
    entity_adj_file, relation_adj_file = kg_root_path / "entity_adj.npy", kg_root_path / "relation_adj.npy"
    if entity_adj_file.exists() and relation_adj_file.exists():
        entity_adj, relation_adj = np.load(str(entity_adj_file)), np.load(str(relation_adj_file))
        logger.info("load adj matrix success!")
    else:
        graph = []  # creat graph from file
        for line in open(kg_root_path / "triple2id.txt", "r", encoding="utf-8"):
            split = line.split("\n")[0].split("\t")
            graph.append((int(split[0])+1, int(split[2])+1, int(split[1])+1))  # head, relation, tail
        kg = defaultdict(lambda: [])  # setup default knowledge graph
        for head, relation, tail in graph:
            # treat the KG as an undirected graph
            kg[head].append((tail, relation))
            kg[tail].append((head, relation))
        entity_num = int(open(kg_root_path / "entity2id.txt", encoding="utf-8").readline().split("\n")[0])
        entity_neighbor_num = kwargs.get("entity_neighbor_num", 20)
        zero_array = [0 for _ in range(entity_neighbor_num)]
        entity_adj = [zero_array] + [[] for _ in range(entity_num)]  # initial empty entity adj
        relation_adj = [zero_array] + [[] for _ in range(entity_num)]  # initial empty relation adj
        for key in kg.keys():
            for index in range(entity_neighbor_num):
                i = random.randint(0, len(kg[key])-1)
                entity_adj[int(key)].append(int(kg[key][i][0]))
                relation_adj[int(key)].append(int(kg[key][i][1]))
        np.save(str(entity_adj_file), entity_adj)
        np.save(str(relation_adj_file), relation_adj)
        logger.info("construct and save adj matrix success!")
    return entity_adj, relation_adj


def load_embeddings_from_text(file):
    """load embeddings from file"""
    return [np.array([float(i) for i in line.strip().split('\t')]) for line in open(file, "r", encoding="utf-8")]


def construct_entity_embedding(**kwargs):
    kg_root_path = kwargs.get("kg_root_path", kg_default_root())
    zero_array = np.zeros(kwargs.get("entity_embedding_dim", 100))  # zero embedding
    entity_embedding_file = kwargs.get("entity_embedding", kg_root_path / "entity2vecd100.vec")
    relation_embedding_file = kwargs.get("relation_embedding", kg_root_path / "relation2vecd100.vec")
    entity_embedding = [zero_array] + load_embeddings_from_text(entity_embedding_file)
    relation_embedding = [zero_array] + load_embeddings_from_text(relation_embedding_file)
    return torch.FloatTensor(np.array(entity_embedding)), torch.FloatTensor(np.array(relation_embedding))


def load_entities(**kwargs):
    kg_root_path = kwargs.get("kg_root_path", kg_default_root())
    entity2id = {}
    fp_entity2id = open(kg_root_path / "entity2id.txt", 'r', encoding='utf-8')
    entity_num = int(fp_entity2id.readline().split('\n')[0])  # read first line of entity2id file
    for line in fp_entity2id.readlines():
        entity, entity_id = line.strip().split('\t')
        entity2id[entity] = int(entity_id) + 1
    return entity2id


def load_entity_feature(title_entity: str, abstract_entity: str, entity_type_dict: dict):
    """position should be 1 (title entities) or 2 (abstract entities)"""
    title_entity_json, abstract_entity_json = json.loads(title_entity), json.loads(abstract_entity)
    entity_feature = defaultdict()
    for entity in title_entity_json:
        if entity["Type"] not in entity_type_dict:
            entity_type_dict[entity["Type"]] = len(entity_type_dict) + 1
        # entity feature contains frequency, position, and category index.
        entity_feature[entity["WikidataId"]] = [len(entity["OccurrenceOffsets"]), 1, entity_type_dict[entity["Type"]]]
    for entity in abstract_entity_json:
        if entity["WikidataId"] in entity_feature:
            feature = [entity_feature[entity["WikidataId"]][0] + len(entity["OccurrenceOffsets"]),  # count freq in abs
                       1, entity_type_dict[entity["Type"]]]  # this entity is regarded as title entity
        else:
            if entity["Type"] not in entity_type_dict:
                entity_type_dict[entity["Type"]] = len(entity_type_dict) + 1
            feature = [len(entity["OccurrenceOffsets"]), 2, entity_type_dict[entity["Type"]]]  # (freq, pos, type)
        entity_feature[entity["WikidataId"]] = feature
    return entity_feature
