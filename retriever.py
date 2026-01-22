import os
import re
import json
import networkx as nx
from networkx import dfs_edges
from collections import deque

from langchain_community.graphs import Neo4jGraph

from log_utils import *

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "12345678"

graph = Neo4jGraph(refresh_schema=False)

def create_node(node, embeddings, dataset="", id=""):
    query = """
    MERGE(n:Node {name: $name, embedding: $embedding})
    RETURN n;
    """
    node = graph.query(query, {"name": node, "id": id, "dataset": dataset, "embedding": embeddings})
    return node

def create_relation(head, tail, relation, embeddings, dataset="", id=""):
    query = """
    MATCH (h:Node {name: $head})
    MATCH (t:Node {name: $tail})
    MERGE (h)-[r:RELATION {name: $relation, embedding: $embedding, head:$head, tail:$tail}]->(t)
    RETURN r;
    """
    relation = graph.query(query, {"head": head, "tail": tail, "relation": relation, "id": id, "dataset": dataset,
                                   "embedding": embeddings})
    return relation

def all_relations_from_node(node):
    query = """
    MATCH (h:Node {name: $node})-[r:RELATION]->(t:Node)
    RETURN r.name as name
    """
    relations = graph.query(query, {"node": node})
    return [r['name'] for r in relations]

def create_node_index(index_name, dimensions):
    query = """
    CREATE VECTOR INDEX $index_name IF NOT EXISTS
    FOR (n:Node)
    ON n.embedding
    OPTIONS { indexConfig: {
     `vector.dimensions`: toInteger($dimensions),
     `vector.similarity_function`: 'cosine'
    }}
    """
    result = graph.query(query, {"index_name": index_name, "dimensions": dimensions})
    return result

def create_vector_index(index_name, dimensions):
    query = """
    CREATE VECTOR INDEX $index_name IF NOT EXISTS
    FOR ()-[r:RELATION]->() ON (r.embedding)
    OPTIONS { indexConfig: {
     `vector.dimensions`: toInteger($dimensions),
     `vector.similarity_function`: 'cosine'
    }}
    """
    result = graph.query(query, {"index_name": index_name, "dimensions": dimensions})
    return result

def search_similarity_relation(embedding, k, skip=0):
    query = """
    MATCH (h:Node)-[r:`RELATION`]->(t:Node)
    WHERE r.`embedding` IS NOT NULL AND size(r.`embedding`) = toInteger($dimension)
    WITH r as relationship, vector.similarity.cosine(r.`embedding`, $embedding) AS score ORDER BY score DESC
    SKIP toInteger($skip)
    LIMIT toInteger($k)
    RETURN score, relationship {.*} AS metadata
    """
    result = graph.query(query, {"embedding": embedding, "k": k, "skip": skip, "dimension": len(embedding)})
    return result

def search_similarity_relation_from_head(head, embedding, k, skip=0):
    query = """
    MATCH (h:Node {name: $head})-[r:RELATION]->(t:Node)
    WHERE r.embedding IS NOT NULL AND size(r.embedding) = toInteger($dimension)
    WITH r AS relationship, h, t, vector.similarity.cosine(r.embedding, $embedding) AS score
    WHERE score >= 0.5
    ORDER BY score DESC
    WITH collect({score: score, r: relationship, h: h.name, t: t.name}) AS results
    WITH results,
         CASE
            WHEN toInteger($skip) = 0
            THEN 1.0
            ELSE results[toInteger($skip) - 1].score
         END AS skip_score,
         CASE
            WHEN size(results) >= (toInteger($skip) + toInteger($k))
            THEN results[toInteger($skip) + toInteger($k) - 1].score
            ELSE 0.0
         END AS kth_score
    UNWIND results AS row
    WITH row, skip_score, kth_score
    WHERE row.score < skip_score AND row.score >= kth_score
    RETURN row.score AS score, properties(row.r) AS metadata, row.h AS head, row.t AS tail
    ORDER BY score DESC
    """
    result = graph.query(query,{"head": head, "embedding": embedding, "k": k, "skip": skip, "dimension": len(embedding)})
    return result

def search_similarity_relation_from_tail(tail, embedding, k, skip=0):
    query = """
    MATCH (h:Node)-[r:RELATION]->(t:Node {name: $tail})
    WHERE r.embedding IS NOT NULL AND size(r.embedding) = toInteger($dimension)
    WITH r AS relationship, h, t, vector.similarity.cosine(r.embedding, $embedding) AS score
    WHERE score >= 0.5
    ORDER BY score DESC
    WITH collect({score: score, r: relationship, h: h.name, t: t.name}) AS results
    WITH results,
         CASE
            WHEN toInteger($skip) = 0
            THEN 1.0
            ELSE results[toInteger($skip) - 1].score
         END AS skip_score,
         CASE
            WHEN size(results) >= (toInteger($skip) + toInteger($k))
            THEN results[toInteger($skip) + toInteger($k) - 1].score
            ELSE 0.0
         END AS kth_score
    UNWIND results AS row
    WITH row, skip_score, kth_score
    WHERE row.score < skip_score AND row.score >= kth_score
    RETURN row.score AS score, properties(row.r) AS metadata, row.h AS head, row.t AS tail
    ORDER BY score DESC
    """
    result = graph.query(query,{"tail": tail, "embedding": embedding, "k": k, "skip": skip, "dimension": len(embedding)})
    return result

def search_similarity_node(embedding, k, skip=0):
    query = """
    MATCH (n:Node)
    WHERE n.`embedding` IS NOT NULL AND size(n.`embedding`) = toInteger($dimension)
    WITH n as node, vector.similarity.cosine(n.`embedding`, $embedding) AS score ORDER BY score DESC
    SKIP toInteger($skip) 
    LIMIT toInteger($k)
    RETURN score, node {.*} AS metadata
    """
    result = graph.query(query, {"embedding": embedding, "k": k, "skip": skip, "dimension": len(embedding)})
    return result

def search_similarity_node_from_head(head, embedding, k):
    query = """
    MATCH (h:Node {name: $head})-[r:`RELATION`]->(t:Node)
    WHERE t.`embedding` IS NOT NULL AND size(t.`embedding`) = toInteger($dimension)
    WITH t as node, vector.similarity.cosine(t.`embedding`, $embedding) AS score ORDER BY score DESC LIMIT toInteger($k)
    RETURN score, node {.*} AS metadata
    """
    result = graph.query(query, {"head": head, "embedding": embedding, "k": k, "dimension": len(embedding)})
    return result

def delete_all_node():
    query = """
    MATCH (n) DETACH DELETE n
    """
    graph.query(query)

def delete_vector_index(index_name):
    query = """
    DROP INDEX $index_name IF EXISTS
    """
    graph.query(query, {"index_name": index_name})

def export_graph_to_json(filename):
    query = """
    CALL apoc.export.json.all($filename, {useTypes:true})
    """
    graph.query(query, {"filename": filename})

def import_json_to_graph(filename):
    query = """
    CALL apoc.import.json($filename)
    """
    graph.query(query, {"filename": filename})

def save_to_json(dataset, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            json_line = json.dumps(example, ensure_ascii=False)
            f.write(json_line + "\n")

def clear_neo4j():
    delete_vector_index("nodePlots")
    delete_vector_index("relationPlots")
    delete_all_node()

def build_neo4j(relations, embedding_model):
    unique_nodes = set()
    unique_rels = set()
    for h, r, t in relations:
        unique_nodes.add(h)
        unique_nodes.add(t)
        unique_rels.add(r)

    node_list = list(unique_nodes)
    rel_list = list(unique_rels)

    node_embeddings = embedding_model.embed_nodes(node_list)
    rel_embeddings = embedding_model.embed_relations(rel_list)

    node_embed_dict = {node: emb for node, emb in zip(node_list, node_embeddings)}
    rel_embed_dict = {rel: emb for rel, emb in zip(rel_list, rel_embeddings)}

    for h, r, t in relations:
        create_node(h, node_embed_dict[h])
        create_node(t, node_embed_dict[t])
        create_relation(h, t, r, rel_embed_dict[r])

    create_node_index("nodePlots", dimensions=embedding_model.get_dimension())
    create_vector_index("relationPlots", dimensions=embedding_model.get_dimension())

def reconstruct_graph_embedding(relations, embedding_model):
    clear_neo4j()
    build_neo4j(relations, embedding_model)

def search_relation_by_question(question_embedding, node, k, skip):
    result = search_similarity_relation_from_head(head=node, embedding=question_embedding, k=k, skip=skip)
    relations = [(p['head'], p['metadata']['name'], p['tail']) for p in result]
    return relations

def search_reverse_relation_by_question(question_embedding, node, k, skip):
    result = search_similarity_relation_from_tail(tail=node, embedding=question_embedding, k=k, skip=skip)
    reverse_relations = [(p['tail'], p['metadata']['name'], p['head']) for p in result]
    return reverse_relations

def path_to_string(path: list) -> str:
    result = ""
    for i, p in enumerate(path):
        if i == 0:
            h, r, t = p
            result += f"{h} -> {r} -> {t}"
        else:
            _, r, t = p
            result += f" -> {r} -> {t}"

    return result.strip()

def get_nodes_at_layer(G, root, n):
    """Returns a list of nodes at depth n from the root."""
    if n == 0:
        return root
    result = []
    for node in root:
        result += [child for parent, child in nx.bfs_edges(G, node, depth_limit=n) if
                   nx.shortest_path_length(G, node, child) == n]
    return result

def get_leaf_nodes(G):
    return [node for node in G.nodes if G.out_degree(node) == 0]

def get_paths_to_leaves(G, root):
    leaf_nodes = get_leaf_nodes(G)
    path_dic = {}

    for node in root:
        for leaf in leaf_nodes:
            try:
                for path in nx.all_shortest_paths(G, source=node, target=leaf):
                    formatted_path = []
                    path_key = ""
                    path_key += node
                    for i in range(len(path) - 1):
                        relation = G[path[i]][path[i + 1]]['relation']
                        path_key += "/" + relation
                        if i == 0:
                            formatted_path.append(f"{path[i]} -> {relation} -> {path[i + 1]}")
                        else:
                            formatted_path.append(f" -> {relation} -> {path[i + 1]}")
                    path_key += "/" + leaf
                    if path_key in path_dic:
                        continue
                    else:
                        path_dic[path_key] = " ".join(formatted_path)
            except nx.exception.NetworkXNoPath:
                # log_info(f"Could not find path for leaf {leaf} from node {node}")
                pass

    paths = list(path_dic.values())

    return paths

def is_grammatical(sentence):
    pattern = re.compile(r'^[a-zA-Z]\.[a-zA-Z0-9_]{1,10}$')

    if not pattern.match(sentence):
        return True
    else:
        return False

def get_retrieved_paths(G, root):
    leaf_nodes = get_leaf_nodes(G)
    path_dic = {}

    for node in root:
        estimated_paths = list(dfs_edges(G, source=node, depth_limit=4))
        # all_paths = nx.all_shortest_paths
        # if len(estimated_paths) > 100:
        if len(estimated_paths) > 25:
            all_paths = nx.all_shortest_paths
        else:
            all_paths = nx.all_simple_paths
        for leaf in leaf_nodes:
            if nx.has_path(G, node, leaf):
                for path in all_paths(G, source=node, target=leaf):
                    formatted_path = []
                    path_nodes = []
                    path_key = ""
                    path_key += node
                    path_nodes.append(node)
                    for i in range(len(path) - 1):
                        relation = G[path[i]][path[i + 1]]['relation']
                        path_key += "/" + relation
                        tail = path[i + 1]
                        path_nodes.append(tail)
                        if is_grammatical(tail):
                            path_key += "/" + tail
                        if i == 0:
                            formatted_path.append(f"{path[i]} -> {relation} -> {path[i + 1]}")
                        else:
                            formatted_path.append(f"-> {relation} -> {path[i + 1]}")
                    if is_grammatical(leaf):
                        path_key += "/" + leaf
                    complete_path = " ".join(formatted_path)
                    if all(G.nodes[n].get("fail", False) for n in path_nodes):
                        # log_info(f"path failed: {complete_path}")
                        continue
                    if path_key in path_dic:
                        continue
                    else:
                        path_dic[path_key] = complete_path

    paths = list(path_dic.values())

    return paths

def add_relations_to_graph(G, relations):
    for h, r, t in relations:
        try:
            if not nx.has_path(G, t, h):
                G.add_edge(h, t, relation=r.strip())
            else:
                pass
                # log_info(f"Skipping edge {h} -[{r}]-> {t} to avoid cycle.")
        except nx.exception.NodeNotFound:
            G.add_edge(h, t, relation=r.strip())

def update_fail_paths(G, fail_paths):
    fail_nodes = set()
    for fail_path in fail_paths:
        nodes = fail_path.split(" -> ")
        fail_nodes |= set(nodes)
    nx.set_node_attributes(G, {node: True for node in fail_nodes if node in G}, name="fail")

def set_entity_depth(G, entity, depth):
    if get_entity_depth(G, entity) > depth:
        nx.set_node_attributes(G, {entity: depth}, name="depth")

def get_entity_depth(G, entity):
    if entity in G:
        return G.nodes[entity].get("depth", 999)
    else:
        return 999

def set_entity_complete(G, entity, is_complete):
    nx.set_node_attributes(G, {entity: is_complete}, name="complete")

def get_entity_complete(G, entity):
    if entity in G:
        return G.nodes[entity].get("complete", False)
    else:
        return False

def get_nodes(relations):
    tails = list({t for _, _, t in relations})
    return tails

K_base = [10]
K_new = [10]
K_base_r = [10]
K_new_r = [10]
bidirectional = True

def do_retrieve(original_question, q_entity, sub_questions, a_entity, eval_func, embedding_model):
    global K_base, K_new, K_base_r, K_new_r, bidirectional

    log_info(f"original_question: {original_question}")
    log_info(f"key terms: {sub_questions[0]}")
    log_info(f"q_entity: {q_entity}")
    log_info(f"a_entity: {a_entity}")
    retrieve_finish = False
    K_base = [10]
    K_new = [10]
    K_base_r = [10]
    K_new_r = [10]
    loop_counter = 0
    MAX_LOOP = 2
    retrieved_knowledge = ""
    G = nx.DiGraph()
    for entity in q_entity:
        G.add_node(entity)
        set_entity_depth(G, entity, 0)

    queue = deque()
    for entity in q_entity:
        queue.append(entity)

    question_embedding = embedding_model.embed_question(sub_questions[0])
    loop_counter += 1
    while queue:
        node = queue.popleft()
        depth = get_entity_depth(G, node)
        if depth < len(K_base) and not get_entity_complete(G, node):
            k = K_base[depth]
            skip = 0
            relations = search_relation_by_question(question_embedding, node, k, skip=skip)
            log_info(f"search question: {original_question}, node: {node} at depth: {depth}")
            log_info(f"search from {skip + 1} to {skip + k}")
            log_info(f"search relations: {relations}")
            add_relations_to_graph(G, relations)
            nodes = get_nodes(relations)
            for n in nodes:
                set_entity_depth(G, n, depth + 1)
            for n in nodes:
                if not get_entity_complete(G, n):
                    queue.append(n)
            if bidirectional:
                k_r = K_base_r[depth]
                reverse_skip = 0
                reverse_relations = search_reverse_relation_by_question(question_embedding, node, k_r, skip=reverse_skip)
                log_info(f"search reverse from {reverse_skip + 1} to {reverse_skip + k_r}")
                log_info(f"search reverse relations: {reverse_relations}")
                add_relations_to_graph(G, reverse_relations)
                nodes_r = get_nodes(reverse_relations)
                for n in nodes_r:
                    set_entity_depth(G, n, depth + 1)
                for n in nodes_r:
                    if not get_entity_complete(G, n):
                        queue.append(n)

            set_entity_complete(G, node, True)

    while retrieve_finish is False and loop_counter <= MAX_LOOP:
        list_of_paths = get_retrieved_paths(G, q_entity)
        if len(list_of_paths) > 0:
            retrieved_knowledge = ""
            retrieved_knowledge += "\n".join(list_of_paths)
            retrieved_knowledge += "\n"
        log_info(f"original_question: {original_question}")
        log_info(f"retrieved knowledge: {retrieved_knowledge}")
        log_info(f"retrieve count: {loop_counter}")
        if len(list_of_paths) > 0:
            confidence, sufficiency = eval_func(original_question, retrieved_knowledge)
            if confidence >= 7 or sufficiency:
                retrieve_finish = True
            elif confidence >= 5:
                pass
            else:
                update_fail_paths(G, list_of_paths)
                discard_truths = []
                for a in a_entity:
                    if a in retrieved_knowledge:
                        discard_truths.append(a)
                if len(discard_truths) > 0:
                    log_info(f"discard truth: {discard_truths}")

        if retrieve_finish is False:
            loop_counter += 1
            do_supplemental_retrieve(original_question, question_embedding, G)
            K_base = [a + b for a, b in zip(K_base, K_new)]
            K_base_r = [a + b for a, b in zip(K_base_r, K_new_r)]

    return retrieved_knowledge, loop_counter

def do_supplemental_retrieve(original_question, question_embedding, G):
    queue = deque()
    for entity in list(G.nodes()):
        queue.append(entity)
        set_entity_complete(G, entity, False)

    while queue:
        node = queue.popleft()
        depth = get_entity_depth(G, node)
        if depth < len(K_new) and not get_entity_complete(G, node):
            k_new = K_new[depth]
            k_base = 0
            if depth < len(K_base):
                k_base = K_base[depth]
            skip = k_base
            relations = search_relation_by_question(question_embedding, node, k_new, skip=skip)
            log_info(f"search question: {original_question}, node: {node} at depth {depth}")
            log_info(f"search from {skip + 1} to {skip + k_new}")
            log_info(f"search relations: {relations}")
            add_relations_to_graph(G, relations)
            nodes = get_nodes(relations)
            for n in nodes:
                set_entity_depth(G, n, depth + 1)
            for n in nodes:
                if not get_entity_complete(G, n):
                    queue.append(n)
            if bidirectional:
                k_new_r = K_new_r[depth]
                k_base_r = 0
                if depth < len(K_base_r):
                    k_base_r = K_base_r[depth]
                reverse_skip = k_base_r
                reverse_relations = search_reverse_relation_by_question(question_embedding, node, k_new_r, skip=reverse_skip)
                log_info(f"search reverse from {reverse_skip + 1} to {reverse_skip + k_new_r}")
                log_info(f"search reverse relations: {reverse_relations}")
                add_relations_to_graph(G, reverse_relations)
                nodes_r = get_nodes(reverse_relations)
                for n in nodes_r:
                    set_entity_depth(G, n, depth + 1)
                for n in nodes_r:
                    if not get_entity_complete(G, n):
                        queue.append(n)

            set_entity_complete(G, node, True)

