# import networkx as nx
# import os
# import pickle
# import torch
# import numpy as np
# import faiss
# from PIL import Image
# import open_clip
# import logging

# # --- Setup Logging ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# log = logging.getLogger(__name__)

# # --- Global Variables ---
# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model = None
# clip_preprocess = None
# clip_tokenizer = None
# faiss_cloth = None
# faiss_person = None
# cloth_paths = None
# person_paths = None
# unified_image_graph = None

# def init_recommender(model_dir='model_data'):
#     """
#     Loads all necessary models, indices, and the graph into memory.
#     This should be called once when the Flask app starts.
#     """
#     global clip_model, clip_preprocess, clip_tokenizer, faiss_cloth, faiss_person, cloth_paths, person_paths, unified_image_graph

#     try:
#         log.info("Initializing recommender system...")
#         model_name = "ViT-B-32"
#         pretrained = "laion2b_s34b_b79k"

#         log.info(f"Loading OpenCLIP model: {model_name} on {device}")
#         clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
#             model_name, pretrained=pretrained, device=device
#         )
#         clip_model.eval()
#         clip_tokenizer = open_clip.get_tokenizer(model_name)

#         # Load FAISS Indices
#         faiss_cloth = faiss.read_index(os.path.join(model_dir, "faiss_openclip_cloth.idx"))
#         faiss_person = faiss.read_index(os.path.join(model_dir, "faiss_openclip_person.idx"))

#         # Load Path Mappings
#         with open(os.path.join(model_dir, "openclip_cloth_paths.pkl"), "rb") as f:
#             cloth_paths = pickle.load(f)
#         with open(os.path.join(model_dir, "openclip_person_paths.pkl"), "rb") as f:
#             person_paths = pickle.load(f)

#         # Load Unified Image Graph
#         graph_path = os.path.join(model_dir, "graphs", "unified_image_graph.pkl")
#         if not os.path.exists(graph_path):
#             raise FileNotFoundError(f"Graph file not found at {graph_path}. Please generate it first.")
        
#         with open(graph_path, "rb") as f:
#             unified_image_graph = pickle.load(f)
#         log.info(f"Unified Image Graph loaded with {len(unified_image_graph.nodes())} nodes.")

#         log.info("Recommender system initialized successfully.")

#     except Exception as e:
#         log.critical(f"FATAL ERROR during recommender initialization: {e}", exc_info=True)
#         raise

# def get_text_embedding(text_query):
#     """Generates a CLIP embedding for a text query."""
#     with torch.no_grad():
#         tokens = clip_tokenizer(text_query).to(device)
#         embedding = clip_model.encode_text(tokens)
#         embedding /= embedding.norm(dim=-1, keepdim=True)
#     return embedding.cpu().numpy()

# def get_image_embedding(image_path):
#     """Generates a CLIP embedding for a given image file path."""
#     try:
#         img = Image.open(image_path).convert("RGB")
#         img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
#         with torch.no_grad():
#             features = clip_model.encode_image(img_tensor)
#             features /= features.norm(dim=-1, keepdim=True)
#         return features.cpu().numpy()
#     except Exception as e:
#         log.error(f"Could not embed image {image_path}: {e}")
#         return None

# def recommend_from_graph(graph, start_nodes, num_recommendations=8, alpha=0.85):
#     """Performs recommendations using Personalized PageRank."""
#     if not start_nodes:
#         return []
    
#     valid_start_nodes = [node for node in start_nodes if node in graph]
#     if not valid_start_nodes:
#         log.warning(f"None of the {len(start_nodes)} start nodes were found in the graph.")
#         return []

#     personalization = {node: 1.0 / len(valid_start_nodes) for node in valid_start_nodes}
    
#     try:
#         pagerank_scores = nx.pagerank(graph, alpha=alpha, personalization=personalization, max_iter=100)
#     except Exception as e:
#         log.error(f"Error running PageRank: {e}")
#         return []

#     sorted_nodes = sorted(pagerank_scores.items(), key=lambda item: item[1], reverse=True)
    
#     recommendations = []
#     for node, score in sorted_nodes:
#         # Ensure we only recommend images and not the seed nodes themselves
#         if graph.nodes[node].get('type') == 'image' and node not in valid_start_nodes:
#             recommendations.append(os.path.basename(node))
#             if len(recommendations) >= num_recommendations:
#                 break
#     return recommendations

# def get_recommendations(query_input, input_type="text", num_recommendations=12, k_initial_neighbors=10):
#     """Main function to get recommendations for any input type."""
#     if unified_image_graph is None:
#         log.error("Recommender not ready: Unified Image Graph is not loaded.")
#         return []

#     initial_seed_nodes = []
#     query_emb = None
    
#     if input_type == "upload":
#         log.info(f"Getting recommendations for uploaded image: '{query_input}'")
#         query_emb = get_image_embedding(query_input)
#     elif input_type == "text":
#         log.info(f"Getting recommendations for text: '{query_input}'")
#         query_emb = get_text_embedding(query_input)
#     elif input_type == "image":
#         log.info(f"Getting recommendations for existing image: '{query_input}'")
#         # Find the full path for the given basename. This is crucial.
#         full_path_candidates = [p for p in (cloth_paths + person_paths) if os.path.basename(p) == query_input]
#         if not full_path_candidates:
#             log.warning(f"Image basename '{query_input}' not found in path mappings.")
#             return []
#         initial_seed_nodes.append(full_path_candidates[0])
#     else:
#         raise ValueError(f"Invalid input_type: {input_type}")

#     # If we have an embedding (from text or upload), find seed nodes using FAISS
#     if query_emb is not None:
#         _, I_cloth = faiss_cloth.search(query_emb, k_initial_neighbors)
#         _, I_person = faiss_person.search(query_emb, k_initial_neighbors)
        
#         for i in range(k_initial_neighbors):
#             if I_cloth[0][i] != -1: initial_seed_nodes.append(cloth_paths[I_cloth[0][i]])
#             if I_person[0][i] != -1: initial_seed_nodes.append(person_paths[I_person[0][i]])

#     if not initial_seed_nodes:
#         log.warning("No initial seed nodes could be found for recommendation.")
#         return []

#     # Get unique seed nodes
#     initial_seed_nodes = sorted(list(set(initial_seed_nodes)))
#     log.info(f"Found {len(initial_seed_nodes)} unique seed nodes for graph traversal.")

#     recommended_basenames = recommend_from_graph(
#         unified_image_graph,
#         initial_seed_nodes,
#         num_recommendations=num_recommendations,
#     )
    
#     # For existing image queries, ensure the query image itself is not in the results
#     if input_type == 'image':
#         recommended_basenames = [b for b in recommended_basenames if b != query_input]
    
#     return recommended_basenames[:num_recommendations]






import networkx as nx
import os
import pickle
import torch
import numpy as np
import faiss
from PIL import Image
import open_clip
import logging
import threading

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Models are now initially None ---
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = None
clip_preprocess = None
clip_tokenizer = None
faiss_cloth = None
faiss_person = None
cloth_paths = None
person_paths = None
unified_image_graph = None

# --- A lock to prevent multiple threads from loading models at the same time ---
recommender_lock = threading.Lock()

def _load_recommender_models():
    """Internal function to load all recommender models. DO NOT CALL DIRECTLY."""
    global clip_model, clip_preprocess, clip_tokenizer, faiss_cloth, faiss_person, cloth_paths, person_paths, unified_image_graph
    
    log.info("Loading recommender models into memory for the first time...")
    model_dir = 'model_data'
    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer(model_name)
    faiss_cloth = faiss.read_index(os.path.join(model_dir, "faiss_openclip_cloth.idx"))
    faiss_person = faiss.read_index(os.path.join(model_dir, "faiss_openclip_person.idx"))
    with open(os.path.join(model_dir, "openclip_cloth_paths.pkl"), "rb") as f:
        cloth_paths = pickle.load(f)
    with open(os.path.join(model_dir, "openclip_person_paths.pkl"), "rb") as f:
        person_paths = pickle.load(f)
    graph_path = os.path.join(model_dir, "graphs", "unified_image_graph.pkl")
    with open(graph_path, "rb") as f:
        unified_image_graph = pickle.load(f)
    log.info("Recommender models loaded successfully.")


def ensure_models_loaded():
    """
    Checks if recommender models are loaded. If not, loads them.
    This is thread-safe.
    """
    global clip_model
    if clip_model is None:
        with recommender_lock:
            # Check again inside the lock to ensure it wasn't loaded by another thread
            if clip_model is None:
                _load_recommender_models()

# --- All other functions (get_text_embedding, get_recommendations, etc.) now start by calling ensure_models_loaded() ---

def get_text_embedding(text_query):
    ensure_models_loaded()
    # ... rest of the function is unchanged
    with torch.no_grad():
        tokens = clip_tokenizer(text_query).to(device)
        embedding = clip_model.encode_text(tokens)
        embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()

def get_image_embedding(image_path):
    ensure_models_loaded()
    # ... rest of the function is unchanged
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = clip_model.encode_image(img_tensor)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()
    except Exception as e:
        log.error(f"Could not embed image {image_path}: {e}")
        return None

def recommend_from_graph(graph, start_nodes, num_recommendations=12, alpha=0.85):
    # This function doesn't need the check as it's called by get_recommendations
    # ... rest of the function is unchanged
    if not start_nodes: return []
    valid_start_nodes = [node for node in start_nodes if node in graph]
    if not valid_start_nodes: return []
    personalization = {node: 1.0 / len(valid_start_nodes) for node in valid_start_nodes}
    pagerank_scores = nx.pagerank(graph, alpha=alpha, personalization=personalization, max_iter=100)
    sorted_nodes = sorted(pagerank_scores.items(), key=lambda item: item[1], reverse=True)
    recommendations = []
    for node, score in sorted_nodes:
        if graph.nodes[node].get('type') == 'image' and node not in valid_start_nodes:
            recommendations.append(os.path.basename(node))
            if len(recommendations) >= num_recommendations:
                break
    return recommendations


def get_recommendations(query_input, input_type="text", num_recommendations=12, k_initial_neighbors=10):
    ensure_models_loaded()
    # ... rest of the function is unchanged
    initial_seed_nodes = []
    query_emb = None
    if input_type == "upload":
        log.info(f"Getting recommendations for uploaded image: '{query_input}'")
        query_emb = get_image_embedding(query_input)
    elif input_type == "text":
        log.info(f"Getting recommendations for text: '{query_input}'")
        query_emb = get_text_embedding(query_input)
    elif input_type == "image":
        log.info(f"Getting recommendations for existing image: '{query_input}'")
        full_path_candidates = [p for p in (cloth_paths + person_paths) if os.path.basename(p) == query_input]
        if not full_path_candidates: return []
        initial_seed_nodes.append(full_path_candidates[0])
    else: raise ValueError(f"Invalid input_type: {input_type}")
    if query_emb is not None:
        _, I_cloth = faiss_cloth.search(query_emb, k_initial_neighbors)
        _, I_person = faiss_person.search(query_emb, k_initial_neighbors)
        for i in range(k_initial_neighbors):
            if I_cloth[0][i] != -1: initial_seed_nodes.append(cloth_paths[I_cloth[0][i]])
            if I_person[0][i] != -1: initial_seed_nodes.append(person_paths[I_person[0][i]])
    if not initial_seed_nodes: return []
    initial_seed_nodes = sorted(list(set(initial_seed_nodes)))
    recommended_basenames = recommend_from_graph(unified_image_graph, initial_seed_nodes, num_recommendations=num_recommendations)
    if input_type == 'image': recommended_basenames = [b for b in recommended_basenames if b != query_input]
    return recommended_basenames[:num_recommendations]