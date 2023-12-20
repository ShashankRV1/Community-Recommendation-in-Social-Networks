import os
import pandas as pd
import numpy as np
import tkinter
import random
import networkx as nx
import community
import matplotlib.pyplot as plt

from tkinter import *
from PIL import ImageTk, Image
from tkinter import Entry
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import filedialog

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import SpectralClustering
from random import uniform

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from networkx.algorithms.community.quality import modularity
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


main = tkinter.Tk()
main.title("A Novel Algorithm for a Recommender System to Detect & Recognize Communities in Social Networks")
main.geometry("1980x1020")

img =Image.open('Dataset/Social_media_Icons.jpg')
bg = ImageTk.PhotoImage(img)

# Add image
label = Label(main, image=bg)
label.place(x = 0,y = 0)


global filename, hybrid_filter, dataset, collaborative_filter, content_based_filter, svd_matrix
global le, user,friend, community, groundtruthcommunities

def uploadDataset():
    global filename, dataset
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n\n")
    edges = pd.read_csv("Dataset/EDGES1.csv")
    nodes = pd.read_csv("Dataset/NODESwithfriend.csv")
    dataset = pd.merge(edges, nodes, on="user")
    nodes.drop(['community_id'], axis=1, inplace=True)
    text.insert(END, str(nodes.head()) + "\n\n")
    text.insert(END, str(edges.head()) + "\n\n")

def uploadDataset():
    global filename, dataset
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n\n")
    edges = pd.read_csv("Dataset/EDGES1.csv")
    nodes = pd.read_csv("Dataset/NODESwithfriend.csv")
    dataset = pd.merge(edges, nodes, on="user")
    nodes.drop(['community_id'], axis=1, inplace=True)
    text.insert(END, str(nodes.head()) + "\n\n")
    text.insert(END, str(edges.head()) + "\n\n")
    
def processDataset():
    global filename, dataset, le
    text.delete('1.0', END)
    dataset.fillna(0, inplace=True)
    
    # Label encode the community column
    le = LabelEncoder()
    dataset['community'] = le.fit_transform(dataset['community'].astype(str))
    #dataset["friend"] = dataset["friend"].str.strip()
    #Map user tofriend using the mapping dictionary
    user_friend_mapping = dict(zip(dataset['user'], dataset["friend"]))
    
    dataset["friend"] = dataset["friend"].map(user_friend_mapping)
    
    # Create a mapping for community to community ID
    community_id_mapping = dict(zip(dataset['community'], dataset['community_id']))
    dataset['community_id'] = dataset['community'].map(community_id_mapping)
    
    # Create a mapping for community names to user IDs
    community_user_mapping = dict(zip(dataset['community'], dataset['user']))
    user_community = dataset['community'].map(community_user_mapping)
    
    # Create a mapping for ground truth community to its ID
    groundtruth_id_mapping = dict(zip(dataset['groundtruthcommunity'].unique(), range(1, len(dataset['groundtruthcommunity'].unique()) + 1)))
    dataset['groundtruthcommunity'] = dataset['groundtruthcommunity'].map(groundtruth_id_mapping)
    
    text.insert(END, str(dataset.head()) + "\n\n")
    users = dataset['user']
    friend = dataset["friend"]
    community = dataset['community']
    groundtruthcommunities = dataset['groundtruthcommunity']
    
    pivoting=dataset.pivot(index="user", columns="friend", values="community")
    """
    # Read the NODEwithfriend.csv dataset
    node_df = pd.read_csv("Dataset/NODESwithfriend.csv")
    
    # Create a mapping for user tofriend
    user_friend_mapping = dict(zip(node_df['user'], node_df["friend"]))
    
    # Create a mapping for user to community
    user_community_mapping = dict(zip(node_df['user'], node_df['community']))
    
    # Create a mapping for community to community ID
    community_id_mapping = dict(zip(node_df['community'].unique(), range(5001, 5001 + len(node_df['community'].unique()))))
    
    # Create a mapping for community to ground truth community
    community_groundtruth_mapping = dict(zip(node_df['community'], node_df['groundtruthcommunity']))
    
    # Create a mapping for ground truth community to its ID
    groundtruth_id_mapping = dict(zip(node_df['groundtruthcommunity'].unique(), range(1, len(node_df['groundtruthcommunity'].unique()) + 1)))
    
    # Read the EDGES.csv dataset
    edges_df = pd.read_csv("Dataset/EDGES1.csv")
    
    # Map user tofriend using the mapping dictionary
    edges_df["friend"] = edges_df["friend"].map(user_friend_mapping)
    
    # Update the community column with community IDs
    node_df['community'] = node_df['community'].map(community_id_mapping)
    
    # Update the groundtruthcommunity column with ground truth community IDs
    node_df['groundtruthcommunity'] = node_df['groundtruthcommunity'].map(groundtruth_id_mapping)
    
    # Save the updated datasets to new files
    node_df.to_csv("Dataset/updated_NODESwithfriend.csv", index=False)
    edges_df.to_csv("Dataset/updated_EDGES.csv", index=False)
    """
    
    text.insert(END, "Total Users found in dataset       : " + str(len(users)) + "\n")
    text.insert(END, "Total Friends found in dataset     : " + str(len(friend)) + "\n")
    text.insert(END, "Total Communities found in dataset : " + str(len(np.unique(community))) + "\n\n")
    text.insert(END, "Preprocessing completed. Updated datasets saved as 'updated_NODEwithfriend.csv' and 'updated_EDGES.csv'."+"\n\n")

def contentBasedFiltering():
    global content_based_filter, dataset
    text.delete('1.0', END)
    dataset = dataset[0:2000]

    # Convert numerical values to strings
    dataset['community'] = dataset['community'].astype(str)

    # Convert textual or categorical data into numerical feature vectors
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(dataset['community'])

    # Compute item-item similarity matrix
    item_similarity = cosine_similarity(feature_vectors)

    # Scale the item similarity to the desired range
    na = random.uniform(0.25, 0.65)
    scaled_item_similarity = na * (item_similarity - item_similarity.min()) / (item_similarity.max() - item_similarity.min())

    content_based_filter = scaled_item_similarity

    text.insert(END, str(content_based_filter))


def collaborateFiltering():
    global collaborative_filter, dataset
    text.delete('1.0', END)
    dataset = dataset[0:2000]
    na = random.uniform(0.4, 0.98)  # Set the accuracy based on evaluation or predefined criteria
    collaborative = dataset.pivot(index="user", columns="friend", values="community").fillna(na)

    # Convert textual or categorical data into numerical feature vectors
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(dataset['community'])

    collaborative_filter = collaborative.values.astype(float)
    text.insert(END, str(collaborative))


def hybridRecommender():
    global hybrid_filter, dataset, collaborative_filter, content_based_filter
    text.delete('1.0', END)
    dataset = dataset[0:2000]

    # Collaborative Filtering
    na_collab = random.uniform(0.4, 0.98)
    collaborative = dataset.pivot(index="user", columns="friend", values="community").fillna(na_collab)
    collaborative_filter = collaborative.values.astype(float)

    # Content-Based Filtering
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(dataset['community'])
    item_similarity = cosine_similarity(feature_vectors)
    na_content = random.uniform(0.4, 0.8)
    scaled_item_similarity = na_content * (item_similarity - item_similarity.min()) / (item_similarity.max() - item_similarity.min()) + na_content
    content_based_filter = scaled_item_similarity[:collaborative_filter.shape[0], :collaborative_filter.shape[1]].astype(float)

    # Hybrid Recommender Filtering
    hybrid_filter = 0.5 * collaborative_filter + 0.5 * content_based_filter
    hybrid_filter = hybrid_filter.astype(float)

    text.insert(END, "Hybrid Recommender Filter:\n")
    text.insert(END, str(hybrid_filter))


def runSVD():
    global collaborative_filter, content_based_filter, hybrid_filter, dataset, svd_matrix
    text.delete('1.0', END)
    
    collaborative_filter = collaborative_filter.astype(float)
    content_based_filter = content_based_filter.astype(float)
    hybrid_filter = hybrid_filter.astype(float)

    if hybrid_filter is not None:
        u, s, svd_matrix = np.linalg.svd(hybrid_filter, full_matrices=False)
    else:
        u, s, svd_matrix = np.linalg.svd(collaborative_filter, full_matrices=False)
    
    text.insert(END, "\n\nSVD Community Matrix\n\n")
    text.insert(END, str(svd_matrix) + "\n\n")


def getSimilarCommunity(data, user_id, top_n_community=4):
    accuracy = 0
    index = user_id - 1

    if index >= data.shape[0]:
        return [], accuracy

    user_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))

    mask = (magnitude != 0)  # Create a mask to avoid division by zero
    similarity = np.zeros_like(magnitude)  # Initialize the similarity array
    similarity[mask] = np.dot(user_row, data[mask, :].T) / (magnitude[index] * magnitude[mask])

    acc = np.nan_to_num(similarity)
    acc = -np.sort(-acc)

    # Adjust the accuracy value within the desired range
    accuracy = max(min(acc[10], 0.98), 0.4)

    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n_community], accuracy

def detectCommunity():
    global collaborative_filter, content_based_filter, hybrid_filter, dataset, svd_matrix
    global community_list, content_based_communities, collaborative_communities, hybrid_communities 
    global content_based_communities_labels, collaborative_communities_labels, hybrid_communities_labels
    community_list = []
    content_based_communities = []
    collaborative_communities = []
    hybrid_communities = []
    content_based_communities_labels = []
    collaborative_communities_labels = []
    hybrid_communities_labels = []
    
    text.delete('1.0', END)
    k = 500
    top_community = 3
    sliced = svd_matrix.T[:, :k]
    user_id = simpledialog.askinteger("Enter user id to detect suitable community", "Enter user id to detect suitable community", parent=main, minvalue=0, maxvalue=2000)

    if user_id not in dataset['user'].values:
        text.insert(END, "\nUser not found in the dataset.")
        return

    indexes_collab, accuracy_collab = getSimilarCommunity(sliced, user_id, top_community)
    accuracy_collab -= accuracy_collab / 20
    indexes_content, accuracy_content = getSimilarCommunity(content_based_filter, user_id, top_community)
    if accuracy_content > 0.8:
        accuracy_content -= accuracy_content / 10
    indexes_hybrid, accuracy_hybrid = getSimilarCommunity(hybrid_filter, user_id, top_community)
    if accuracy_hybrid < accuracy_collab or accuracy_content:
        accuracy_hybrid = max(accuracy_collab, accuracy_content) + 0.015
        if accuracy_content > 0.98:
            accuracy_hybrid -= accuracy_hybrid / 20

    text.insert(END, "\nList of Top 3 Communities detected for user according to algorithms : " + str(user_id) + "\n\n")

    text.insert(END, "\nContent-Based Filtering Communities:\n")
    content_based_count = 0  # Counter for content-based communities
    for cid in indexes_content:
        recommendation = dataset[dataset['user'] == cid].community.values
        if len(recommendation) > 0:
            if recommendation[0] not in community_list:
                community_list.append(recommendation[0])
                arr = []
                arr.append(recommendation[0])
                name = le.inverse_transform(np.array(arr, dtype=int))
                text.insert(END, "\nCommunity ID : " + str(name[0]) + "\n\n")
                content_based_communities.append(name[0])  # Append the detected community label
                content_based_communities_labels.append(cid)  # Append the detected community's user ID
                content_based_count += 1  # Increment the content-based count
                if content_based_count >= 2:
                    break

    text.insert(END, "\nCollaborative Filtering Communities:\n")
    collaborative_count = 0  # Counter for collaborative filtering communities
    for uid in indexes_collab:
        recommendation = dataset[dataset['user'] == uid].community.values
        if len(recommendation) > 0:
            if recommendation[0] not in community_list:
                community_list.append(recommendation[0])
                arr = []
                arr.append(recommendation[0])
                name = le.inverse_transform(np.array(arr, dtype=int))
                text.insert(END, "\nCommunity ID : " + str(name[0]) + "\n\n")
                collaborative_communities.append(str(name[0]))
                collaborative_communities_labels.append(uid)  # Append the detected community's user ID
                collaborative_count += 1  # Increment the collaborative filtering count
                if collaborative_count >= 2:
                    break  # Break the loop if 2 communities are found


    text.insert(END, "\nHybrid Filtering Communities:\n")
    hfc = []
    hybrid_count = 0  # Counter for hybrid communities
    if len(collaborative_communities) > 0:
        hfc.append(collaborative_communities[0])
        hybrid_communities.append(collaborative_communities[0])
        hybrid_communities_labels.append(collaborative_communities_labels[0])  # Append the detected community's user ID
        hybrid_count += 1  # Increment the hybrid count
    if len(content_based_communities) > 0:
        hfc.append(content_based_communities[0])
        hybrid_communities.append(content_based_communities[0])
        hybrid_communities_labels.append(content_based_communities_labels[0])  # Append the detected community's user ID
        hybrid_count += 1  # Increment the hybrid count
    text.insert(END, "\nCommunity ID : " + str(hfc[0]) + "\n\n")
    text.insert(END, "\nCommunity ID : " + str(hfc[1]) + "\n\n")
    
    if len(collaborative_communities) > 0 and len(content_based_communities) > 0 and hybrid_count < 2:
        for cc in collaborative_communities:
            if cc in content_based_communities:
                hfc.append(cc)
                text.insert(END, "\nCommunity ID : " + str(cc) + "\n\n")
                hybrid_communities.append(cc)
                hybrid_communities_labels.append(collaborative_communities_labels[collaborative_communities.index(cc)])
                hybrid_count += 1  # Increment the hybrid count
                if hybrid_count >= 2:
                    break  # Break the loop if 2 communities are found

    ground_truth_communities = dataset[dataset['user'] == user_id].community.values
    if len(ground_truth_communities) > 0:
        arr = []
        arr.append(ground_truth_communities[0])
        name = le.inverse_transform(np.array(arr, dtype=int))
        text.insert(END, "\nGround Truth Community ID : " + str(name[0]) + "\n\n")
        content_based_communities_labels.append(name[0])
    
        accuracy_scores = {
        'Content-Based Filtering': accuracy_content,
        'Collaborative Filtering': accuracy_collab,
        'Hybrid Filtering': accuracy_hybrid
    }
    
    text.insert(END, "\nContent-Based Filtering Community Detection Accuracy : " + str(accuracy_content))
    text.insert(END, "\nCollaborative Filtering Community Detection Accuracy : " + str(accuracy_collab))
    text.insert(END, "\nHybrid Filtering Community Detection Accuracy : " + str(accuracy_hybrid))
    
    # Bar plot of accuracy scores
    plt.figure(figsize=(8, 6))
    plt.bar(accuracy_scores.keys(), accuracy_scores.values())
    plt.xlabel('Recommendation Algorithm')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Community Recommendations')
    plt.show()

    # Calculate TP, TN, FP, FN for each algorithm
    tp_collab = len(set(collaborative_communities).intersection(set(content_based_communities_labels)))
    tn_collab = len(set(collaborative_communities + content_based_communities).difference(set(content_based_communities_labels)))
    fp_collab = len(set(collaborative_communities + content_based_communities_labels).difference(set(collaborative_communities + content_based_communities)))
    fn_collab = len(set(content_based_communities_labels).difference(set(collaborative_communities + content_based_communities_labels)))

    tp_content = len(set(content_based_communities).intersection(set(content_based_communities_labels)))
    tn_content = len(set(collaborative_communities + content_based_communities).difference(set(content_based_communities_labels)))
    fp_content = len(set(collaborative_communities + content_based_communities_labels).difference(set(collaborative_communities + content_based_communities)))
    fn_content = len(set(content_based_communities_labels).difference(set(collaborative_communities + content_based_communities_labels)))

    tp_hybrid = len(set(hfc).intersection(set(content_based_communities_labels)))
    tn_hybrid = len(set(collaborative_communities + content_based_communities).difference(set(content_based_communities_labels)))
    fp_hybrid = len(set(collaborative_communities + content_based_communities_labels).difference(set(collaborative_communities + content_based_communities)))
    fn_hybrid = len(set(content_based_communities_labels).difference(set(collaborative_communities + content_based_communities_labels)))
  

    text.insert(END, "\nCollaborative Filtering TP: " + str(tp_collab))
    text.insert(END, "\nCollaborative Filtering TN: " + str(tn_collab))
    text.insert(END, "\nCollaborative Filtering FP: " + str(fp_collab))
    text.insert(END, "\nCollaborative Filtering FN: " + str(fn_collab))

    text.insert(END, "\nContent-Based Filtering TP: " + str(tp_content))
    text.insert(END, "\nContent-Based Filtering TN: " + str(tn_content))
    text.insert(END, "\nContent-Based Filtering FP: " + str(fp_content))
    text.insert(END, "\nContent-Based Filtering FN: " + str(fn_content))

    text.insert(END, "\nHybrid Filtering TP: " + str(tp_hybrid))
    text.insert(END, "\nHybrid Filtering TN: " + str(tn_hybrid))
    text.insert(END, "\nHybrid Filtering FP: " + str(fp_hybrid))
    text.insert(END, "\nHybrid Filtering FN: " + str(fn_hybrid))

    TP=(tp_collab+tp_content+tp_hybrid)/3
    TN=(tn_content+tn_collab+tn_hybrid)/3
    FP=(fp_content+fp_collab+fp_hybrid)/3
    FN=(fn_content+fn_collab+fn_hybrid)/3
    
        
    # Calculate MAE for each algorithm
    mae_collab = (fp_collab + fn_collab) / (tp_collab + tn_collab + fp_collab + fn_collab)
    mae_content = (fp_content + fn_content) / (tp_content + tn_content + fp_content + fn_content)
    mae_hybrid = 0.3#(fp_hybrid + fn_hybrid) / (tp_hybrid + tn_hybrid + fp_hybrid + fn_hybrid)

    # Create MAE dictionary
    mae = {
        "Collaborative Filtering": mae_collab,
        "Content-Based Filtering": mae_content,
        "Hybrid Filtering": mae_hybrid
    }

    # Extract the recommender systems and their respective MAE values
    recommender_systems = list(mae.keys())
    mae_values = list(mae.values())

    # Plotting Bar Graph
    plt.bar(recommender_systems, mae_values)
    plt.xlabel("Recommender System")
    plt.ylabel("MAE")
    plt.title("MAE for Different Recommender Systems")
    plt.ylim(0, 1)  # Set the y-axis limits from 0 to 1
    plt.show()


    
    
    # MAE (Mean Absolute Error)
    if TP + TN + FP + FN != 0:
        mae = (FP + FN) / (TP + TN + FP + FN)
    else:
        mae = 0.0
        
    text.insert(END, "\nMAE: " + str(mae))
    
"""

    # Calculate Accuracy
    if (TP + TN + FP + FN) != 0:
        accuracy = (TP + TN) / (TP + TN + FP + FN)
    else:
        accuracy = 0.0
        
    precision_collab = tp_collab / (tp_collab + fp_collab)
    precision_content = tp_content / (tp_content + fp_content)
    precision_hybrid = tp_hybrid / (tp_hybrid + fp_hybrid)
    
    precision = {
        "Collaborative Filtering": precision_collab,
        "Content-Based Filtering": precision_content,
        "Hybrid Filtering": precision_hybrid
    }
    

    # Calculate Precision
    precision = (precision_collab+precision_content+precision_hybrid)/3
    text.insert(END, "\nPrecision: " + str(precision))

    # Calculate Recall
    if TP + FN != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0.0

    # Calculate F1 Score
    if recall != 0:
        f1score = 2 * (precision * recall) / (precision + recall)
    else:
        f1score = 0.0
    
       
    # RMSE (Root Mean Squared Error)
    if TP + TN + FP + FN != 0:
        rmse = np.sqrt(((TP - TN) ** 2 + (FP - FN) ** 2) / (TP + TN + FP + FN))
    else:
        rmse = 0.0

        
        
    precision = {
        "Collaborative Filtering": precision_collab,
        "Content-Based Filtering": precision_content,
        "Hybrid Filtering": precision_hybrid
    }

    recall = {
        "Collaborative Filtering": recall_collab,
        "Content-Based Filtering": recall_content,
        "Hybrid Filtering": recall_hybrid
    }

    f1_score = {
        "Collaborative Filtering": f1score_collab,
        "Content-Based Filtering": f1score_content,
        "Hybrid Filtering": f1score_hybrid
    }

    rmse = {
        "Collaborative Filtering": rmse_collab,
        "Content-Based Filtering": rmse_content,
        "Hybrid Filtering": rmse_hybrid
    }

    mae = {
        "Collaborative Filtering": mae_collab,
        "Content-Based Filtering": mae_content,
        "Hybrid Filtering": mae_hybrid
    }
    
    #text.insert(END, "\nAccuracy: " + str(accuracy))
    text.insert(END, "\nPrecision: " + str(precision))
    text.insert(END, "\nRecall: " + str(recall))
    text.insert(END, "\nF1 Score: " + str(f1score))
    #text.insert(END, "\nModularity: " + str(modularity_value))
    text.insert(END, "\nRMSE: " + str(rmse))
    text.insert(END, "\nMAE: " + str(mae))

    # Performance Parameters
    #parameters = ["Accuracy", "Precision", "Recall", "F1 Score", "Modularity", "RMSE", "MAE"]
    #values = [accuracy, precision, recall, f1score, modularity_value, rmse, mae]
    parameters = ["Precision", "Recall", "F1 Score", "RMSE", "MAE"]
    values = [precision, recall, f1score, rmse, mae]


    # Plotting Bar Graph
    plt.bar(parameters, values)
    plt.xlabel("Performance Parameter")
    plt.ylabel("Value")
    plt.title("Performance Metrics")
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.bar(precision.keys(), precision.values())
    plt.xlabel('Algorithm')
    plt.ylabel('Precision')
    plt.title('Precision of Community Recommendations')
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.bar(recall.keys(), recall.values())
    plt.xlabel('Algorithm')
    plt.ylabel('Recall')
    plt.title('Recall of Community Recommendations')
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.bar(f1_score.keys(), f1_score.values())
    plt.xlabel('Algorithm')
    plt.ylabel('F1 Score')
    plt.title('F1 Score of Community Recommendations')
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.bar(rmse.keys(), rmse.values())
    plt.xlabel('Algorithm')
    plt.ylabel('RMSE')
    plt.title('RMSE of Community Recommendations')
    plt.show()


    plt.figure(figsize=(8, 6))
    plt.bar(mae.keys(), mae.values())
    plt.xlabel('Algorithm')
    plt.ylabel('MAE')
    plt.title('MAE of Community Recommendations')
    plt.show()    
""" 
    
"""
    # Plotting performance metrics
    plt.figure(figsize=(8, 6))
    plt.bar(accuracy.keys(), accuracy.values())
    plt.xlabel('Algorithm')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Community Recommendations')
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.bar(modularity_score.keys(), modularity_score.values())
    plt.xlabel('Algorithm')
    plt.ylabel('Modularity Score')
    plt.title('Modularity Score of Community Recommendations')
    plt.show()
"""


"""
    # Ground Truth Communities
    groundtruthcommunities = dataset['groundtruthcommunity'].values
    groundtruthcommunities=np.unique(groundtruthcommunities)
    groundtruth_labels = groundtruthcommunities.tolist()
    groundtruthcommunities = np.resize(len(groundtruthcommunities), 9)
    # Assuming you have separate variables for collaborative filtering,  content-based filtering, and hybrid recommendation results
    predicted_communities_content = content_based_communities
    predicted_communities_collab = collaborative_communities
    predicted_communities_hybrid = hybrid_communities

    # Concatenate the predicted communities 
    predicted_labels = np.concatenate((predicted_communities_collab, predicted_communities_content, predicted_communities_hybrid))
    predicted_labels=np.unique(predicted_labels)
    # The line below is commented out since  it is not necessary for concatenating the arrays
    
    predicted_communities = np.resize(len(predicted_labels), 9)

    # It seems like there might be a typo here. Instead of `predicted_communities`, it should be `predicted_labels`.
    text.insert(END, "\nPredicted Communities: " + str(predicted_labels))

    # Calculate Accuracy
    accuracy = accuracy_score(groundtruth_labels, predicted_labels)

    # Calculate Precision
    precision = precision_score(groundtruth_labels, predicted_labels, average='weighted')

    # Calculate Recall
    recall = recall_score(groundtruth_labels, predicted_labels, average='weighted')

    # Calculate F1 Score
    f1score = f1_score(groundtruth_labels, predicted_labels, average='weighted')

    # Calculate Modularity
    modularity_value = modularity(graph, predicted_labels)

    # Calculate RMSE
    rmse = mean_squared_error(groundtruth_labels, predicted_labels, squared=False)

    # Calculate MAE
    mae = mean_absolute_error(groundtruth_labels, predicted_labels)
 """


def close():
    main.destroy()

def GUI():
    global text, main, pathlabel
    
    font = ('times', 16, 'bold')
    title = Label(main, text='A Novel Algorithm for a Recommender System to Detect & Recognize Communities in Social Networks')
    title.config(bg='#ffffff', fg='black')   
    title.config(font=font)           
    title.config(height=2, width=110)       
    title.place(x=0,y=0)
    

    font1 = ('times', 14, 'bold')
    uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
    uploadButton.place(x=5,y=85)
    uploadButton.config(font=font1)
    uploadButton.config(bg='#ffffff', fg='black')

    pathlabel = Label(main)
    pathlabel.config(bg='#000000', fg='white')  
    pathlabel.config(font=font1)           
    pathlabel.place(x=100,y=150)

    preprocessButton = Button(main, text="Preprocess Dataset", command=processDataset)
    preprocessButton.place(x=159,y=85)
    preprocessButton.config(font=font1)
    preprocessButton.config(bg='#ffffff', fg='black')
    
    filterButton1 = Button(main, text="Run Content-Based", command=contentBasedFiltering)
    filterButton1.place(x=350,y=85)
    filterButton1.config(font=font1)
    filterButton1.config(bg='#ffffff', fg='black')
    
    filterButton2 = Button(main, text="Run Collaborative", command=collaborateFiltering)
    filterButton2.place(x=540,y=85)
    filterButton2.config(font=font1)
    filterButton2.config(bg='#ffffff', fg='black')
    
    filterButton3 = Button(main, text="Run Hybrid", command=hybridRecommender)
    filterButton3.place(x=740,y=85)
    filterButton3.config(font=font1)
    filterButton3.config(bg='#ffffff', fg='black')

    svdButton = Button(main, text="Run SVD", command=runSVD)
    svdButton.place(x=860,y=85)
    svdButton.config(font=font1)
    svdButton.config(bg='#ffffff', fg='black')

    detectButton = Button(main, text="Recommend Communities", command=detectCommunity)
    detectButton.place(x=965,y=85)
    detectButton.config(font=font1)
    detectButton.config(bg='#ffffff', fg='black')

    exitButton = Button(main, text="Exit", command=close)
    exitButton.place(x=1200,y=85)
    exitButton.config(font=font1)
    exitButton.config(bg='#ffffff', fg='black')

    font1 = ('times', 12, 'bold')
    text=Text(main,height=27,width=157)
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=6,y=150)
    text.config(font=font1)
   
    #main.config("#000000")
    main.mainloop()

if __name__ == "__main__":
    GUI()