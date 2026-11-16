import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src import config
from src.evaluation import invite

BACKGROUND = config.BACKGROUND_COLUMNS_FOR_CLUSTERING


def fit_clusters(train_students, n_clusters=config.NUMBER_OF_CLUSTERS):
    # Scaling stops admission grade (0 to 200) from outweighing yes/no flags in the distances.
    clusters = make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=n_clusters, n_init=20, random_state=config.SEED),
    )
    return clusters.fit(train_students[BACKGROUND])


def assign_clusters(clusters, students):
    return clusters.predict(students[BACKGROUND])


def describe_clusters(clusters, train_students):
    """Each cluster's centre in original units, with its size and dropout rate in the training classes."""
    scaler, kmeans = clusters[0], clusters[-1]
    centres = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=BACKGROUND)
    labels = assign_clusters(clusters, train_students)
    centres["training_students"] = np.bincount(labels, minlength=len(centres))
    centres["training_dropout_rate"] = train_students.groupby(labels)["dropped_out"].mean()
    return centres.rename_axis("cluster")


def invite_within_clusters(risk, entering_year, cluster, share=config.SHARE_OF_CLASS_INVITED,
                           seed=config.SEED):
    """Alternative policy: invite the top share of each cluster within each class."""
    # Treating every class-and-cluster pair as its own group gives each cluster
    # invitations in proportion to its size.
    group = pd.Series(np.asarray(entering_year)).astype(str) + "-" + pd.Series(np.asarray(cluster)).astype(str)
    return invite(risk, group, share=share, seed=seed)

def silhouette_by_k(train_students, ks=range(2, 9)):
    """How well separated the clusters are for each k (from -1 to 1, higher is better)."""
    rows = []
    for k in ks:
        clusters = fit_clusters(train_students, k)
        scaled = clusters[0].transform(train_students[BACKGROUND])
        rows.append({"k": k, "inertia": clusters[-1].inertia_,
                     "silhouette": silhouette_score(scaled, clusters[-1].labels_, random_state=config.SEED)})
    return pd.DataFrame(rows)


def first_semester_components(train_students):
    """Share of variance and loadings of each principal component of the first-semester columns."""
    scaled = StandardScaler().fit_transform(train_students[config.FIRST_SEMESTER_COLUMNS])
    pca = PCA().fit(scaled)
    names = [f"PC{i + 1}" for i in range(pca.n_components_)]
    loadings = pd.DataFrame(pca.components_.T, index=config.FIRST_SEMESTER_COLUMNS, columns=names)
    variance = pd.Series(pca.explained_variance_ratio_, index=names, name="share_of_variance")
    return variance, loadings