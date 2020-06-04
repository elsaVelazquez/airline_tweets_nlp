import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from helpers import create_stop_words
import os
import imageio
import glob

from d2v_custom import CustomDoc2Vec


def create_3d_scatter(ax, x, y, z, colors, title):
    '''
    Create 3D scatter okit with given X, Y, and Z arrays
    '''
    ax.scatter(x, y, z, c=colors, alpha=1, s=1)
    ax.set_title(title, fontdict={'fontsize': 16})
    return ax


def create_3d_scatter_frames(
                                ax, min_angle, max_angle,
                                num_frames, out_directory
                            ):
    '''
    Create frames of 3D scatter plot for use in an animation
    Will create num_frames, rotating between min_angle and max_angle
    '''
    angles = np.linspace(min_angle, max_angle, num_frames)
    plt.tight_layout()
    for i, angle in enumerate(angles):
        ax.view_init(30, angle)
        plt.savefig(f"{out_directory}{str(i).zfill(len(str(num_frames)))}.png")
    return


def create_gif(
            folder_of_pngs, file_path_out,
            sec_per_frame=0.5, remove_source_files=False
        ):
    """
    Create gif from a folder of PNG files

    folder_of_pngs : path to folder of png files with frames in ascending order
        (ex. 001.png, 002.png, etc.)
    file_path_out : path to output the resulting GIF
    frame_rate : number of seconds to display each frame. Defaults to 0.5
    remove_source_files : if True, deletes the frames used to create the GIF
    """
    images = []
    img_paths = sorted(glob.glob(os.path.join(folder_of_pngs, "*.png")))
    for img_path in img_paths:
        images.append(imageio.imread(img_path))
    imageio.mimsave(
                file_path_out,
                images,
                duration=sec_per_frame,
                subrectangles=True
            )
    if remove_source_files:
        for img_path in img_paths:
            os.remove(img_path)
    return


def create_pca_animation(
                    data, targets,
                    title, color_dict,
                    frame_path, outfilepath
                ):
    '''
    Create animation of rotating 3D scatter plot of PCA
    down to three principal components for given data

    Intended for NLP exploration

    data : X feature matrix
    targets : List of targets
    color_dict : Dictionary with targets as keys and desired colors as values
    frame_path : Path to create 'pca_anim' directory to store frames to animate
    outfilepath : Path of output GIF animation
    '''

    plt.style.use("seaborn")

    # Run PCA w/ 3 components so data can be visualized in 3d space
    pca = PCA(n_components=3)

    if type(data) != np.ndarray:
        data = data.toarray()

    X_pca = pca.fit_transform(data)

    # define colors for scatter plot
    colors = [color_dict[y] for y in targets]

    fig = plt.figure(figsize=(10, 6), dpi=80)
    ax = plt.axes(projection='3d')

    # create scatter plot
    create_3d_scatter(
                    ax=ax,
                    x=X_pca[:, 0],
                    y=X_pca[:, 1],
                    z=X_pca[:, 2],
                    colors=colors,
                    title=title
                )

    # Create animation directory if doesn't exist to temporarily store frames
    anim_dir = f"{frame_path}/pca_anim/"
    if not os.path.exists(anim_dir):
        os.mkdir(anim_dir)

    # create frames
    create_3d_scatter_frames(
                        ax=ax,
                        min_angle=0,
                        max_angle=360,
                        num_frames=200,
                        out_directory=anim_dir
                    )
    # combine frames to gif
    create_gif(
            anim_dir,
            outfilepath,
            sec_per_frame=0.05,
            remove_source_files=True
        )
    return

def tfidf_vectorize(text, stop_words):
    # Transform using CountVectorizer and TF-IDF matrix
    
    # combine stop words with SKLearn default stop words
    sw = create_stop_words(stop_words)
    
    count_vect = CountVectorizer(
                            tokenizer=None,
                            stop_words=sw,
                            analyzer='word',
                            min_df=100,  # words must appear 100 times
                            max_features=None
                        )

    tfidf_transformer = TfidfTransformer(use_idf=True)

    X_vec = count_vect.fit_transform(text)
    X_tfidf = tfidf_transformer.fit_transform(X_vec)
    return X_tfidf

if __name__ == '__main__':

    more_sw = [
        "im",
        "fly",
        "wa",
        "airline",
        "fleek",
        "got"
    ]

    color_dict = {
        "positive": "seagreen",
        "negative": "firebrick",
        "neutral": "darkgoldenrod"
    }

    # read data
    csv_path = "data/Clean_T_Tweets.csv"
    data = pd.read_csv(csv_path)
    X_raw = data['text']

    # vectorize
    X_tfidf = tfidf_vectorize(X_raw, stop_words=more_sw)

    create_pca_animation(
            data=X_tfidf,
            targets=data['airline_sentiment'],
            title="Tweet Sentiment With 3 Principal Components",
            color_dict=color_dict,
            frame_path="images",
            outfilepath="images/pca_animation.gif"
        )

    # read data
    csv_path = "data/Clean_T_Tweets_wo_Users.csv"
    data = pd.read_csv(csv_path)
    X_raw = data['text']

    # vectorize
    X_tfidf = tfidf_vectorize(X_raw, stop_words=more_sw)

    create_pca_animation(
            data=X_tfidf,
            targets=data['airline_sentiment'],
            title="Tweet Sentiment With 3 Principal Components (Std. Users, TF-IDF)",
            color_dict=color_dict,
            frame_path="images",
            outfilepath="images/pca_animation_no_users.gif"
        )

    d2v = CustomDoc2Vec()
    d2v_transformed = d2v.fit_transform(X_raw, data['airline_sentiment'])

    create_pca_animation(
        data=np.array(d2v_transformed),
        targets=data['airline_sentiment'],
        title="Tweet Sentiment With 3 Principal Components (Std. Users, Doc2Vec)",
        color_dict=color_dict,
        frame_path="images",
        outfilepath="images/pca_animation_no_users_d2v.gif"
    )