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

def create_3d_scatter(ax, x, y, z, colors, title):
    ax.scatter(x, y, z, c=colors, alpha=1, s=.7)
    ax.set_title(title, fontdict = {'fontsize': 16})
    return ax

def create_3d_scatter_frames(ax, min_angle, max_angle, num_frames, out_directory):
    angles = np.linspace(min_angle, max_angle, num_frames) 
    plt.tight_layout()
    for i, angle in enumerate(angles):
        ax.view_init(30, angle)
        plt.savefig(f"{anim_dir}{str(i).zfill(len(str(num_frames)))}.png")
    return

def create_gif(folder_of_pngs, file_path_out, sec_per_frame=0.5, remove_source_files=False):
    """
    Create gif from a folder of PNG files
    Keyword arguments:
    folder_of_pngs -- path to folder of png images with files named in ascending order (ex. 001.png, 002.png, etc.)
    file_path_out -- path to output the resulting GIF
    frame_rate -- number of seconds to display each frame. Defaults to 0.5
    remove_source_files -- if True, deletes the frames used to create the GIF
    """
    images = []
    img_paths = sorted(glob.glob(os.path.join(folder_of_pngs, "*.png")))
    for img_path in img_paths:
        images.append(imageio.imread(img_path))
    imageio.mimsave(file_path_out, images, duration=sec_per_frame, subrectangles=True)
    if remove_source_files:
        for img_path in img_paths:
            os.remove(img_path)
    return

if __name__ == '__main__':
    plt.style.use("seaborn")

    more_sw = [
        "im",
        "fly",
        "wa",
        "airline",
        "fleek",
        "got"
    ]

    sw = create_stop_words(more_sw)

    data = pd.read_csv("data/Tweets.csv")
    X_raw = data['text']

    count_vect = CountVectorizer(
                            tokenizer=None,
                            stop_words=sw,
                            analyzer='word',
                            min_df=100, # words must appear 100 times to be considered
                            max_features=None
    )

    tfidf_transformer = TfidfTransformer(use_idf=True)

    X_vec = count_vect.fit_transform(X_raw)
    X_tfidf = tfidf_transformer.fit_transform(X_vec)

    pca = PCA(n_components=3) 
    X_pca = pca.fit_transform(X_tfidf.toarray())

    # define scatter plot colors
    color_dict = {
        "positive": "green",
        "negative": "red",
        "neutral": "black"
    }

    colors = [color_dict[y] for y in data['airline_sentiment']]

    fig = plt.figure(figsize=(10,6), dpi=80)
    ax = plt.axes(projection='3d')

    create_3d_scatter(
                    ax = ax,
                    x = X_pca[:, 0],
                    y = X_pca[:, 1],
                    z = X_pca[:, 2],
                    colors=colors,
                    title="Airline Sentiment with 3 Principal Components"
    )

    # Create animation directory if doesn't exist
    anim_dir = "images/pca_anim/"
    if not os.path.exists(anim_dir):
        os.mkdir(anim_dir)

    create_3d_scatter_frames(
                        ax = ax,
                        min_angle = 0,
                        max_angle = 360,
                        num_frames = 200,
                        out_directory = anim_dir
    )

    create_gif(anim_dir, "images/pca_animation.gif", sec_per_frame=0.05, remove_source_files=True)