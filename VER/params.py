import os


class Params(object):
    def __init__(self) -> None:
        self.root = os.getcwd()
        root = self.root
        self.img_path = os.path.join(root, r"Repo/data/img")
        self.audio_path = os.path.join(root, r"Repo/data/audio")
        self.mp4_path = os.path.join(root, r"Repo/video/mp4")
        self.sub_mp4_path = os.path.join(root, r"Repo/video/sub_mp4")
        self.img_model_path = os.path.join(root, r"imageR/model")
        self.audio_model_path = os.path.join(root, r"audioR/model")
        self.img_results_path = os.path.join(root, r"Repo/results/img")
        self.gif_results_path = os.path.join(root, r"Repo/results/gif")


if __name__ == "__main__":
    p = Params()
    for item in p.__dict__:
        print(item, ":", p.__dict__[item])