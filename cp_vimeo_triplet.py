import os
import shutil


file = open("vimeo_test_triplet.txt", 'r')
count = 0

while True:
    line = file.readline()
    if not line:
        break

    line = line.strip()

    name="/home/kim/ssd/vimeo_triplet/sequences/" + line

    if not os.path.exists(f"data/{count}"):
        os.makedirs(f"data/{count}")

    shutil.copyfile(f"{name}/im1.png", f"data_vimeo/{count}/im1.png")
    shutil.copyfile(f"{name}/im2.png", f"data_vimeo/{count}/im2.png")
    shutil.copyfile(f"{name}/im3.png", f"data_vimeo/{count}/im3.png")

    # if count == 2:
    #     break

    count += 1