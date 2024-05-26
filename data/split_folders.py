#%%
import os
import shutil
import random
import splitfolders

SOURCE_FOLDER = r"C:\Users\vivek\Desktop\GD"
DEST_FOLDER = r"C:\Users\vivek\Desktop"

splitfolders.ratio(SOURCE_FOLDER, output=DEST_FOLDER, seed=1337, ratio=(0.7, 0.2, 0.1))

# %%
