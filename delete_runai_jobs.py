import os

scene_names = ["fern", "fortress", "horns", "room"]
exp_types = ["0-100-even-odd", "30-70-even-odd", "50-50"]

for scene_name in scene_names:
    for exp_type in exp_types:
        os.system("runai delete job leo-{}".format(scene_name+"-"+exp_type))
