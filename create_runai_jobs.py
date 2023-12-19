import os
import sys

def main(reg_pipline):
    scene_names = ["fern", "horns", "room", "trex"]
    timestamps = {"fern": "2023-12-18_110607",
                  "horns": "2023-12-18_110806",
                  "room": "2023-12-18_110607",
                  "trex": "2023-12-18_110612"}
    exp_types = ["0-100-even-odd", "30-70-even-odd", "50-50"]
    downscale = 2

    for scene_name in scene_names:
        for exp_type in exp_types:
            cmd = f"runai submit --pvc=storage:/storage -i leosegre/nerfstudio_reg --name leo3-{scene_name}-{exp_type}{reg_pipline.replace('_', '-').replace('reg-pipeline', '')} " \
                  f"-g 1 --large-shm --command -- bash entrypoint.sh python {reg_pipline}.py " \
            f"/storage/leo/data /storage/leo/outputs/ {scene_name} {exp_type.replace('-', '_')} {downscale} {timestamps[scene_name]}"
            # f"/storage/leo/data /storage/leo/outputs/ {scene_name} {exp_type.replace('-', '_')} {downscale}"

            os.system(cmd)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_runai_jobs.py <reg_pipline>")
    else:
        reg_pipline = sys.argv[1]
        main(reg_pipline)

