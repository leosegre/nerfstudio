import os
import sys

def main(reg_pipline):
    scene_names = ["fern", "horns", "room", "trex"]
    # scene_names = ["trex"]
    # timestamps = {"fern-0-100-even-odd": "2024-01-14_125638",
    #               "fern-30-70-even-odd": "2024-01-14_125642",
    #               "fern-50-50": "2024-01-14_125700",
    #               "horns-0-100-even-odd": "2024-01-14_125733",
    #               "horns-30-70-even-odd": "2024-01-14_125658",
    #               "horns-50-50": "2024-01-14_125658",
    #               "room-0-100-even-odd": "2024-01-14_125734",
    #               "room-30-70-even-odd": "2024-01-14_125646",
    #               "room-50-50": "2024-01-14_125649",
    #               "trex-0-100-even-odd": "2024-01-14_125633",
    #               "trex-30-70-even-odd": "2024-01-14_125619",
    #               "trex-50-50": "2024-01-14_125628"}

    exp_types = ["0-100-even-odd", "30-70-even-odd", "50-50"]
    # exp_types = ["30-70-even-odd"]
    # exp_types = ["0-100-even-odd"]

    # scene_names = ["lion", "table"]
    # exp_types = ["30-70-even-odd"]
    timestamps = {"fern-0-100-even-odd": "2024-01-15_180431",
                  "fern-30-70-even-odd": "2024-01-15_180422",
                  "fern-50-50": "2024-01-15_180446",
                  "horns-0-100-even-odd": "2024-01-15_180420",
                  "horns-30-70-even-odd": "2024-01-15_180421",
                  "horns-50-50": "2024-01-15_180419",
                  "room-0-100-even-odd": "2024-01-15_180424",
                  "room-30-70-even-odd": "2024-01-15_180454",
                  "room-50-50": "2024-01-15_180459",
                  "trex-0-100-even-odd": "2024-01-15_183725",
                  "trex-30-70-even-odd": "2024-01-15_183729",
                  "trex-50-50": "2024-01-15_183744"}


    downscale = 2

    for scene_name in scene_names:
        for exp_type in exp_types:
            cmd = f"runai submit --pvc=storage:/storage -i leosegre/nerfstudio_reg --name leo3-{scene_name}-{exp_type}{reg_pipline.replace('_', '-').replace('reg-pipeline', '')} " \
                  f"-g 1 --large-shm --command -- bash entrypoint.sh python {reg_pipline}.py " \
            f"/storage/leo/data /storage/leo/outputs/ {scene_name} {exp_type.replace('-', '_')} {downscale} {timestamps[f'{scene_name}-{exp_type}']}"
            # f"/storage/leo/data /storage/leo/outputs/ {scene_name} {exp_type.replace('-', '_')} {downscale}"



            os.system(cmd)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_runai_jobs.py <reg_pipline>")
    else:
        reg_pipline = sys.argv[1]
        main(reg_pipline)

