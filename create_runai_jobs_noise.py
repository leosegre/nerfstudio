import os
import sys

def main(reg_pipline):
    scene_names = ["trex"]
    # scene_names = ["trex"]
    timestamps = {"fern-0-100-even-odd": "2023-12-18_110607",
                  "fern-30-70-even-odd": "2023-12-18_110622",
                  "fern-50-50": "2023-12-18_110747",
                  "horns-0-100-even-odd": "2023-12-18_110806",
                  "horns-30-70-even-odd": "2023-12-18_110614",
                  "horns-50-50": "2023-12-18_110614",
                  "room-0-100-even-odd": "2023-12-18_110618",
                  "room-30-70-even-odd": "2023-12-18_110603",
                  "room-50-50": "2023-12-18_110607",
                  "trex-0-100-even-odd": "2023-12-18_110617",
                  "trex-30-70-even-odd": "2023-12-18_110612",
                  "trex-50-50": "2023-12-18_110740",
                  "lion-30-70-even-odd": "2024-02-05_084037",
                  "table-30-70-even-odd": "2024-02-05_084039"}

    exp_types = ["0-100-even-odd", "30-70-even-odd", "50-50"]
    # exp_types = ["30-70-even-odd"]
    # exp_types = ["0-100-even-odd"]

    noise_levels = ["0.01", "0.05", "0.1", "0.2"]

    # scene_names = ["lion", "table"]
    # exp_types = ["30-70-even-odd"]
    # timestamps = {"fern-0-100-even-odd": "2024-01-15_180431",
    #               "fern-30-70-even-odd": "2024-01-15_180422",
    #               "fern-50-50": "2024-01-15_180446",
    #               "horns-0-100-even-odd": "2024-01-15_180420",
    #               "horns-30-70-even-odd": "2024-01-15_180421",
    #               "horns-50-50": "2024-01-15_180419",
    #               "room-0-100-even-odd": "2024-01-15_180424",
    #               "room-30-70-even-odd": "2024-01-15_180454",
    #               "room-50-50": "2024-01-15_180459",
    #               "trex-0-100-even-odd": "2024-01-15_183725",
    #               "trex-30-70-even-odd": "2024-01-15_183729",
    #               "trex-50-50": "2024-01-15_183744"}


    downscale = 2

    for scene_name in scene_names:
        for exp_type in exp_types:
            for noise_level in noise_levels:
                cmd = f"runai submit --pvc=storage:/storage -i leosegre/nerfstudio_reg --name leo3-{scene_name}-{exp_type}-{noise_level.replace('.', '-')}{reg_pipline.replace('_', '-').replace('reg-pipeline', '')} " \
                      f"-g 1 --large-shm --command -- bash entrypoint.sh python {reg_pipline}.py " \
                f"/storage/leo/data /storage/leo/outputs/ {scene_name} {exp_type.replace('-', '_')} {noise_level} {downscale}"
                # print(cmd)


                os.system(cmd)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_runai_jobs.py <reg_pipline>")
    else:
        reg_pipline = sys.argv[1]
        main(reg_pipline)

