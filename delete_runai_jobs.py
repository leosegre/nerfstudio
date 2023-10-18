import os
import sys
def main(reg_pipline):
    scene_names = ["fern", "fortress", "horns", "room"]
    exp_types = ["0-100-even-odd", "30-70-even-odd", "50-50"]

    for scene_name in scene_names:
        for exp_type in exp_types:
            cmd = f"runai delete job leo2-{scene_name}-{exp_type}{reg_pipline.replace('_', '-').replace('reg-pipeline', '')}"
            # cmd = f"runai delete job leo-{scene_name}-{exp_type}-{reg_pipline.replace('_', '-')}"
            os.system(cmd)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_runai_jobs.py <reg_pipline>")
    else:
        reg_pipline = sys.argv[1]
        main(reg_pipline)
