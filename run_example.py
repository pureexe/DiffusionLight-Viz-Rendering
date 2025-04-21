import os 
ENVMAP_DIR = 'example/envmap'
BALL_DIR = 'output/stylelight_ball'
CROPPED_DIR = 'output/cropped'

def main():
    # render 3 kinds of ball
    os.system(f"python renderer/job_distributor.py --input_dir {ENVMAP_DIR} --output_dir {BALL_DIR}")
    # tonemap
    os.system(f"python crop2viz.py --input_dir {BALL_DIR} --output_dir {CROPPED_DIR}")
    
if __name__ == "__main__":
    main()