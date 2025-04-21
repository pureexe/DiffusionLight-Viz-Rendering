import argparse
import os 

def is_vll_machine():
    valid_hosts = [f"vision{str(i).zfill(2)}" for i in range(1, 24)]
    try:
        import socket
        hostname = socket.gethostname().lower()
        if hostname in valid_hosts:
            return True
        else:
            return False    
    except:
        return False

def create_argparser():    
    BLENDER_PATH = "/home/vll/software/blender-3.2.2-linux-x64/blender" if is_vll_machine() else "blender"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True ,help='output directory') 
    parser.add_argument("--input_dir", type=str, required=True ,help='input directory')
    parser.add_argument("--tasks", type=str, default="mirror,matte_silver,diffuse" ,help='name of the job to do') 
    parser.add_argument("--idx", type=int, default=0 ,help='Current id of the job (start by index 0)')
    parser.add_argument("--total", type=int, default=1 ,help='Total process avalible')
    parser.add_argument("--blender_path", type=str, default=BLENDER_PATH ,help='input directory')
    parser.add_argument("--batch_size", type=int, default=10, help='How much to rnder at the same time')
    parser.add_argument("--mode", type=str, default="front", help='Mode for rendering, this repo will support front mode only, standard mode please see http://github.com/DiffusionLight/DiffusionLight-evaluation')
    return parser

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    args = create_argparser().parse_args()
    files = os.listdir(args.input_dir)
    total_files  = len(files)
    # https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
    files_for_this_thread = (total_files + args.total - 1) // args.total
    real_begin = files_for_this_thread * args.idx
    real_ends = files_for_this_thread * (args.idx+1)
    loop_counter = 0
    mode = args.mode
    job_names = [f.strip() for f in args.tasks.split(",")]
    while (loop_counter * args.batch_size) < (real_ends - real_begin):
        begin = loop_counter * args.batch_size + real_begin
        finish = min(begin + args.batch_size, real_ends)
        for job_name in job_names:
            if job_name == "diffuse" and mode=="front":
                cmd = f'{args.blender_path} --background --python {current_dir}/front_diffuse.py -- 50 5 "{args.input_dir}" "{args.output_dir}" {begin} {finish}'
            elif job_name == "matte_silver" and mode=="front":
                cmd = f'{args.blender_path} --background --python {current_dir}/front_matte_silver.py -- 50 5 "{args.input_dir}" "{args.output_dir}" {begin} {finish}'
            elif job_name == "mirror" and mode=="front":
                cmd = f'{args.blender_path} --background --python {current_dir}/front_mirror.py -- 50 5 {args.input_dir} {args.output_dir} {begin} {finish}'
            else:
                raise NotImplementedError("This mode is not implmented yet, maybe you are looking for standard evaluation at http://github.com/DiffusionLight/DiffusionLight-evaluation")
            print(cmd)
            os.system(cmd)
        loop_counter += 1
           
    
if __name__ == "__main__":
    main()