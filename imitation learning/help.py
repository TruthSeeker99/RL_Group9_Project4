import atexit

import os
HOME=os.getenv("HOME")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=1)
    args = parser.parse_args()
    for i in range(args.n):
        cmd = f"""docker run --memory=512m  -p 127.0.0.1:{50051+i}:{50051} -d --rm --name test-engine-{i} \
  -v {HOME}/.diambra/credentials:/tmp/.diambra/credentials \
  -v {HOME}/Desktop/sfiii/:/opt/diambraArena/roms \
   docker.io/diambra/engine:v2.2
            """
        print(cmd)
        import shlex
        import subprocess
        subprocess.run(shlex.split(cmd))
    import time
    try:
        while True:
            time.sleep(1)
    except:
        pass
    def kill_all():
         subprocess.Popen( "docker container kill $(docker ps | awk ' { print $1 }')", stdin=subprocess.PIPE, shell=True )
    atexit.register(kill_all)
        
