from multiProcessingTask import *
from multiprocessing.dummy import Process

if __name__ == '__main__':
    logfacelib()
    Process(target=expression_loop).start()
    Process(target=detection_loop).start()
    Process(target=recognition_loop).start()
    Process(target=videoplay_loop).start()
    camera_loop(cap)



