import argparse
from find_offside_line import find_offside_line

if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    # 1. yolo detect
    # 2. trajectory_convert
    # 3. team classify
    # 4. draw offside line
    # 5. +@ find pass frame

    find_offside_line('your path')