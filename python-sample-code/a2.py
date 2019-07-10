import sys


if __name__ == "__main__":
    i = sys.argv.__len__()
    part = sys.argv[1]

    if part == 'part1':
        # compare_n_cluster(part, k, in_file, out_file)
        k = sys.argv[2]
        in_file = sys.argv[3:i - 1]
        out_file = sys.argv[i - 1]

    if part == 'part2':
        ver = 0
    if part == 'part3':
        var=0



