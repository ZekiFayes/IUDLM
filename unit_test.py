import cv2
import numpy as np
import matplotlib.pyplot as plt


def diff(src, x1, y1, x2, y2):
    dist = np.sqrt(np.sum(np.square(src[x1, y1] - src[x2, y2])))
    return dist


img = cv2.imread('Data/sample.jpg')
im = cv2.GaussianBlur(img, (5, 5), 0.8)
dim = im.shape
width = dim[1]
height = dim[0]
edges = []
num = 0

for y in range(height):
    for x in range(width):

        if x < width - 1:
            a = y * width + x
            b = y * width + (x+1)
            w = diff(im, x, y, x+1, y)
            edges.append([w, a, b])
            num += 1

        if y < height - 1:
            a = y * width + x
            b = (y+1) * width + x
            w = diff(im, x, y, x, y+1)
            edges.append([w, a, b])
            num += 1

        if x < width - 1 and y < height - 1:
            a = y * width + x
            b = (y+1) * width + (x+1)
            w = diff(im, x, y, x+1, y+1)
            edges.append([w, a, b])
            num += 1

        if x < width - 1 and y > 0:
            a = y * width + x
            b = (y-1) * width + (x+1)
            w = diff(im, x, y, x+1, y-1)
            edges.append([w, a, b])
            num += 1

sorted_edges = sorted(edges)

num_vertices = width*height
thresh = []
rank = []
size = []
p = []

for i in range(num_vertices):
    th = 0.8
    rank.append(0)
    size.append(1)
    p.append(i)
    thresh.append(th)


def find(x_p):
    y_p = x_p
    while y_p != p[y_p]:
        y_p = p[y_p]
    p[x_p] = y_p
    return y_p


def join(vx, vy):
    if rank[vx] > rank[vy]:
        p[vy] = vx
        size[vx] += size[vy]
    else:
        p[vx] = vy
        size[vy] += size[vx]
        if rank[vx] == rank[vy]:
            rank[vy] += 1


def threshold(s, c):
    return c/s


for i in range(num):
    a = find(sorted_edges[i][1])
    b = find(sorted_edges[i][2])

    if a != b:
        if sorted_edges[i][0] <= thresh[a] and sorted_edges[i][0] <= thresh[b]:
            join(a, b)
            a = find(a)
            thresh[a] = sorted_edges[i][0] + threshold(size[a] + 0.8)