import cv2
import numpy as np
from videoanalysis import Utils


def outlineRect(image, rect, color):
    if rect is None:
        return
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)


def copyRect(src, dst, srcRect, dstRect, mask=None, interpolation=cv2.INTER_LINEAR):
    """
    copy part of the source to part of the destination
    """

    x0, y0, w0, h0 = srcRect
    x1, y1, w1, h1 = dstRect

    if mask is None:
        dst[y1:y1+h1, x1:x1+w1] = cv2.resize(src[y0:y0+h0, x0:x0+w0], (w1, h1), interpolation=interpolation)
    else:
        # if not Utils.isGray(src):
            # mask = np.tile(mask, (3, 1)).reshape(3, h0, w0)
        dst[y1:y1 + h1, x1:x1 + w1] = np.where(cv2.resize(mask, (w1, h1), interpolation=interpolation),
                                               cv2.resize(src[y0:y0 + h0, x0:x0 + w0], (w1, h1),
                                                          interpolation=interpolation),
                                               dst[y0:y0 + h0, x0:x0 + w0])


def swapRects(src, dst, rects, mask=None, interpolation=cv2.INTER_LINEAR):

    """Copy the source with two or more sub-rectangles swapped."""
    if dst is not src:
        dst[:] = src

    numRects = len(rects)
    if numRects < 2:
        return

    if mask is None:
        mask = [None] * numRects

    # Copy the contents of the last rectangle into temporary storage.
    x, y, w, h = rects[numRects - 1]
    temp = src[y:y + h, x:x + w].copy()

    # Copy the contents of each rectangle into the next.
    i = numRects - 2
    while i >= 0:
        copyRect(src, dst, rects[i], rects[i + 1], mask[i], interpolation)
        i -= 1
    # Copy the temporarily stored content into the first rectangle.
    copyRect(temp, dst, (0, 0, w, h), rects[0], mask[numRects-1], interpolation)
