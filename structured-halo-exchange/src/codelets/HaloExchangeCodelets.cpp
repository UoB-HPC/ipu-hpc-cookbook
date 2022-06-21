#include <poplar/Vertex.hpp>
#include <cstddef>
#include <cstdlib>
#include <print.h>
#include <math.h>
#include <ipudef.h>
#include <stdint.h>
#include <assert.h>
#include <cmath>
#include "HaloExchangeCommon.h"


/**
 * Demonstrates a memory-saving "in-place" scheme for stencil updates that also
 * uses a custom data structure and uses several workers sharing the same data structure
 */

using namespace poplar;

inline auto asTileData(void *ref) -> TileData *const{
    return reinterpret_cast<TileData *const>(ref);
}

inline auto indexInCoreData(const int row, const int col, const int numCols) -> int {
    return (row + 1) * (numCols + 2) + (col + 1);
}

inline auto indexWithHalo(const int row, const int col, const int numColsNoHalo) -> int {
    return (row) * (numColsNoHalo + 1) + col;
}

class Initialise : public Vertex {
public:
    InOut <Vector<char, VectorLayout::ONE_PTR, 8>> data;
    int numRows;
    int numCols;

    bool compute() {
        auto tileData = asTileData(&data[0]);
        tileData->numRows = numRows;
        tileData->numCols = numCols;
        tileData->writeScheme = 0;
        for (int i = 0; i < NumCellsInTileSide + 2; i++) {
            for (int j = 0; j < NumCellsInTileSide + 2; j++) {
               tileData->cells[i * (NumCellsInTileSide + 2)  + j] = 0.f;
            }
        }
        return true;
    }
};

class PackHalo : public Vertex {
public:
    Output <Vector<float, VectorLayout::ONE_PTR, 8>> halo;
    InOut <Vector<char, VectorLayout::ONE_PTR, 8>> data;

    bool compute() {
        auto tileData = asTileData(&data[0]);
        const auto cells = tileData->cells;
        auto out = reinterpret_cast<ToNeighboursHalo *>(&halo[0]);
        auto nc = tileData->numCols;

        if (tileData->writeScheme == 0) { // Data is in expected place
            auto idxTop = indexInCoreData(0, 0, nc);
            auto idxBottom = indexInCoreData(tileData->numRows - 1, 0, nc);
            for (int i = 0; i < NumCellsInTileSide; i++) {
                out->top[i] = cells[idxTop + i];
                out->bottom[i] = cells[idxBottom + i];
                out->left[i] = cells[indexInCoreData(i, 0, nc)];
                out->right[i] = cells[indexInCoreData(i, nc - 1, nc)];
            }
        } else { // Data is shifted by (-1,-1) and wrapped around
            auto idxTop = indexWithHalo(0, 0, nc);
            auto idxBottom = indexWithHalo(tileData->numRows - 1, 0, nc);
            for (int i = 0; i < NumCellsInTileSide; i++) {
                out->top[i] = cells[idxTop + i];
                out->bottom[i] = cells[idxBottom + i];
                out->left[i] = cells[indexWithHalo(i, 0, nc)];
                out->right[i] = cells[indexWithHalo(i, nc - 1, nc)];
            }
        }
        return true;
    }
};


/* This is the 1 worker doing everything version */
class UnpackHaloAll : public Vertex {
public:
    Input <Vector<float, VectorLayout::ONE_PTR, 8>> halo;
    Input <Vector<char, VectorLayout::ONE_PTR, 8>> data; // Naughty! We actually write

    bool compute() {
        auto tileData = asTileData((void*)&data[0]);
        auto cells = tileData->cells;
        const auto in = reinterpret_cast<const FromNeighboursHalo *>(&halo[0]);
        const auto nc = tileData->numCols;
        const auto nr = tileData->numRows;
        if (tileData->writeScheme == 0) { // Data is in expected place

            cells[indexWithHalo(0, 0, nc)] = in->topLeft;
            cells[indexWithHalo(0, nc + 1, nc)] = in->topRight;
            cells[indexWithHalo(nr + 1, 0, nc)] = in->bottomLeft;
            cells[indexWithHalo(nr + 1, nc + 1, nc)] = in->bottomRight;
            for (int i = 0; i < NumCellsInTileSide; i++) {
                cells[indexWithHalo(0, i + 1, nc)] = in->top[i];
                cells[indexWithHalo(0, i + 1, nc)] = in->bottom[i];
                cells[indexWithHalo(i, 0, nc)] = in->left[i];
                cells[indexWithHalo(i, nc + 1, nc)] = in->right[i];
            }
        } else { // Data is shifted (-1,-1) with wraparound
            cells[indexWithHalo(nr + 1, nc + 1, nc)] = in->topLeft;
            cells[indexWithHalo(nr + 1, nc, nc)] = in->topRight;
            cells[indexWithHalo(nr, nc + 1, nc)] = in->bottomLeft;
            cells[indexWithHalo(nr, nc, nc)] = in->bottomRight;
            for (int i = 0; i < NumCellsInTileSide; i++) {
                cells[indexWithHalo(nr + 1, i, nc)] = in->top[i];
                cells[indexWithHalo(nr, i, nc)] = in->bottom[i];
                cells[indexWithHalo(i, nc + 1, nc)] = in->left[i];
                cells[indexWithHalo(i, nc, nc)] = in->right[i];
            }
        }

        return true;
    }
};


/* This is the 1 worker doing everything version */
class UnpackHalo : public Vertex {
public:
    Input <Vector<float, VectorLayout::ONE_PTR, 8>> halo;
    Input <Vector<char, VectorLayout::ONE_PTR, 8>> data; // Naughty! We actually write

    bool compute() {
        auto tileData = asTileData((void*)&data[0]);
        auto cells = tileData->cells;
        const auto in = reinterpret_cast<const FromNeighboursHalo *>(&halo[0]);
        const auto nc = tileData->numCols;
        const auto nr = tileData->numRows;
        if (tileData->writeScheme == 0) { // Data is in expected place

            cells[indexWithHalo(0, 0, nc)] = in->topLeft;
            cells[indexWithHalo(0, nc + 1, nc)] = in->topRight;
            cells[indexWithHalo(nr + 1, 0, nc)] = in->bottomLeft;
            cells[indexWithHalo(nr + 1, nc + 1, nc)] = in->bottomRight;
            for (int i = 0; i < NumCellsInTileSide; i++) {
                //      cells[indexWithHalo(0, i + 1, nc)] = in->top[i];
                cells[indexWithHalo(0, i + 1, nc)] = in->bottom[i];
                cells[indexWithHalo(i, 0, nc)] = in->left[i];
                cells[indexWithHalo(i, nc + 1, nc)] = in->right[i];
            }
        } else { // Data is shifted (-1,-1) with wraparound
            cells[indexWithHalo(nr + 1, nc + 1, nc)] = in->topLeft;
            cells[indexWithHalo(nr + 1, nc, nc)] = in->topRight;
            cells[indexWithHalo(nr, nc + 1, nc)] = in->bottomLeft;
            cells[indexWithHalo(nr, nc, nc)] = in->bottomRight;
            for (int i = 0; i < NumCellsInTileSide; i++) {
                //        cells[indexWithHalo(nr + 1, i, nc)] = in->top[i];
                cells[indexWithHalo(nr, i, nc)] = in->bottom[i];
                cells[indexWithHalo(i, nc + 1, nc)] = in->left[i];
                cells[indexWithHalo(i, nc, nc)] = in->right[i];
            }
        }

        return true;
    }
};

/* We can do separate workers for top, bottom, left, right and corners to spread the load */
class UnpackHaloBottom : public Vertex {
public:
    Input <Vector<float, VectorLayout::ONE_PTR, 8>> halo;
    Input <Vector<char, VectorLayout::ONE_PTR, 8>> data; // Naughty! We actually write to it

    bool compute() {
        auto tileData = asTileData((void*)&data[0]);
        auto cells = tileData->cells;
        const auto in = reinterpret_cast<const FromNeighboursHalo *>(&halo[0]);
        const auto nc = tileData->numCols;
        const auto nr = tileData->numRows;
        if (tileData->writeScheme == 0) { // Data is in expected place
            for (int i = 0; i < NumCellsInTileSide; i++) {
                cells[indexWithHalo(0, i + 1, nc)] = in->bottom[i];
            }
        } else { // Data is shifted (-1,-1) with wraparound
            for (int i = 0; i < NumCellsInTileSide; i++) {
                cells[indexWithHalo(nr, i, nc)] = in->bottom[i];
            }
        }

        return true;
    }
};

class UnpackHaloTop : public Vertex {
public:
    Input <Vector<float, VectorLayout::ONE_PTR, 8>> halo;
    Input <Vector<char, VectorLayout::ONE_PTR, 8>> data; // We mark as input but actually we are going to be
    // naughty and write to it

    bool compute() {
        auto tileData = asTileData((void*)&data[0]);
        auto cells = tileData->cells;
        const auto in = reinterpret_cast<const FromNeighboursHalo *>(&halo[0]);
        const auto nc = tileData->numCols;
        const auto nr = tileData->numRows;
        if (tileData->writeScheme == 0) { // Data is in expected place
            for (int i = 0; i < NumCellsInTileSide; i++) {
                cells[indexWithHalo(0, i + 1, nc)] = in->top[i];
            }
        } else { // Data is shifted (-1,-1) with wraparound
            for (int i = 0; i < NumCellsInTileSide; i++) {
                cells[indexWithHalo(nr + 1, i, nc)] = in->top[i];
            }
        }

        return true;
    }
};


/**
 * Here we also demonstrate splitting between a number of worker threads that share tha same data structure on
 * the tile
 */

class Stencil : public Vertex {
public:
    Input <Vector<char, VectorLayout::ONE_PTR, 8>> data; // Naughty! We write to it even though it's an input
    int threadRowFrom;
    int threadRowTo;

    bool compute() {
        auto tileData = asTileData((void*)&data[0]);
        auto cells = tileData->cells;

        const auto nc = tileData->numCols;
        const auto nr = tileData->numRows;

        if (tileData->writeScheme == 0) {
            for (int r = threadRowFrom; r < threadRowTo; r++) {
                for (int c = 0; c < nc; c++) {
                    cells[indexInCoreData(r - 1, c - 1, nc)] = (cells[indexInCoreData(r - 1, c - 1, nc)]
                                                                + cells[indexInCoreData(r - 1, c, nc)]
                                                                + cells[indexInCoreData(r - 1, c + 1, nc)]
                                                                + cells[indexInCoreData(r, c - 1, nc)]
                                                                + cells[indexInCoreData(r, c, nc)]
                                                                + cells[indexInCoreData(r, c + 1, nc)]
                                                                + cells[indexInCoreData(r + 1, c - 1, nc)]
                                                                + cells[indexInCoreData(r + 1, c, nc)]
                                                                + cells[indexInCoreData(r + 1, c + 1, nc)]) / 9.f;
                }
            }
        } else {
            for (int rr = threadRowFrom; rr < threadRowTo; rr++) {
                for (int cc = nc - 1; cc >= 0; cc--) {

                    auto r = [&](int row = 0) -> int {
                        return (rr + row) % nr;
                    };
                    auto c = [&](int col = 0) -> int {
                        return (cc + col) % nc;
                    };

                    cells[indexInCoreData(rr, cc, nc)] = (cells[indexInCoreData(r(-1), c(-1), nc)]
                                                          + cells[indexInCoreData(r(-1), c(), nc)]
                                                          + cells[indexInCoreData(r(-1), c(1), nc)]
                                                          + cells[indexInCoreData(r(), c(-1), nc)]
                                                          + cells[indexInCoreData(r(), c(), nc)]
                                                          + cells[indexInCoreData(r(), c(1), nc)]
                                                          + cells[indexInCoreData(r(1), c(-1), nc)]
                                                          + cells[indexInCoreData(r(1), c(), nc)]
                                                          + cells[indexInCoreData(r(1), c(1), nc)]) / 9.f;

                }
            }
        }

        tileData->writeScheme = 1 - tileData->writeScheme;
        return true;

    }
};
