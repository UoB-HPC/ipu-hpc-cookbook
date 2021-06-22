#ifndef IPUSOMETHING_HALOEXCHANGECOMMON_H
#define IPUSOMETHING_HALOEXCHANGECOMMON_H


const size_t BufferSize = 150 * 1024;
const auto NumCellsInTileSide = 120;
const auto NumNeighbours = 8;
const auto HaloSizeToNeighbours = NumCellsInTileSide * 4;
const auto HaloSizeFromNeighbours = 4 * NumCellsInTileSide + 4;

using Cell = float;

struct TileData {
    Cell cells[(NumCellsInTileSide + 2) * (NumCellsInTileSide * 2)];
    unsigned int numRows;
    unsigned int numCols;
    int writeScheme;
};


enum Directions {
    nw, n, ne, e, se, s, sw, w
};
using Direction = int;


const auto ToTopNeighbourHaloIndex = 0;
const auto ToBottomNeighbourHaloIndex = ToTopNeighbourHaloIndex + NumCellsInTileSide;
const auto ToLeftNeighbourHaloIndex = ToBottomNeighbourHaloIndex + NumCellsInTileSide;
const auto ToRightNeighbourHaloIndex = ToLeftNeighbourHaloIndex + NumCellsInTileSide;
const auto ToTopLeftNeighbourHaloIndex = ToTopNeighbourHaloIndex;
const auto ToTopRightNeighbourHaloIndex = ToTopNeighbourHaloIndex + NumCellsInTileSide - 1;
const auto ToBottomLeftNeighbourHaloIndex = ToBottomNeighbourHaloIndex;
const auto ToBottomRightNeighbourHaloIndex = ToBottomNeighbourHaloIndex - 1;

const auto FromTopNeighbourHaloIndex = 0;
const auto FromBottomNeighbourHaloIndex = FromTopNeighbourHaloIndex + NumCellsInTileSide;
const auto FromLeftNeighbourHaloIndex = FromBottomNeighbourHaloIndex + NumCellsInTileSide;
const auto FromRightNeighbourHaloIndex = FromLeftNeighbourHaloIndex + NumCellsInTileSide;
const auto FromTopLeftNeighbourHaloIndex = FromRightNeighbourHaloIndex + NumCellsInTileSide;
const auto FromTopRightNeighbourHaloIndex = FromTopLeftNeighbourHaloIndex + 1;
const auto FromBottomLeftNeighbourHaloIndex = FromTopRightNeighbourHaloIndex + 1;
const auto FromBottomRightNeighbourHaloIndex = FromBottomLeftNeighbourHaloIndex + 1;


auto oppositeDirection(Direction d) {
    return Directions::w - d;
}

struct ToNeighboursHalo {
    Cell top[NumCellsInTileSide], bottom[NumCellsInTileSide], left[NumCellsInTileSide], right[NumCellsInTileSide];
};

struct FromNeighboursHalo {
    Cell top[NumCellsInTileSide], bottom[NumCellsInTileSide], left[NumCellsInTileSide], right[NumCellsInTileSide];
    Cell topLeft, topRight, bottomLeft, bottomRight;
};


#endif //IPUSOMETHING_HALOEXCHANGECOMMON_H
