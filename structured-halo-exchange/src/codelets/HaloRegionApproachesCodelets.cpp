#include <poplar/Vertex.hpp>
#include <cstddef>
#include <print.h>
#include <math.h>
#include <ipudef.h>


using namespace poplar;

template<typename T>
T stencil(const T nw, const T n, const T ne, const T w, const T m,
          const T e, const T sw,
          const T s, const T se) {
    return (nw + n + ne + w + m + e + sw + s + se) / 9;
}


template<typename T>
class Fill : public Vertex {

public:
    Output <Vector<T>> result;
    Input <T> val;

    bool compute() {
        for (auto i = 0; i < result.size(); i++) result[i] = *val;
        return true;
    }
};

template
class Fill<float>;

template<typename T>
class IncludedHalosApproach : public Vertex {

public:
    Input <VectorList<T, poplar::VectorListLayout::COMPACT_DELTAN, 4, false>> in;
    Output <VectorList<T, poplar::VectorListLayout::COMPACT_DELTAN, 4, false>> out;

    // Average the moore neighbourhood of the non-ghost part of the block
    bool compute() {
        // Only works if this is at least a 3x3 block, and in must be same size as out
        if (out.size() == in.size() && in.size() > 2 && in[0].size() > 2 && in[0].size() == out[0].size()) {
            for (auto y = 1u; y < in.size() - 1; y++) {
                for (auto x = 1u; x < in[y].size() - 1; x++) {
                    out[y][x] = stencil(in[y - 1][x - 1], in[y - 1][x], in[y - 1][x + 1],
                                        in[y][x - 1], in[y][x], in[y][x + 1],
                                        in[y + 1][x - 1], in[y + 1][x], in[y + 1][x + 1]);
                }
            }
            return true;
        }
        return false;
    }
};

template
class IncludedHalosApproach<float>;


template<typename T>
class ExtraHalosApproach : public Vertex {

public:
    Input <VectorList<T, poplar::VectorListLayout::COMPACT_DELTAN, 4, false>> in;
    Input <Vector<T>> n, s, w, e;
    Output <VectorList<T, poplar::VectorListLayout::COMPACT_DELTAN, 4, false>> out;

    // Average the moore neighbourhood of the non-ghost part of the block
    bool compute() {
        // Only works if this is at least a 3x3 block (excluding halos), and in must be same size as out
        if (out.size() == in.size() && in.size() > 2 && in[0].size() > 2 && in[0].size() == out[0].size()) {
            const auto nx = in[0].size();
            const auto ny = in.size();

            // top left
            {
                constexpr auto x = 0u;
                constexpr auto y = 0u;
                out[y][x] = stencil(n[x], n[x + 1], n[x + 2],
                                    w[y], in[y][x], in[y][x + 1],
                                    w[y + 1], in[y + 1][x], in[y + 1][x + 1]);
            }

            // top
            {
                constexpr auto y = 0u;
                for (auto x = 1u; x < in[0].size() - 1; x++) {
                    out[y][x] = stencil(n[x], n[x + 1], n[x + 2],
                                        in[y][x - 1], in[y][x], in[y][x + 1],
                                        in[y + 1][x - 1], in[y + 1][x], in[y + 1][x + 1]);
                }
            }

            // top right
            {
                const auto x = nx - 1u;
                constexpr auto y = 0u;
                out[y][x] =
                        stencil(n[x], n[x + 1], n[x + 2],
                                in[y][x - 1], in[y][x], e[y],
                                in[y + 1][x - 1], in[y + 1][x], e[y + 1]);
            }


            // left col
            {
                constexpr auto x = 0u;
                for (auto y = 1; y < ny - 1; y++) {
                    out[y][x] = stencil(w[y - 1], in[y - 1][x], in[y - 1][x + 1],
                                        w[y], in[y][x], in[y][x + 1],
                                        w[y + 1], in[y + 1][x], in[y + 1][x + 1]);
                }
            }

            // middle block
            for (auto y = 1; y < ny - 1; y++) {
                for (auto x = 1; x < nx - 1; x++) {
                    out[y][x] = stencil(in[y - 1][x - 1], in[y - 1][x], in[y - 1][x + 1],
                                        in[y][x - 1], in[y][x], in[y][x + 1],
                                        in[y + 1][x - 1], in[y + 1][x], in[y + 1][x + 1]);
                }
            }

            // right col
            {
                const auto x = nx - 1u;
                for (auto y = 1; y < ny - 1u; y++) {
                    out[y][x] = stencil(in[y - 1][x - 1], in[y - 1][x], e[y - 1],
                                        in[y][x - 1], in[y][x], e[y],
                                        in[y + 1][x - 1], in[y + 1][x], e[y + 1]);
                }
            }

            // bottom left
            {
                const auto y = ny - 1;
                constexpr auto x = 0u;

                out[y][x] = stencil(w[y - 1], in[y - 1][x], in[y - 1][x + 1],
                                    w[y], in[y][x], in[y][x + 1],
                                    s[x], s[x + 1], s[x + 2]);
            }

            // bottom
            {
                const auto y = ny - 1;
                for (auto x = 1u; x < nx - 1u; x++) {
                    out[y][x] = stencil(in[y - 1][x - 1], in[y - 1][x], in[y - 1][x + 1],
                                        in[y][x - 1], in[y][x], in[y][x + 1],
                                        s[x], s[x + 1], s[x + 2]);
                }
            }

            // bottom right
            {
                const auto y = ny - 1;
                const auto x = nx - 1;

                out[y][x] = stencil(in[y - 1][x - 1], in[y - 1][x], e[y - 1],
                                    in[y][x - 1], in[y][x], e[y],
                                    s[x], s[x + 1], s[x + 2]);
            }
            return true;
        }
        return false;
    }
};

template
class ExtraHalosApproach<float>;












