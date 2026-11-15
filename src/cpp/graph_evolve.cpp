// ============================================================================
//  graph_evolve.cpp
//  «Графовое ядро» (CPU-native AI): умножение на разреженном графе операций.
//
//  Идея: интеллект — это структура вычислений, а не количество весов.
//  Узлы графа — элементарные целочисленные операции, рёбра — поток данных.
//  Такой граф целиком помещается в L1/L2 и не требует ни GPU, ни матричных
//  умножений: только ветвления, сложение, умножение и извлечение цифр.
//
//  Задача: 4-значное × 4-значное = 8-значное (например 1234 * 5678).
//  Решение: честное умножение в столбик с ПОЛНЫМ каскадом переносов
//  (decimal ripple-carry adder).
//
//  Требования: C++17, только целочисленная арифметика (int32/int64),
//  никаких сторонних библиотек, никаких float в вычислительном ядре.
//
//  Сборка (MSVC):  cl /std:c++17 /O2 /utf-8 graph_evolve.cpp
//  Сборка (GCC):   g++ -std=c++17 -O2 -march=native graph_evolve.cpp -o graph_evolve
// ============================================================================

#if defined(_WIN32)
#  define NOMINMAX
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <utility>
#include <vector>

// ============================================================================
//  1. Операции
// ============================================================================

// Минимальный и полный набор примитивов. Всё остальное (умножение, деление,
// извлечение цифры, перенос) собирается из них.
enum class Op : uint8_t {
    Input = 0,   // входной узел: значение подаётся извне
    Add,         // in[0] + in[1]
    Mul,         // in[0] * in[1]
    Mod10,       // in[0] % 10   — младшая десятичная цифра
    Div10,       // in[0] / 10   — перенос в следующую колонку
    Copy,        // in[0]
    Const,       // константа: значение лежит в in[0] (arity = 0)
    // --- логический базис ---------------------------------------------------
    // Для графов, структура которых найдена дифференцируемым обучением
    // (см. src/python/diffgraph). Значения — биты 0/1, поэтому побитовые
    // операции совпадают с булевыми.
    And,         // in[0] & in[1]
    Or,          // in[0] | in[1]
    Xor,         // in[0] ^ in[1]
    Nand,        // 1 - (in[0] & in[1])   — NOT AND для битов 0/1
    OpCount
};

inline const char* opName(Op op) noexcept {
    switch (op) {
        case Op::Input: return "IN";
        case Op::Add:   return "ADD";
        case Op::Mul:   return "MUL";
        case Op::Mod10: return "MOD10";
        case Op::Div10: return "DIV10";
        case Op::Copy:  return "COPY";
        case Op::Const: return "CONST";
        case Op::And:   return "AND";
        case Op::Or:    return "OR";
        case Op::Xor:   return "XOR";
        case Op::Nand:  return "NAND";
        default:        return "?";
    }
}

// Сколько входных слотов использует операция.
inline uint8_t arityOf(Op op) noexcept {
    switch (op) {
        case Op::Add:
        case Op::Mul:
        case Op::And:
        case Op::Or:
        case Op::Xor:
        case Op::Nand:  return 2;
        case Op::Mod10:
        case Op::Div10:
        case Op::Copy:  return 1;
        default:        return 0;   // Input, Const
    }
}

// Обратное отображение: имя -> операция. Нужно загрузчику JSON-графов.
inline bool opFromName(const std::string& name, Op& out) noexcept {
    static const struct { const char* name; Op op; } kTable[] = {
        {"INPUT", Op::Input}, {"IN", Op::Input}, {"CONST", Op::Const},
        {"ADD", Op::Add},     {"MUL", Op::Mul},
        {"MOD10", Op::Mod10}, {"DIV10", Op::Div10}, {"COPY", Op::Copy},
        {"AND", Op::And},     {"OR", Op::Or},
        {"XOR", Op::Xor},     {"NAND", Op::Nand},
    };
    for (const auto& e : kTable) {
        if (name == e.name) { out = e.op; return true; }
    }
    return false;
}

// Терминальные операции ничего не читают: это листья графа.
inline bool isLeafOp(Op op) noexcept {
    return op == Op::Input || op == Op::Const;
}

// ============================================================================
//  2. Узел и граф
// ============================================================================

// Узел: тип операции + массив индексов входов.
// Ровно 12 байт, trivially copyable — граф можно копировать, мутировать и
// скрещивать целыми блоками памяти без единой аллокации на узел.
struct Node {
    Op      op    = Op::Const;   // тип операции
    uint8_t arity = 0;           // сколько слотов in[] реально используется
    int32_t in[2] = {0, 0};      // индексы входных узлов (для Const: in[0] = значение)
};

static_assert(sizeof(Node) <= 16, "Node должен оставаться компактным для кэша");

// Граф: массив узлов + массив входов + массив выходов.
//
// ИНВАРИАНТ ИСПОЛНЕНИЯ: все рёбра указывают строго назад (in[k] < i).
// Это топологический порядок, он даёт линейный проход по узлам без рекурсии,
// без проверок и без ветвлений на поиск «а посчитан ли уже вход».
// Инвариант поддерживается методами repair() / mutate() / crossover().
class Graph {
public:
    std::vector<Node>    nodes;    // узлы в порядке исполнения
    std::vector<int32_t> inputs;   // индексы узлов Op::Input, в порядке подачи значений
    std::vector<int32_t> outputs;  // индексы узлов, чьи значения образуют ответ

    // ---------- сборка ----------
    int32_t addNode(Op op, int32_t a = 0, int32_t b = 0) {
        Node n;
        n.op    = op;
        n.arity = arityOf(op);
        n.in[0] = a;
        n.in[1] = b;
        nodes.push_back(n);
        return static_cast<int32_t>(nodes.size()) - 1;
    }

    int32_t addInput() {
        const int32_t idx = addNode(Op::Input);
        inputs.push_back(idx);
        return idx;
    }

    int32_t addConst(int32_t value) {
        return addNode(Op::Const, value, 0);
    }

    // ---------- метрики ----------
    size_t nodeCount() const noexcept { return nodes.size(); }

    size_t edgeCount() const noexcept {
        size_t e = 0;
        for (const Node& n : nodes) e += n.arity;
        return e;
    }

    // «Параметры» графа в терминах Python-версии: узлы + рёбра.
    size_t paramCount() const noexcept { return nodeCount() + edgeCount(); }

    bool empty() const noexcept { return nodes.empty(); }

    void clear() {
        nodes.clear();
        inputs.clear();
        outputs.clear();
    }

    // ---------- валидация ----------
    bool isValidDag() const noexcept {
        const int32_t n = static_cast<int32_t>(nodes.size());
        for (int32_t i = 0; i < n; ++i) {
            const Node& node = nodes[static_cast<size_t>(i)];
            if (node.arity != arityOf(node.op)) return false;
            for (uint8_t k = 0; k < node.arity; ++k) {
                if (node.in[k] < 0 || node.in[k] >= i) return false;
            }
        }
        for (const int32_t o : outputs) if (o < 0 || o >= n) return false;
        for (const int32_t in : inputs)  if (in < 0 || in >= n) return false;
        return true;
    }

    // Гарантирует, что узел 0 — это Const 0, сдвигая все индексы на единицу.
    void ensureZeroAtFront() {
        if (!nodes.empty() && nodes[0].op == Op::Const && nodes[0].in[0] == 0) return;

        for (Node& n : nodes) {
            for (uint8_t k = 0; k < n.arity; ++k) {
                if (n.in[k] >= 0) n.in[k] += 1;
            }
        }
        for (int32_t& x : inputs)  if (x >= 0) x += 1;
        for (int32_t& x : outputs) if (x >= 0) x += 1;

        Node zero;
        zero.op    = Op::Const;
        zero.arity = 0;
        zero.in[0] = 0;
        zero.in[1] = 0;
        nodes.insert(nodes.begin(), zero);
    }

    // Починка после мутаций/скрещивания: восстанавливает инвариант «рёбра
    // смотрят строго назад». Битые рёбра подключаются к константе 0, а не
    // отбрасываются — так структура графа сохраняется для дальнейшей эволюции.
    void repair() {
        for (Node& n : nodes) n.arity = arityOf(n.op);

        bool needZero = false;
        const int32_t n = static_cast<int32_t>(nodes.size());
        for (int32_t i = 0; i < n && !needZero; ++i) {
            for (uint8_t k = 0; k < nodes[static_cast<size_t>(i)].arity; ++k) {
                const int32_t s = nodes[static_cast<size_t>(i)].in[k];
                if (s < 0 || s >= i) { needZero = true; break; }
            }
        }
        if (needZero) ensureZeroAtFront();

        const int32_t n2 = static_cast<int32_t>(nodes.size());
        for (int32_t i = 0; i < n2; ++i) {
            Node& node = nodes[static_cast<size_t>(i)];
            for (uint8_t k = 0; k < node.arity; ++k) {
                const int32_t s = node.in[k];
                if (s < 0 || s >= i) node.in[k] = 0;   // 0 = Const 0
            }
        }
        for (int32_t& o : outputs) if (o < 0 || o >= n2) o = 0;
        for (int32_t& i : inputs)  if (i < 0 || i >= n2) i = 0;
    }
};

// ============================================================================
//  3. Исполнение
// ============================================================================

// Быстрый путь: ни одной аллокации, ни одной проверки границ.
// values — рабочий буфер длиной nodes.size(), переиспользуется между вызовами.
//
// memset нужен ровно для одного: гарантировать 0 в тех слотах, которые ещё не
// посчитаны (на случай графа, отремонтированного в Const 0 в конце массива).
inline void executeInto(const Graph& g,
                        const int32_t* __restrict in, size_t nIn,
                        int32_t* __restrict out,
                        int32_t* __restrict values) noexcept {
    const Node* __restrict nodes = g.nodes.data();
    const size_t n = g.nodes.size();

    std::memset(values, 0, n * sizeof(int32_t));

    // 1) подать входные значения
    const size_t ni = std::min(nIn, g.inputs.size());
    for (size_t i = 0; i < ni; ++i) values[g.inputs[i]] = in[i];

    // 2) линейный проход по узлам (топологический порядок гарантирован)
    for (size_t i = 0; i < n; ++i) {
        const Node& nd = nodes[i];
        switch (nd.op) {
            case Op::Input:
                break;                                  // значение уже записано
            case Op::Add:
                values[i] = static_cast<int32_t>(
                    static_cast<uint32_t>(values[nd.in[0]]) +
                    static_cast<uint32_t>(values[nd.in[1]]));   // без UB на переполнении
                break;
            case Op::Mul:
                values[i] = static_cast<int32_t>(
                    static_cast<int64_t>(values[nd.in[0]]) *
                    static_cast<int64_t>(values[nd.in[1]]));
                break;
            case Op::Mod10:
                values[i] = values[nd.in[0]] % 10;
                break;
            case Op::Div10:
                values[i] = values[nd.in[0]] / 10;
                break;
            case Op::Copy:
                values[i] = values[nd.in[0]];
                break;
            case Op::Const:
                values[i] = nd.in[0];
                break;
            case Op::And:
                values[i] = values[nd.in[0]] & values[nd.in[1]];
                break;
            case Op::Or:
                values[i] = values[nd.in[0]] | values[nd.in[1]];
                break;
            case Op::Xor:
                values[i] = values[nd.in[0]] ^ values[nd.in[1]];
                break;
            case Op::Nand:
                // NOT (a AND b). Для битов 0/1 результат тоже 0/1.
                values[i] = 1 - (values[nd.in[0]] & values[nd.in[1]]);
                break;
            default:
                values[i] = 0;
                break;
        }
    }

    // 3) снять выходы
    for (size_t i = 0; i < g.outputs.size(); ++i) out[i] = values[g.outputs[i]];
}

// Публичный API из ТЗ: execute(graph, inputs) -> outputs.
std::vector<int32_t> execute(const Graph& g, const std::vector<int32_t>& inputs) {
    std::vector<int32_t> values(g.nodes.size(), 0);
    std::vector<int32_t> out(g.outputs.size(), 0);
    executeInto(g, inputs.data(), inputs.size(), out.data(), values.data());
    return out;
}

// ============================================================================
//  4. Структурный граф умножения (каскадный десятичный сумматор)
// ============================================================================
//
//  Вход:  a0..a_{d-1}, b0..b_{d-1}    (младшие разряды первыми)
//  Выход: r0..r_{2d-1}
//
//      p[i][j]   = a_i * b_j                  — d² частичных произведений
//      colsum[k] = Σ p[i][j] по i + j == k    — «колонка» умножения в столбик
//
//  Каскад (это и есть ripple-carry):
//      s_k     = colsum[k] + carry_{k-1}
//      digit_k = s_k % 10
//      carry_k = s_k / 10
//
//  КЛЮЧЕВОЙ МОМЕНТ, из-за которого ломалась Python-версия:
//  carry_k — это ЦЕЛОЕ число, а не одна цифра. Для 4×4 максимум colsum = 324,
//  поэтому перенос доходит до ~35. Python-код считал узел переноса, но строил
//  его во ВТОРОМ проходе — уже после того, как все суммы колонок были собраны.
//  В момент сборки colsum[k] список carries[k] был пуст, поэтому перенос НЕ
//  подключался к следующей колонке вообще: получалась сумма частичных
//  произведений без переносов. Ошибка проявлялась при s_k >= 10 (то есть
//  практически всегда), а не только при s_k > 99.
//
//  Здесь перенос — обычный узел Div10, и он явно подаётся на вход суммы
//  следующей колонки. Каскад замкнут, поэтому s_k любого размера (100, 324,
//  1000) обрабатывается корректно: деление на 10 и остаток дают верную цифру
//  и верный перенос.
// ============================================================================

struct MulGraph {
    Graph   graph;
    int32_t overflowCarry = -1;   // перенос из последней колонки (для 4×4 всегда 0)
};

// Сбалансированное дерево сложений: меньше глубина зависимости — больше ILP.
static int32_t addTree(Graph& g, const std::vector<int32_t>& terms) {
    if (terms.empty()) return -1;
    if (terms.size() == 1) return terms[0];

    std::vector<int32_t> cur = terms;
    while (cur.size() > 1) {
        std::vector<int32_t> next;
        next.reserve((cur.size() + 1) / 2);
        size_t i = 0;
        for (; i + 1 < cur.size(); i += 2) {
            next.push_back(g.addNode(Op::Add, cur[i], cur[i + 1]));
        }
        if (i < cur.size()) next.push_back(cur[i]);
        cur.swap(next);
    }
    return cur[0];
}

static MulGraph buildMulGraph(int digits) {
    MulGraph res;
    Graph& g = res.graph;

    const int d     = digits;
    const int nCols = 2 * d;

    g.nodes.reserve(static_cast<size_t>(2 * d + d * d + 2 * nCols + nCols));

    // --- входы: a0..a_{d-1}, затем b0..b_{d-1} ---
    std::vector<int32_t> a(static_cast<size_t>(d));
    std::vector<int32_t> b(static_cast<size_t>(d));
    for (int i = 0; i < d; ++i) a[static_cast<size_t>(i)] = g.addInput();
    for (int j = 0; j < d; ++j) b[static_cast<size_t>(j)] = g.addInput();

    // --- частичные произведения p[i][j] = a_i * b_j ---
    std::vector<int32_t> p(static_cast<size_t>(d) * static_cast<size_t>(d), -1);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            p[static_cast<size_t>(i * d + j)] =
                g.addNode(Op::Mul, a[static_cast<size_t>(i)], b[static_cast<size_t>(j)]);
        }
    }

    // --- каскадный сумматор по колонкам ---
    std::vector<int32_t> digitIdx(static_cast<size_t>(nCols), -1);
    std::vector<int32_t> terms;
    terms.reserve(static_cast<size_t>(d));

    int32_t carry  = -1;   // carry_{k-1}; -1 = переноса нет
    for (int k = 0; k < nCols; ++k) {
        // 1. частичные произведения колонки k
        terms.clear();
        for (int i = 0; i < d; ++i) {
            const int j = k - i;
            if (j >= 0 && j < d) {
                terms.push_back(p[static_cast<size_t>(i * d + j)]);
            }
        }

        // 2. сумма колонки (дерево, а не цепочка)
        bool hasSum = false;
        int32_t sum = -1;
        if (!terms.empty()) {
            sum    = addTree(g, terms);
            hasSum = true;
        }

        // 3. ПЕРЕНОС ИЗ ПРЕДЫДУЩЕЙ КОЛОНКИ — то, чего не было в Python
        if (carry >= 0) {
            sum    = hasSum ? g.addNode(Op::Add, sum, carry) : carry;
            hasSum = true;
        }

        // 4. колонка без слагаемых и без переноса — это ноль
        if (!hasSum) sum = g.addConst(0);

        // 5. цифра = s_k % 10, новый перенос = s_k / 10
        digitIdx[static_cast<size_t>(k)] = g.addNode(Op::Mod10, sum);
        carry = g.addNode(Op::Div10, sum);
    }

    res.overflowCarry = carry;
    g.outputs = digitIdx;
    return res;
}

// ============================================================================
//  5. Данные и бенчмарк
// ============================================================================

struct Dataset {
    int      digits = 4;
    size_t   n      = 0;
    std::vector<int32_t> inputs;   // n * 2*digits
    std::vector<int32_t> targets;  // n * 2*digits
    std::vector<int32_t> a, b;     // исходные числа (для отчёта)

    const int32_t* in(size_t e) const noexcept {
        return inputs.data() + e * static_cast<size_t>(2 * digits);
    }
    const int32_t* tg(size_t e) const noexcept {
        return targets.data() + e * static_cast<size_t>(2 * digits);
    }
};

// 10^e. int64 обязателен: для 9-значных множителей нужен 10^17.
static int64_t ipow10(int e) {
    int64_t r = 1;
    while (e-- > 0) r *= 10;
    return r;
}

static Dataset genExamples(size_t n, int digits, uint64_t seed) {
    Dataset ds;
    ds.digits = digits;
    ds.n      = n;

    const int32_t lo = static_cast<int32_t>(ipow10(digits - 1));
    const int32_t hi = static_cast<int32_t>(ipow10(digits) - 1);
    const uint64_t span = static_cast<uint64_t>(hi - lo) + 1ULL;

    std::vector<int64_t> pow10(static_cast<size_t>(2 * digits));
    for (int k = 0; k < 2 * digits; ++k) pow10[static_cast<size_t>(k)] = ipow10(k);

    ds.inputs.resize(n * static_cast<size_t>(2 * digits));
    ds.targets.resize(n * static_cast<size_t>(2 * digits));
    ds.a.resize(n);
    ds.b.resize(n);

    std::mt19937_64 rng(seed);

    for (size_t e = 0; e < n; ++e) {
        const int32_t x = lo + static_cast<int32_t>(rng() % span);
        const int32_t y = lo + static_cast<int32_t>(rng() % span);
        const int64_t prod = static_cast<int64_t>(x) * static_cast<int64_t>(y);

        ds.a[e] = x;
        ds.b[e] = y;

        int32_t* in = ds.inputs.data() + e * static_cast<size_t>(2 * digits);
        for (int i = 0; i < digits; ++i) {
            in[i]          = static_cast<int32_t>((static_cast<int64_t>(x) / pow10[static_cast<size_t>(i)]) % 10);
            in[digits + i] = static_cast<int32_t>((static_cast<int64_t>(y) / pow10[static_cast<size_t>(i)]) % 10);
        }

        int32_t* tg = ds.targets.data() + e * static_cast<size_t>(2 * digits);
        for (int k = 0; k < 2 * digits; ++k) {
            tg[k] = static_cast<int32_t>((prod / pow10[static_cast<size_t>(k)]) % 10);
        }
    }
    return ds;
}

struct BenchStats {
    size_t examples      = 0;
    size_t correctDigits = 0;
    size_t totalDigits   = 0;
    size_t correctWhole  = 0;   // примеров, где верны ВСЕ цифры
    double accuracyDigits  = 0.0;
    double accuracyWhole   = 0.0;
    double seconds         = 0.0;
    double examplesPerSec  = 0.0;
    size_t repeats         = 0;
};

static BenchStats runBenchmark(const Graph& g, const Dataset& ds, double minSeconds) {
    using clock = std::chrono::steady_clock;

    BenchStats st;
    st.examples = ds.n;

    std::vector<int32_t> values(g.nodes.size(), 0);
    std::vector<int32_t> out(g.outputs.size(), 0);
    const int cols = 2 * ds.digits;

    // --- точность: один честный проход по всем примерам ---
    for (size_t e = 0; e < ds.n; ++e) {
        executeInto(g, ds.in(e), static_cast<size_t>(2 * ds.digits),
                    out.data(), values.data());
        bool whole = true;
        for (int k = 0; k < cols; ++k) {
            ++st.totalDigits;
            if (out[static_cast<size_t>(k)] == ds.tg(e)[k]) {
                ++st.correctDigits;
            } else {
                whole = false;
            }
        }
        if (whole) ++st.correctWhole;
    }

    // --- скорость: повторяем батч, пока не наберём окно измерения ---
    for (int warm = 0; warm < 3; ++warm) {
        for (size_t e = 0; e < ds.n; ++e) {
            executeInto(g, ds.in(e), static_cast<size_t>(2 * ds.digits),
                        out.data(), values.data());
        }
    }

    const auto t0 = clock::now();
    double elapsed = 0.0;
    do {
        for (size_t e = 0; e < ds.n; ++e) {
            executeInto(g, ds.in(e), static_cast<size_t>(2 * ds.digits),
                        out.data(), values.data());
        }
        ++st.repeats;
        elapsed = std::chrono::duration<double>(clock::now() - t0).count();
    } while (elapsed < minSeconds && st.repeats < 1000000);

    st.seconds = std::max(elapsed, 1e-9);
    st.examplesPerSec = static_cast<double>(ds.n * st.repeats) / st.seconds;

    st.accuracyDigits = st.totalDigits
        ? static_cast<double>(st.correctDigits) / static_cast<double>(st.totalDigits)
        : 0.0;
    st.accuracyWhole = ds.n
        ? static_cast<double>(st.correctWhole) / static_cast<double>(ds.n)
        : 0.0;

    return st;
}

// Диагностика каскада: какие суммы и переносы реально возникают.
struct CascadeInfo {
    int32_t maxColumnSum = 0;
    int32_t maxCarry     = 0;
    size_t  worst        = 0;
};

static CascadeInfo analyzeCascade(const Dataset& ds) {
    CascadeInfo info;
    const int d = ds.digits;

    for (size_t e = 0; e < ds.n; ++e) {
        int32_t carry = 0;
        for (int k = 0; k < 2 * d; ++k) {
            int32_t s = carry;
            for (int i = 0; i < d; ++i) {
                const int j = k - i;
                if (j >= 0 && j < d) {
                    s += ds.in(e)[i] * ds.in(e)[d + j];
                }
            }
            if (carry > info.maxCarry) info.maxCarry = carry;
            if (s > info.maxColumnSum) {
                info.maxColumnSum = s;
                info.worst = e;
            }
            carry = s / 10;
        }
    }
    return info;
}

// ============================================================================
//  6. Готовность к эволюции: мутации и скрещивание
// ============================================================================

struct Rng {
    std::mt19937_64 gen;
    explicit Rng(uint64_t seed) : gen(seed) {}

    double unit() { return static_cast<double>(gen() >> 11) * (1.0 / 9007199254740992.0); }
    bool   chance(double p) { return unit() < p; }
    int32_t range(int32_t lo, int32_t hi) {
        return lo + static_cast<int32_t>(gen() % (static_cast<uint64_t>(hi - lo) + 1ULL));
    }
};

// Случайный граф — стартовая точка для «чистой» эволюции, без структурного
// знания о задаче. Никакой арифметики в нём нет: это просто шум из операций.
static Graph buildRandomGraph(int digits, int internalNodes, uint64_t seed) {
    static const Op kOps[] = {Op::Add, Op::Mul, Op::Mod10, Op::Div10, Op::Copy};
    constexpr int kNumOps = static_cast<int>(sizeof(kOps) / sizeof(kOps[0]));

    Graph g;
    g.addConst(0);                                     // узел 0 — гарантированный ноль
    for (int i = 0; i < 2 * digits; ++i) g.addInput(); // входы

    Rng rng(seed);
    for (int k = 0; k < internalNodes; ++k) {
        const Op op = kOps[rng.range(0, kNumOps - 1)];
        const int32_t hi = static_cast<int32_t>(g.nodes.size()) - 1;
        const int32_t a = rng.range(0, hi);
        const int32_t b = (arityOf(op) == 2) ? rng.range(0, hi) : 0;
        g.addNode(op, a, b);
    }

    // Выходы — последние 2*digits узлов.
    const int32_t n = static_cast<int32_t>(g.nodes.size());
    for (int i = 0; i < 2 * digits; ++i) {
        g.outputs.push_back(n - 2 * digits + i);
    }
    g.repair();
    return g;
}

// Мутация: сменить тип операции или перенаправить одно ребро.
// Входы и константы неприкосновенны — иначе граф теряет интерфейс.
static void mutate(Graph& g, double rate, Rng& rng) {
    static const Op kMutableOps[] = {Op::Add, Op::Mul, Op::Mod10, Op::Div10, Op::Copy};
    constexpr int kNumMutable = static_cast<int>(sizeof(kMutableOps) / sizeof(kMutableOps[0]));

    const int32_t n = static_cast<int32_t>(g.nodes.size());
    for (int32_t i = 1; i < n; ++i) {          // 0 — зарезервирован под Const 0
        Node& node = g.nodes[static_cast<size_t>(i)];
        if (isLeafOp(node.op)) continue;
        if (!rng.chance(rate)) continue;

        if (rng.chance(0.35)) {
            node.op    = kMutableOps[rng.range(0, kNumMutable - 1)];
            node.arity = arityOf(node.op);
        } else {
            const uint8_t slot = static_cast<uint8_t>(rng.range(0, node.arity - 1));
            node.in[slot] = rng.range(0, i - 1);
        }
    }
    g.repair();
}

// Скрещивание: префикс узлов от одного родителя, хвост от другого.
// Рёбра хвоста сохраняют свои индексы, поэтому структура «сшивается» без
// переиндексации; конфликты устраняет repair().
static Graph crossover(const Graph& a, const Graph& b, Rng& rng) {
    const size_t n = std::min(a.nodes.size(), b.nodes.size());
    if (n < 3) return a;

    const size_t cut = 1 + static_cast<size_t>(rng.range(0, static_cast<int32_t>(n) - 2));

    Graph child;
    child.nodes.reserve(a.nodes.size() + b.nodes.size());
    for (size_t i = 0; i < cut; ++i) child.nodes.push_back(a.nodes[i]);
    for (size_t i = cut; i < b.nodes.size(); ++i) child.nodes.push_back(b.nodes[i]);

    child.inputs  = rng.chance(0.5) ? a.inputs  : b.inputs;
    child.outputs = rng.chance(0.5) ? a.outputs : b.outputs;
    child.repair();
    return child;
}

static double fitness(const Graph& g, const Dataset& ds) {
    std::vector<int32_t> values(g.nodes.size(), 0);
    std::vector<int32_t> out(g.outputs.size(), 0);
    const int cols = 2 * ds.digits;

    size_t ok = 0, total = 0;
    for (size_t e = 0; e < ds.n; ++e) {
        executeInto(g, ds.in(e), static_cast<size_t>(2 * ds.digits),
                    out.data(), values.data());
        for (int k = 0; k < cols; ++k) {
            ++total;
            if (out[static_cast<size_t>(k)] == ds.tg(e)[k]) ++ok;
        }
    }
    return total ? static_cast<double>(ok) / static_cast<double>(total) : 0.0;
}

// Турнирный отбор + элитизм. Демонстрирует, что структура готова к эволюции:
// операторы не знают ничего о задаче, граф — просто данные.
static Graph evolve(const Graph& seedGraph, const Dataset& train,
                    int popSize, int generations, double rate,
                    uint64_t seed, bool verbose) {
    Rng rng(seed);

    std::vector<Graph> pop;
    pop.reserve(static_cast<size_t>(popSize));
    pop.push_back(seedGraph);
    for (int i = 1; i < popSize; ++i) {
        Graph child = seedGraph;
        mutate(child, rate, rng);
        pop.push_back(std::move(child));
    }

    std::vector<double> fit(static_cast<size_t>(popSize), 0.0);
    for (int i = 0; i < popSize; ++i) fit[static_cast<size_t>(i)] = fitness(pop[static_cast<size_t>(i)], train);

    size_t bestIdx = 0;
    for (int i = 1; i < popSize; ++i) {
        if (fit[static_cast<size_t>(i)] > fit[bestIdx]) bestIdx = static_cast<size_t>(i);
    }
    double bestFit = fit[bestIdx];
    Graph  best    = pop[bestIdx];

    for (int gen = 0; gen < generations; ++gen) {
        std::vector<Graph> next;
        next.reserve(static_cast<size_t>(popSize));
        next.push_back(best);                       // элитизм: лучшее не теряем

        while (static_cast<int>(next.size()) < popSize) {
            const int32_t i1 = rng.range(0, popSize - 1);
            const int32_t i2 = rng.range(0, popSize - 1);
            const int32_t i3 = rng.range(0, popSize - 1);
            const int32_t winner =
                (fit[static_cast<size_t>(i1)] >= fit[static_cast<size_t>(i2)])
                    ? ((fit[static_cast<size_t>(i1)] >= fit[static_cast<size_t>(i3)]) ? i1 : i3)
                    : ((fit[static_cast<size_t>(i2)] >= fit[static_cast<size_t>(i3)]) ? i2 : i3);

            Graph child = pop[static_cast<size_t>(winner)];
            if (rng.chance(0.25) && popSize > 2) {
                const int32_t other = rng.range(0, popSize - 1);
                child = crossover(child, pop[static_cast<size_t>(other)], rng);
            }
            mutate(child, rate, rng);
            next.push_back(std::move(child));
        }

        pop.swap(next);
        for (int i = 0; i < popSize; ++i) fit[static_cast<size_t>(i)] = fitness(pop[static_cast<size_t>(i)], train);

        const size_t genBest = static_cast<size_t>(std::max_element(fit.begin(), fit.end()) - fit.begin());
        if (fit[genBest] > bestFit) {
            bestFit = fit[genBest];
            best    = pop[genBest];
        }

        if (verbose && (gen % 5 == 0 || gen == generations - 1)) {
            std::printf("    gen %3d  best_в_поколении=%.4f  overall=%.4f  узлов=%zu\n",
                        gen, fit[genBest], bestFit, best.nodeCount());
        }
    }
    return best;
}

// ============================================================================
//  7. Вывод результатов
// ============================================================================

static std::string nowIso() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    return std::string(buf);
}

static bool dirExists(const std::filesystem::path& p) {
    std::error_code ec;
    return std::filesystem::is_directory(p, ec);
}

// Ищем папку results/ рядом с текущей: ./results, ../results, ../../results.
static std::filesystem::path findResultsDir() {
    namespace fs = std::filesystem;
    for (const char* rel : {"results", "../results", "../../results"}) {
        if (dirExists(rel)) return fs::path(rel);
    }
    std::error_code ec;
    fs::create_directories("results", ec);
    return fs::path("results");
}

static bool writeJson(const std::filesystem::path& path,
                      int digits, size_t nExamples, uint64_t seed,
                      const Graph& g, const BenchStats& st,
                      const CascadeInfo& ci, bool overflowZero) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    f << std::fixed << std::setprecision(6);
    f << "{\n"
      << "  \"engine\": \"cpp\",\n"
      << "  \"task\": \"multiplication\",\n"
      << "  \"digits\": " << digits << ",\n"
      << "  \"examples\": " << nExamples << ",\n"
      << "  \"seed\": " << seed << ",\n"
      << "  \"graph\": {\n"
      << "    \"nodes\": " << g.nodeCount() << ",\n"
      << "    \"edges\": " << g.edgeCount() << ",\n"
      << "    \"params\": " << g.paramCount() << ",\n"
      << "    \"inputs\": " << g.inputs.size() << ",\n"
      << "    \"outputs\": " << g.outputs.size() << "\n"
      << "  },\n"
      << "  \"accuracy_digits\": " << st.accuracyDigits << ",\n"
      << "  \"accuracy_whole_examples\": " << st.accuracyWhole << ",\n"
      << "  \"correct_digits\": " << st.correctDigits << ",\n"
      << "  \"total_digits\": " << st.totalDigits << ",\n"
      << "  \"correct_whole_examples\": " << st.correctWhole << ",\n"
      << "  \"seconds\": " << st.seconds << ",\n"
      << "  \"repeats\": " << st.repeats << ",\n"
      << "  \"examples_per_sec\": " << st.examplesPerSec << ",\n"
      << "  \"cascade\": {\n"
      << "    \"max_column_sum\": " << ci.maxColumnSum << ",\n"
      << "    \"max_carry\": " << ci.maxCarry << ",\n"
      << "    \"overflow_carry_zero\": " << (overflowZero ? "true" : "false") << "\n"
      << "  },\n"
      << "  \"timestamp\": \"" << nowIso() << "\"\n"
      << "}\n";
    return true;
}

// ============================================================================
//  8. main
// ============================================================================

// ============================================================================
//  9. Загрузка графа, обученного в Python (формат diffgraph)
// ============================================================================
//
// Обучение структуры живёт в Python (src/python/diffgraph): там граф
// дифференцируемый, и топология подбирается градиентом через Gumbel-Softmax.
// На выходе получается ЖЁСТКИЙ граф — конкретные операции и конкретные рёбра, —
// который сохраняется в JSON. Здесь этот JSON читается и исполняется тем же
// целочисленным движком, что и умножение: ни нейросетей, ни float, ни матриц.
//
// Формат (version 1):
//   { "format": "diffgraph", "version": 1, "task": "parity4",
//     "n_inputs": 4, "n_outputs": 1,
//     "nodes": [ {"op":"INPUT"}, {"op":"XOR","in":[0,3]}, ... ],
//     "outputs": [6], "meta": {...} }
//
// Индексация сквозная: 0..n_inputs-1 — входы, дальше узлы вычислений. Ровно так
// же индексируется Node::in[] в этом движке, поэтому пересчёт индексов не нужен.

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type        type = Type::Null;
    bool        boolean = false;
    double      number = 0.0;
    std::string text;
    std::vector<JsonValue> items;                              // для Array
    std::vector<std::pair<std::string, JsonValue>> fields;     // для Object

    const JsonValue* find(const std::string& key) const {
        if (type != Type::Object) return nullptr;
        for (const auto& f : fields) if (f.first == key) return &f.second;
        return nullptr;
    }
};

// Маленький рекурсивный парсер JSON: ровно столько, сколько нужно формату.
// Сторонние библиотеки в проекте запрещены, а формат простой и стабильный.
class JsonParser {
public:
    explicit JsonParser(const std::string& src)
        : begin_(src.data()), p_(src.data()), end_(src.data() + src.size()) {}

    bool parse(JsonValue& out) {
        skipWs();
        if (!parseValue(out)) return false;
        skipWs();
        if (p_ != end_) return fail("лишние данные после JSON");
        return true;
    }

    const std::string& error() const { return err_; }

private:
    const char* begin_;
    const char* p_;
    const char* end_;
    std::string err_;

    bool fail(const char* msg) {
        // Смещение в байтах — чтобы человек, правящий JSON руками, нашёл место.
        if (err_.empty())
            err_ = std::string(msg) + " (смещение " + std::to_string(p_ - begin_) + ")";
        return false;
    }

    void skipWs() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) ++p_;
    }

    bool literal(const char* s) {
        const size_t n = std::strlen(s);
        if (static_cast<size_t>(end_ - p_) < n || std::strncmp(p_, s, n) != 0)
            return fail("неверный литерал");
        p_ += n;
        return true;
    }

    bool parseValue(JsonValue& v) {
        skipWs();
        if (p_ >= end_) return fail("неожиданный конец файла");
        switch (*p_) {
            case '{': return parseObject(v);
            case '[': return parseArray(v);
            case '"':
                v.type = JsonValue::Type::String;
                return parseString(v.text);
            case 't':
                if (!literal("true")) return false;
                v.type = JsonValue::Type::Bool;
                v.boolean = true;
                return true;
            case 'f':
                if (!literal("false")) return false;
                v.type = JsonValue::Type::Bool;
                v.boolean = false;
                return true;
            case 'n':
                if (!literal("null")) return false;
                v.type = JsonValue::Type::Null;
                return true;
            default:
                return parseNumber(v);
        }
    }

    bool parseNumber(JsonValue& v) {
        char* stop = nullptr;
        const double d = std::strtod(p_, &stop);
        if (stop == p_) return fail("ожидалось число");
        p_ = stop;
        v.type = JsonValue::Type::Number;
        v.number = d;
        return true;
    }

    bool parseString(std::string& out) {
        if (p_ >= end_ || *p_ != '"') return fail("ожидалась строка");
        ++p_;
        out.clear();
        while (p_ < end_) {
            const char c = *p_++;
            if (c == '"') return true;
            if (c != '\\') { out.push_back(c); continue; }
            if (p_ >= end_) break;
            const char esc = *p_++;
            switch (esc) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'u': {
                    if (end_ - p_ < 4) return fail("обрезанный \\u");
                    unsigned cp = 0;
                    for (int i = 0; i < 4; ++i) {
                        const char h = *p_++;
                        cp <<= 4;
                        if (h >= '0' && h <= '9')      cp |= static_cast<unsigned>(h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= static_cast<unsigned>(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= static_cast<unsigned>(h - 'A' + 10);
                        else return fail("плохая \\u-последовательность");
                    }
                    // BMP -> UTF-8; суррогатные пары нам не нужны.
                    if (cp < 0x80) {
                        out.push_back(static_cast<char>(cp));
                    } else if (cp < 0x800) {
                        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                    } else {
                        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                    }
                    break;
                }
                default:
                    return fail("неизвестная escape-последовательность");
            }
        }
        return fail("незакрытая строка");
    }

    bool parseArray(JsonValue& v) {
        v.type = JsonValue::Type::Array;
        ++p_;                                   // '['
        skipWs();
        if (p_ < end_ && *p_ == ']') { ++p_; return true; }
        for (;;) {
            JsonValue item;
            if (!parseValue(item)) return false;
            v.items.push_back(std::move(item));
            skipWs();
            if (p_ >= end_) return fail("незакрытый массив");
            if (*p_ == ',') { ++p_; continue; }
            if (*p_ == ']') { ++p_; return true; }
            return fail("ожидалась ',' или ']'");
        }
    }

    bool parseObject(JsonValue& v) {
        v.type = JsonValue::Type::Object;
        ++p_;                                   // '{'
        skipWs();
        if (p_ < end_ && *p_ == '}') { ++p_; return true; }
        for (;;) {
            skipWs();
            std::string key;
            if (!parseString(key)) return false;
            skipWs();
            if (p_ >= end_ || *p_ != ':') return fail("ожидалось ':'");
            ++p_;
            JsonValue val;
            if (!parseValue(val)) return false;
            v.fields.emplace_back(std::move(key), std::move(val));
            skipWs();
            if (p_ >= end_) return fail("незакрытый объект");
            if (*p_ == ',') { ++p_; continue; }
            if (*p_ == '}') { ++p_; return true; }
            return fail("ожидалась ',' или '}'");
        }
    }
};

static bool readTextFile(const std::filesystem::path& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const std::streamoff size = f.tellg();
    if (size < 0) return false;
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(size));
    if (size > 0) f.read(out.data(), size);

    // Блокнот и PowerShell сохраняют UTF-8 с BOM. Если его не срезать,
    // парсер увидит мусорный байт вместо '{' и скажет «ожидалось число».
    if (out.size() >= 3 &&
        static_cast<unsigned char>(out[0]) == 0xEF &&
        static_cast<unsigned char>(out[1]) == 0xBB &&
        static_cast<unsigned char>(out[2]) == 0xBF) {
        out.erase(0, 3);
    }
    return true;
}

struct LoadedGraph {
    Graph       graph;
    std::string task;
    int         nInputs  = 0;
    int         nOutputs = 0;
};

// Читает diffgraph-JSON и собирает из него обычный Graph движка.
static bool loadDiffGraph(const std::filesystem::path& path, LoadedGraph& out, std::string& err) {
    std::string text;
    if (!readTextFile(path, text)) { err = "не удалось прочитать файл"; return false; }

    JsonValue root;
    JsonParser parser(text);
    if (!parser.parse(root)) { err = "JSON: " + parser.error(); return false; }
    if (root.type != JsonValue::Type::Object) { err = "корень JSON должен быть объектом"; return false; }

    const JsonValue* format = root.find("format");
    if (!format || format->type != JsonValue::Type::String || format->text != "diffgraph") {
        err = "это не файл формата diffgraph";
        return false;
    }

    const JsonValue* task = root.find("task");
    if (!task || task->type != JsonValue::Type::String) { err = "нет поля task"; return false; }
    out.task = task->text;

    const JsonValue* nIn  = root.find("n_inputs");
    const JsonValue* nOut = root.find("n_outputs");
    if (!nIn || !nOut) { err = "нет полей n_inputs/n_outputs"; return false; }
    out.nInputs  = static_cast<int>(nIn->number);
    out.nOutputs = static_cast<int>(nOut->number);

    const JsonValue* nodes = root.find("nodes");
    if (!nodes || nodes->type != JsonValue::Type::Array) { err = "нет массива nodes"; return false; }

    out.graph.clear();
    for (size_t i = 0; i < nodes->items.size(); ++i) {
        const JsonValue& item = nodes->items[i];
        if (item.type != JsonValue::Type::Object) { err = "узел должен быть объектом"; return false; }

        const JsonValue* name = item.find("op");
        if (!name || name->type != JsonValue::Type::String) { err = "у узла нет поля op"; return false; }

        Op op = Op::Const;
        if (!opFromName(name->text, op)) {
            err = "неизвестная операция: " + name->text;
            return false;
        }

        int32_t a = 0;
        int32_t b = 0;
        const JsonValue* in = item.find("in");
        if (in && in->type == JsonValue::Type::Array) {
            if (in->items.size() > 0) a = static_cast<int32_t>(in->items[0].number);
            if (in->items.size() > 1) b = static_cast<int32_t>(in->items[1].number);
        }

        const int32_t idx = out.graph.addNode(op, a, b);
        if (op == Op::Input) {
            out.graph.inputs.push_back(idx);
            if (idx != static_cast<int32_t>(i)) {
                err = "входные узлы должны идти первыми";
                return false;
            }
        }
    }

    const JsonValue* outs = root.find("outputs");
    if (!outs || outs->type != JsonValue::Type::Array) { err = "нет массива outputs"; return false; }
    for (const JsonValue& o : outs->items) out.graph.outputs.push_back(static_cast<int32_t>(o.number));

    if (static_cast<int>(out.graph.inputs.size()) != out.nInputs) {
        err = "число входов не совпало с n_inputs";
        return false;
    }
    if (static_cast<int>(out.graph.outputs.size()) != out.nOutputs) {
        err = "число выходов не совпало с n_outputs";
        return false;
    }
    if (!out.graph.isValidDag()) { err = "граф не является корректным DAG"; return false; }
    return true;
}

// ---------------------------------------------------------------------------
// Эталонные таблицы истинности. Считаются здесь, в C++, независимо от Python:
// иначе проверка «граф решает задачу» ничего не проверяет.
// ---------------------------------------------------------------------------
struct TaskTable {
    std::string name;
    int nInputs  = 0;
    int nOutputs = 0;
    std::vector<std::vector<int32_t>> rows;       // входы
    std::vector<std::vector<int32_t>> expected;   // ожидаемые выходы
};

static bool buildTaskTable(const std::string& name, TaskTable& t, std::string& err) {
    if (name != "parity4" && name != "add2") {
        err = "неизвестная задача: " + name + " (есть parity4, add2)";
        return false;
    }

    t.name     = name;
    t.nInputs  = 4;
    t.nOutputs = (name == "parity4") ? 1 : 3;

    const int combos = 1 << t.nInputs;
    for (int m = 0; m < combos; ++m) {
        std::vector<int32_t> in(4);
        for (int i = 0; i < 4; ++i) in[static_cast<size_t>(i)] = (m >> i) & 1;
        t.rows.push_back(in);

        if (name == "parity4") {
            t.expected.push_back({ in[0] ^ in[1] ^ in[2] ^ in[3] });
        } else {
            const int32_t a = in[0] + 2 * in[1];   // a0 a1
            const int32_t b = in[2] + 2 * in[3];   // b0 b1
            const int32_t s = a + b;               // 0..6
            t.expected.push_back({ s & 1, (s >> 1) & 1, (s >> 2) & 1 });
        }
    }
    return true;
}

struct LoadReport {
    size_t  correctRows  = 0;
    size_t  totalRows    = 0;
    size_t  correctBits  = 0;
    size_t  totalBits    = 0;
    bool    booleanOnly  = true;    // все значения графа остались в {0, 1}
    int32_t maxValue     = 0;
    double  seconds      = 0.0;
    size_t  repeats      = 0;
};

static LoadReport runLoadedGraph(const LoadedGraph& lg, const TaskTable& tt, double minSeconds) {
    LoadReport rep;
    const Graph& g = lg.graph;

    std::vector<int32_t> values(g.nodes.size(), 0);
    std::vector<int32_t> out(static_cast<size_t>(lg.nOutputs), 0);

    rep.totalRows = tt.rows.size();
    rep.totalBits = rep.totalRows * static_cast<size_t>(lg.nOutputs);

    for (size_t r = 0; r < tt.rows.size(); ++r) {
        executeInto(g, tt.rows[r].data(), tt.rows[r].size(), out.data(), values.data());

        bool rowOk = true;
        for (int k = 0; k < lg.nOutputs; ++k) {
            if (out[static_cast<size_t>(k)] == tt.expected[r][static_cast<size_t>(k)])
                ++rep.correctBits;
            else
                rowOk = false;
        }
        if (rowOk) ++rep.correctRows;

        for (const int32_t v : values) {
            if (v != 0 && v != 1) rep.booleanOnly = false;
            if (v > rep.maxValue) rep.maxValue = v;
        }
    }

    // Скорость: гоняем всю таблицу, пока не истечёт окно измерения.
    const auto t0 = std::chrono::steady_clock::now();
    size_t repeats = 0;
    do {
        for (size_t r = 0; r < tt.rows.size(); ++r) {
            executeInto(g, tt.rows[r].data(), tt.rows[r].size(), out.data(), values.data());
        }
        ++repeats;
    } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < minSeconds);

    rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    rep.repeats = repeats;
    return rep;
}

// Режим --load: один обученный граф — структура, таблица истинности, вердикт.
struct Options {
    int      digits      = 4;
    size_t   examples    = 1000;
    uint64_t seed        = 42;
    double   minSeconds  = 0.25;
    bool     doEvolve    = false;
    bool     fromRandom  = false;
    int      popSize     = 64;
    int      generations = 20;
    bool     writeJson   = false;
    bool     help        = false;
    std::vector<std::string> loadPaths;   // --load <файл>: исполнить обученные графы
};

static void printHelp() {
    std::printf(
        "Графовое ядро (C++) — умножение на разреженном графе операций\n"
        "\n"
        "Использование: graph_evolve [параметры]\n"
        "\n"
        "  --digits N      число цифр в каждом множителе (1..9, по умолчанию 4)\n"
        "  --examples N    сколько случайных примеров сгенерировать (по умолчанию 1000)\n"
        "  --seed N        зерно генератора (по умолчанию 42)\n"
        "  --time MS       минимальное окно измерения скорости в мс (по умолчанию 250)\n"
        "  --evolve        дополнительно прогнать демо-эволюцию графа\n"
        "  --from-random   эволюция стартует со случайного графа, а не со структурного\n"
        "  --pop N         размер популяции для эволюции (по умолчанию 64)\n"
        "  --gens N        число поколений для эволюции (по умолчанию 20)\n"
        "  --json          сохранить результаты в results/graph_cpp_results.json\n"
        "  --load FILE     исполнить граф, обученный в Python (формат diffgraph),\n"
        "                  и проверить его на таблице истинности задачи\n"
        "  --help          эта справка\n");
}

static bool parseArgs(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](int& out) -> bool {
            if (i + 1 >= argc) return false;
            out = std::atoi(argv[++i]);
            return true;
        };

        if (arg == "--help" || arg == "-h") {
            opt.help = true;
        } else if (arg == "--digits") {
            if (!value(opt.digits)) return false;
            if (opt.digits < 1) opt.digits = 1;
            if (opt.digits > 9) opt.digits = 9;
        } else if (arg == "--examples") {
            int v = 0;
            if (!value(v) || v <= 0) return false;
            opt.examples = static_cast<size_t>(v);
        } else if (arg == "--seed") {
            int v = 0;
            if (!value(v)) return false;
            opt.seed = static_cast<uint64_t>(static_cast<uint32_t>(v));
        } else if (arg == "--time") {
            int v = 0;
            if (!value(v) || v <= 0) return false;
            opt.minSeconds = static_cast<double>(v) / 1000.0;
        } else if (arg == "--pop") {
            if (!value(opt.popSize) || opt.popSize < 2) return false;
        } else if (arg == "--gens") {
            if (!value(opt.generations) || opt.generations < 1) return false;
        } else if (arg == "--evolve") {
            opt.doEvolve = true;
        } else if (arg == "--from-random") {
            opt.fromRandom = true;
        } else if (arg == "--json") {
            opt.writeJson = true;
        } else if (arg == "--load") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "После --load нужен путь к файлу\n");
                return false;
            }
            opt.loadPaths.push_back(argv[++i]);
        } else {
            std::fprintf(stderr, "Неизвестный параметр: %s (см. --help)\n", arg.c_str());
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Режим --load: исполнить обученные графы и проверить их на таблице истинности.
// ---------------------------------------------------------------------------
struct LoadEntry {
    std::string task;
    std::string file;
    size_t      nodes       = 0;
    size_t      edges       = 0;
    size_t      correctRows = 0;
    size_t      totalRows   = 0;
    size_t      correctBits = 0;
    size_t      totalBits   = 0;
    bool        booleanOnly = true;
    double      examplesPerSec = 0.0;
    bool        solved      = false;
};

// Ширина строки в символах, а не в байтах: printf("%-10s") считает байты и
// разъезжается на кириллице (одна буква — два байта в UTF-8).
static size_t displayWidth(const std::string& s) {
    size_t w = 0;
    for (const unsigned char c : s) if ((c & 0xC0) != 0x80) ++w;
    return w;
}

static std::string padRight(const std::string& s, size_t w) {
    const size_t n = displayWidth(s);
    return n >= w ? s : s + std::string(w - n, ' ');
}

static std::string padLeft(const std::string& s, size_t w) {
    const size_t n = displayWidth(s);
    return n >= w ? s : std::string(w - n, ' ') + s;
}

static bool writeLoadResults(const std::filesystem::path& path,
                             const std::vector<LoadEntry>& entries) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    char buf[64];
    f << "{\n";
    f << "  \"generated\": \"" << nowIso() << "\",\n";
    f << "  \"engine\": \"cpp\",\n";
    f << "  \"graphs\": [\n";
    for (size_t i = 0; i < entries.size(); ++i) {
        const LoadEntry& e = entries[i];
        f << "    {\n";
        f << "      \"task\": \"" << e.task << "\",\n";
        f << "      \"file\": \"" << e.file << "\",\n";
        f << "      \"nodes\": " << e.nodes << ",\n";
        f << "      \"edges\": " << e.edges << ",\n";
        std::snprintf(buf, sizeof buf, "%.6f",
                      e.totalRows ? static_cast<double>(e.correctRows) / static_cast<double>(e.totalRows) : 0.0);
        f << "      \"accuracy_examples\": " << buf << ",\n";
        std::snprintf(buf, sizeof buf, "%.6f",
                      e.totalBits ? static_cast<double>(e.correctBits) / static_cast<double>(e.totalBits) : 0.0);
        f << "      \"accuracy_bits\": " << buf << ",\n";
        f << "      \"correct_rows\": " << e.correctRows << ",\n";
        f << "      \"total_rows\": " << e.totalRows << ",\n";
        f << "      \"boolean_domain\": " << (e.booleanOnly ? "true" : "false") << ",\n";
        std::snprintf(buf, sizeof buf, "%.0f", e.examplesPerSec);
        f << "      \"examples_per_sec\": " << buf << ",\n";
        f << "      \"solved\": " << (e.solved ? "true" : "false") << "\n";
        f << "    }" << (i + 1 < entries.size() ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
    return static_cast<bool>(f);
}

// Печатает структуру графа в виде «индекс  ОПЕРАЦИЯ входы».
static void printStructure(const Graph& g) {
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        const Node& n = g.nodes[i];
        if (n.op == Op::Input) {
            std::printf("    %3zu  INPUT\n", i);
        } else if (n.arity == 0) {
            std::printf("    %3zu  %-6s\n", i, opName(n.op));
        } else if (n.arity == 1) {
            std::printf("    %3zu  %-6s %zu\n", i, opName(n.op), static_cast<size_t>(n.in[0]));
        } else {
            std::printf("    %3zu  %-6s %zu, %zu\n", i, opName(n.op),
                        static_cast<size_t>(n.in[0]), static_cast<size_t>(n.in[1]));
        }
    }
}

static bool loadOneGraph(const std::filesystem::path& path, const Options& opt,
                         size_t index, LoadEntry& entry) {
    std::printf("----------------------------------------------------------------------\n");
    std::printf("[%zu] Граф: %s\n", index, path.generic_string().c_str());
    std::printf("----------------------------------------------------------------------\n\n");

    LoadedGraph lg;
    std::string err;
    if (!loadDiffGraph(path, lg, err)) {
        std::fprintf(stderr, "ОШИБКА загрузки: %s\n\n", err.c_str());
        return false;
    }
    const Graph& g = lg.graph;

    std::printf("    задача:     %s\n", lg.task.c_str());
    std::printf("    узлов:      %zu   (%zu входа + %zu вычислительных)\n",
                g.nodeCount(), g.inputs.size(), g.nodeCount() - g.inputs.size());
    std::printf("    связей:     %zu\n", g.edgeCount());
    std::printf("    параметров: %zu   (узлы + связи)\n", g.paramCount());
    std::printf("    входов:     %d\n", lg.nInputs);
    std::printf("    выходов:    %d\n", lg.nOutputs);
    std::printf("    DAG:        OK (все рёбра смотрят строго назад)\n\n");

    std::printf("    структура:\n");
    printStructure(g);
    std::printf("    выходы: ");
    for (size_t i = 0; i < g.outputs.size(); ++i) {
        std::printf("%s%zu", i ? ", " : "", static_cast<size_t>(g.outputs[i]));
    }
    std::printf("\n\n");

    TaskTable tt;
    if (!buildTaskTable(lg.task, tt, err)) {
        std::fprintf(stderr, "ОШИБКА: %s\n\n", err.c_str());
        return false;
    }
    if (tt.nInputs != lg.nInputs || tt.nOutputs != lg.nOutputs) {
        std::fprintf(stderr,
                     "ОШИБКА: у графа %d входов и %d выходов, а у задачи %s — %d и %d\n\n",
                     lg.nInputs, lg.nOutputs, tt.name.c_str(), tt.nInputs, tt.nOutputs);
        return false;
    }

    std::printf("    таблица истинности (эталон посчитан здесь же, в C++):\n");
    {
        std::vector<int32_t> values(g.nodes.size(), 0);
        std::vector<int32_t> out(static_cast<size_t>(lg.nOutputs), 0);
        for (size_t r = 0; r < tt.rows.size(); ++r) {
            executeInto(g, tt.rows[r].data(), tt.rows[r].size(), out.data(), values.data());

            std::string in;
            for (size_t i = 0; i < tt.rows[r].size(); ++i)
                in.push_back(static_cast<char>('0' + tt.rows[r][i]));

            std::string want;
            std::string got;
            bool ok = true;
            for (int k = 0; k < lg.nOutputs; ++k) {
                const int32_t gotValue = out[static_cast<size_t>(k)];
                const int32_t wantValue = tt.expected[r][static_cast<size_t>(k)];
                want.push_back(static_cast<char>('0' + (wantValue ? 1 : 0)));
                got.push_back(static_cast<char>('0' + (gotValue ? 1 : 0)));
                if (gotValue != wantValue) ok = false;
            }
            std::printf("      %-7s %-5s %-5s %s\n",
                        in.c_str(), want.c_str(), got.c_str(), ok ? "OK" : "FAIL");
        }
        std::printf("\n");
    }

    const LoadReport rep = runLoadedGraph(lg, tt, opt.minSeconds);

    entry.task           = lg.task;
    entry.file           = path.filename().string();
    entry.nodes          = g.nodeCount();
    entry.edges          = g.edgeCount();
    entry.correctRows    = rep.correctRows;
    entry.totalRows      = rep.totalRows;
    entry.correctBits    = rep.correctBits;
    entry.totalBits      = rep.totalBits;
    entry.booleanOnly    = rep.booleanOnly;
    entry.solved         = (rep.correctRows == rep.totalRows);

    const double total = static_cast<double>(rep.totalRows * rep.repeats);
    entry.examplesPerSec = (rep.seconds > 0.0 && total > 0.0) ? total / rep.seconds : 0.0;

    std::printf("    точность по примерам:  %6.2f%%  (%zu/%zu)\n",
                100.0 * static_cast<double>(rep.correctRows) / static_cast<double>(rep.totalRows),
                rep.correctRows, rep.totalRows);
    std::printf("    точность по битам:     %6.2f%%  (%zu/%zu)\n",
                100.0 * static_cast<double>(rep.correctBits) / static_cast<double>(rep.totalBits),
                rep.correctBits, rep.totalBits);
    std::printf("    значения в {0,1}:       %s  (максимум %d)\n",
                rep.booleanOnly ? "да" : "НЕТ", rep.maxValue);
    std::printf("    скорость:              %.0f примеров/сек\n", entry.examplesPerSec);
    std::printf("    на один пример:        %.3f мкс\n",
                entry.examplesPerSec > 0.0 ? 1e6 / entry.examplesPerSec : 0.0);
    std::printf("    нейросетей в ядре:     0  (только целочисленные операции)\n\n");
    std::printf("    ВЕРДИКТ: %s\n\n",
                entry.solved ? "граф решает задачу полностью (100%)"
                             : "граф НЕ решает задачу");
    return entry.solved;
}

static int runLoadMode(const Options& opt) {
    std::printf("======================================================================\n");
    std::printf("  ГРАФОВОЕ ЯДРО (C++) — исполнение графов, обученных в Python\n");
    std::printf("======================================================================\n\n");

    std::vector<LoadEntry> entries;
    bool allSolved = true;

    for (size_t i = 0; i < opt.loadPaths.size(); ++i) {
        LoadEntry entry;
        const bool solved = loadOneGraph(std::filesystem::path(opt.loadPaths[i]),
                                         opt, i + 1, entry);
        entries.push_back(entry);
        if (!solved) allSolved = false;
    }

    std::printf("======================================================================\n");
    std::printf("  СВОДКА\n");
    std::printf("======================================================================\n");
    std::printf("    %s %s %s %s %s\n",
                padRight("задача", 10).c_str(),
                padLeft("узлов", 6).c_str(),
                padLeft("связей", 6).c_str(),
                padLeft("точность", 9).c_str(),
                padLeft("вердикт", 7).c_str());
    for (const LoadEntry& e : entries) {
        char acc[32];
        std::snprintf(acc, sizeof acc, "%.2f%%",
                      e.totalRows ? 100.0 * static_cast<double>(e.correctRows) / static_cast<double>(e.totalRows) : 0.0);
        std::printf("    %s %s %s %s %s\n",
                    padRight(e.task, 10).c_str(),
                    padLeft(std::to_string(e.nodes), 6).c_str(),
                    padLeft(std::to_string(e.edges), 6).c_str(),
                    padLeft(acc, 9).c_str(),
                    padLeft(e.solved ? "OK" : "FAIL", 7).c_str());
    }
    std::printf("\n");

    if (opt.writeJson) {
        const std::filesystem::path dir  = findResultsDir();
        const std::filesystem::path path = dir / "diff_graph_cpp.json";
        if (writeLoadResults(path, entries)) {
            std::printf("    Результаты сохранены: %s\n\n", path.string().c_str());
        } else {
            std::fprintf(stderr, "    Не удалось записать %s\n\n", path.string().c_str());
        }
    }

    std::printf("======================================================================\n");
    std::printf("  %s\n", allSolved ? "Готово: все графы решают свои задачи." : "Готово: есть нерешённые графы.");
    std::printf("======================================================================\n");
    return allSolved ? 0 : 1;
}

int main(int argc, char** argv) {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    Options opt;
    if (!parseArgs(argc, argv, opt)) return 2;
    if (opt.help) { printHelp(); return 0; }

    // Режим исполнения обученных графов: никакого умножения, только загрузка.
    if (!opt.loadPaths.empty()) return runLoadMode(opt);

    const int digits = opt.digits;

    std::printf("======================================================================\n");
    std::printf("  ГРАФОВОЕ ЯДРО (C++) — умножение %d×%d цифр\n", digits, digits);
    std::printf("======================================================================\n\n");

    int section = 1;

    // ---------- [1] Граф ----------
    const MulGraph mg = buildMulGraph(digits);
    const Graph& graph = mg.graph;

    std::printf("[%d] Структура графа\n", section++);
    std::printf("    узлов:      %zu\n", graph.nodeCount());
    std::printf("    связей:     %zu\n", graph.edgeCount());
    std::printf("    параметров: %zu   (узлы + связи)\n", graph.paramCount());
    std::printf("    входов:     %zu\n", graph.inputs.size());
    std::printf("    выходов:    %zu\n", graph.outputs.size());

    if (!graph.isValidDag()) {
        std::fprintf(stderr, "\nОШИБКА: граф не является корректным DAG\n");
        return 1;
    }
    std::printf("    DAG:        OK (все рёбра смотрят строго назад)\n\n");

    // ---------- [2] Каскад переносов ----------
    const Dataset ds = genExamples(opt.examples, digits, opt.seed);
    const CascadeInfo ci = analyzeCascade(ds);

    std::printf("[%d] Каскадный сумматор (ripple-carry)\n", section++);
    std::printf("    макс. сумма в колонке: %d   (порог «одной цифры» — 9)\n", ci.maxColumnSum);
    std::printf("    макс. перенос:         %d   (перенос — целое, не цифра)\n", ci.maxCarry);

    // Худший по сумме пример прогоняем через сам граф и сверяем с int64-эталоном.
    {
        std::vector<int32_t> values(graph.nodes.size(), 0);
        std::vector<int32_t> out(graph.outputs.size(), 0);
        executeInto(graph, ds.in(ci.worst), static_cast<size_t>(2 * digits),
                    out.data(), values.data());

        int64_t got = 0;
        int64_t mul = 1;
        for (size_t k = 0; k < out.size(); ++k) {
            got += static_cast<int64_t>(out[k]) * mul;
            mul *= 10;
        }
        const int64_t want = static_cast<int64_t>(ds.a[ci.worst]) * ds.b[ci.worst];
        std::printf("    худший пример: %d × %d = %lld  [%s]\n",
                    ds.a[ci.worst], ds.b[ci.worst],
                    static_cast<long long>(got), got == want ? "OK" : "FAIL");

        // Проверяем, что перенос за пределы 2*d цифр равен нулю.
        const bool overflowZero = values[static_cast<size_t>(mg.overflowCarry)] == 0;
        std::printf("    перенос за старшую цифру: %d  [%s]\n",
                    values[static_cast<size_t>(mg.overflowCarry)],
                    overflowZero ? "OK" : "FAIL");
    }

    // Крайний случай «все девятки»: сумма в средней колонке максимальна и
    // многократно превышает 99 — ровно тот режим, где ломалась Python-версия.
    {
        const int64_t nine = ipow10(digits) - 1;
        const int64_t want = nine * nine;

        std::vector<int32_t> in(static_cast<size_t>(2 * digits), 9);
        // Заодно проверяем публичный API из ТЗ: execute(graph, inputs) -> outputs.
        const std::vector<int32_t> out = execute(graph, in);

        int64_t got = 0;
        int64_t mul = 1;
        for (size_t k = 0; k < out.size(); ++k) {
            got += static_cast<int64_t>(out[k]) * mul;
            mul *= 10;
        }
        std::printf("    все девятки:  %lld × %lld = %lld  [%s]\n\n",
                    static_cast<long long>(nine), static_cast<long long>(nine),
                    static_cast<long long>(got), got == want ? "OK" : "FAIL");
    }

    // ---------- [3] Бенчмарк ----------
    const BenchStats st = runBenchmark(graph, ds, opt.minSeconds);

    std::printf("[%d] Бенчмарк на %zu случайных примерах\n", section++, ds.n);
    std::printf("    точность по цифрам:     %.4f%%  (%zu/%zu)\n",
                st.accuracyDigits * 100.0, st.correctDigits, st.totalDigits);
    std::printf("    полностью верных:       %.4f%%  (%zu/%zu)\n",
                st.accuracyWhole * 100.0, st.correctWhole, ds.n);
    std::printf("    время:                  %.6f сек  (батч × %zu)\n", st.seconds, st.repeats);
    std::printf("    скорость:               %.0f примеров/сек\n", st.examplesPerSec);
    std::printf("    на один пример:         %.3f мкс\n",
                st.examplesPerSec > 0.0 ? 1e6 / st.examplesPerSec : 0.0);
    std::printf("    узлов на пример:        %zu\n\n", graph.nodeCount());

    // ---------- [4] Сравнение ----------
    if (st.examplesPerSec > 0.0) {
        const double baselineTasksPerSec = 0.07;   // 3B-трансформер на CPU (из baseline.py)
        std::printf("[%d] Сравнение с плотной моделью\n", section++);
        std::printf("    граф:            %zu параметров, %.2f млн примеров/сек\n",
                    graph.paramCount(), st.examplesPerSec / 1e6);
        std::printf("    3B трансформер:  ~0.07 задач/сек на CPU (GSM8K, 74%%)\n");
        std::printf("    отношение:       ~%.2e×  (задачи разные — сравнение условное)\n",
                    st.examplesPerSec / baselineTasksPerSec);
        std::printf("    параметров:      %zu vs 3e9 → в %.2e× меньше\n\n",
                    graph.paramCount(),
                    3.0e9 / static_cast<double>(graph.paramCount()));
    }

    // ---------- [5] Демо эволюции ----------
    if (opt.doEvolve) {
        std::printf("[%d] Демо-эволюция (мутации + скрещивание)\n", section++);
        const Dataset train = genExamples(std::min<size_t>(ds.n, 200), digits, opt.seed + 1);

        const Graph start = opt.fromRandom
            ? buildRandomGraph(digits, 200, opt.seed + 11)
            : graph;
        const double startFit = fitness(start, train);

        std::printf("    старт:      %s, узлов=%zu\n",
                    opt.fromRandom ? "случайный граф" : "структурный граф",
                    start.nodeCount());
        std::printf("    популяция=%d  поколений=%d  примеров=%zu\n",
                    opt.popSize, opt.generations, train.n);
        std::printf("    fitness до эволюции: %.4f\n", startFit);

        const Graph evolved = evolve(start, train, opt.popSize, opt.generations,
                                     0.10, opt.seed + 7, true);
        const BenchStats est = runBenchmark(evolved, ds, 0.05);

        std::printf("    fitness после:       %.4f\n", fitness(evolved, train));
        std::printf("    на свежих примерах:  %.4f%%  узлов=%zu\n",
                    est.accuracyDigits * 100.0, evolved.nodeCount());
        if (!opt.fromRandom) {
            std::printf("    (старт уже идеален — мутации его только портят, это ожидаемо)\n");
        }
        std::printf("\n");
    }

    // ---------- [6] JSON ----------
    if (opt.writeJson) {
        const std::filesystem::path dir  = findResultsDir();
        const std::filesystem::path path = dir / "graph_cpp_results.json";

        std::vector<int32_t> values(graph.nodes.size(), 0);
        std::vector<int32_t> out(graph.outputs.size(), 0);
        executeInto(graph, ds.in(ci.worst), static_cast<size_t>(2 * digits),
                    out.data(), values.data());
        const bool overflowZero = values[static_cast<size_t>(mg.overflowCarry)] == 0;

        if (writeJson(path, digits, ds.n, opt.seed, graph, st, ci, overflowZero)) {
            std::printf("[%d] Результаты сохранены: %s\n\n", section++, path.string().c_str());
        } else {
            std::fprintf(stderr, "[%d] Не удалось записать %s\n\n",
                         section++, path.string().c_str());
        }
    }

    std::printf("======================================================================\n");
    std::printf("  Готово.\n");
    std::printf("======================================================================\n");
    return 0;
}
