#ifndef TEST_CAST_ANGLEMETER_H
#define TEST_CAST_ANGLEMETER_H
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#define M_PI 3.14159265358979323846

// ------------------------------------
// Простейшие структуры координат
// ------------------------------------
struct pos_t {
    uint16_t x;
    uint16_t y;
};

// ------------------------------------
// Координата в виде float для точной геометрии
// ------------------------------------
struct posf_t {
    float x;
    float y;
};

// ------------------------------------
// Удобный переход из целочисленных координат в вещественные
// ------------------------------------
inline posf_t pos2f(const pos_t p) {
    return posf_t{ static_cast<float>(p.x), static_cast<float>(p.y) };
}

// ------------------------------------
// Результаты поиска перепадов яркости по линии сканирования
// ------------------------------------
struct scan_t {
    pos_t posDifMin;
    pos_t posDifMax;
};

// ------------------------------------
// Цвет пикселя в формате RGB
// ------------------------------------
struct rgb_t {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

// ------------------------------------
// Вспомогательные структуры для локального анализа
// ------------------------------------
namespace {
    struct EdgeY {
        int y;
        int dBr;
    };

    struct PairY {
        int yMin;
        int yMax;
        float avgUp;
        float avgDown;
        float avgBright;
    };

    struct EdgeX {
        int x;
        int dBr;
    };

    struct PairX {
        int xMin;
        int xMax;
        float avgLeft;
        float avgRight;
        float avgBright;
    };
}

// ------------------------------------
// Основное состояние алгоритма измерения угла
// ------------------------------------
struct anglemeter_t {
    int img_width;
    int img_height;

    int bright_lim;
    int max_pairs;

    std::vector<scan_t> x_scans;
    std::vector<scan_t> y_scans;

    std::vector<posf_t> points_1;
    std::vector<posf_t> points_2;

    float last_angle_deg;
};

// ------------------------------------
// Проверка валидности результата сканирования
// ------------------------------------
inline bool isValidScan(const scan_t* scan) {
    return scan->posDifMin.x != 0ui16;
}

// ------------------------------------
// Сброс результата сканирования
// ------------------------------------
inline void invalidateScan(scan_t* scan) {
    scan->posDifMin.x = 0ui16;
}

// ------------------------------------
// Создание и инициализация структуры измерителя угла
// ------------------------------------
void anglemeterCreate(anglemeter_t** am_ptr) {
    auto *am = new anglemeter_t();
    am->img_width = 0;
    am->img_height = 0;
    am->bright_lim = 70;
    am->max_pairs = 4;
    am->last_angle_deg = -1.0f;
    *am_ptr = am;
}

// ------------------------------------
// Корректное освобождение ресурсов
// ------------------------------------
void anglemeterDestroy(anglemeter_t* am) {
    delete am;
}

// ------------------------------------
// Настройка размеров входного изображения
// ------------------------------------
void anglemeterSetImageSize(anglemeter_t* am, int width, int height) {
    am->img_width = width;
    am->img_height = height;
    am->max_pairs = width / 160;
    am->x_scans.resize(static_cast<size_t>(width));
    am->y_scans.resize(static_cast<size_t>(height));
    am->points_1.reserve(static_cast<size_t>(std::max(width, height)));
    am->points_2.reserve(static_cast<size_t>(std::max(width, height)));
}

// ------------------------------------
// Полный сброс внутренних буферов перед новым кадром
// ------------------------------------
void anglemeterRestoreState(anglemeter_t* am) {
    const auto w = static_cast<size_t>(am->img_width);
    const auto h = static_cast<size_t>(am->img_height);
    for (size_t i = 0; i < w; ++i) {
        invalidateScan(&am->x_scans[i]);
    }
    for (size_t i = 0; i < h; ++i) {
        invalidateScan(&am->y_scans[i]);
    }
    am->points_1.clear();
    am->points_2.clear();
}


// ------------------------------------
// Быстрый расчёт яркости пикселя в приблизительном сером
// ------------------------------------
inline int brightness(const uint8_t r, const uint8_t g, const uint8_t b) {
    return ((r + g + b) * 21845) >> 16;
}

// ------------------------------------
// Сканирование столбца изображения и поиск перепадов яркости
// ------------------------------------
inline void scanCol(anglemeter_t* am, const rgb_t* img, const int x, const int y_from, const int y_to)
{
    const int height = am->img_height;
    const int width = am->img_width;
    const int bright_lim = am->bright_lim;
    const int max_pairs  = am->max_pairs;
    scan_t* scan = &am->x_scans[x];

    std::vector<EdgeY> mins, maxs;
    mins.reserve(static_cast<size_t>(max_pairs));
    maxs.reserve(static_cast<size_t>(max_pairs));

    // 1. Поиск локальных минимумов/максимумов
    for (int y = y_from; y < y_to; ++y) {
        const rgb_t& p1 = img[(static_cast<size_t>(y) - 2u) * width + x];
        const rgb_t& p2 = img[static_cast<size_t>(y) * width + x];

        const int br1 = brightness(p1.r, p1.g, p1.b);
        const int br2 = brightness(p2.r, p2.g, p2.b);
        const int dBr = br2 - br1;
        const int yEdge = y - 1;

        if (dBr < -static_cast<int>(bright_lim)) {
            if (mins.size() < static_cast<size_t>(max_pairs))
                mins.push_back({ yEdge, dBr });
            else if (dBr < mins.back().dBr)
                mins.back() = EdgeY{ yEdge, dBr };
        }

        if (dBr > static_cast<int>(bright_lim)) {
            if (maxs.size() < static_cast<size_t>(max_pairs))
                maxs.push_back({ yEdge, dBr });
            else if (dBr > maxs.back().dBr)
                maxs.back() = EdgeY{ yEdge, dBr };
        }
    }

    if (mins.empty() || maxs.empty()) {
        invalidateScan(scan);
        return;
    }

    // 2. Сортировка: mins по возрастанию (более глубокие минимумы в начале), maxs по убыванию
    std::sort(mins.begin(), mins.end(),
              [](const EdgeY& a, const EdgeY& b){ return a.dBr < b.dBr; });
    std::sort(maxs.begin(), maxs.end(),
              [](const EdgeY& a, const EdgeY& b){ return a.dBr > b.dBr; });

    // 3. Подбор candidate-пар
    auto avgRegion = [&](const int startY, const int endY) {
        if (startY < 0 || endY >= height)
            return 0.0f;
        int sum = 0;
        int cnt = 0;

        for (int y = startY; y <= endY; ++y) {
            const rgb_t& p = img[static_cast<size_t>(y) * width + x];
            sum += brightness(p.r, p.g, p.b);
            cnt++;
        }

        return cnt > 0 ? static_cast<float>(sum) / static_cast<float>(cnt) : 0.0f;
    };

    std::vector<PairY> candidates;

    for (const auto& minE : mins) {
        for (const auto& maxE : maxs) {
            int dy = maxE.y - minE.y;
            if (dy > 10 && dy < 50) {

                int sum = 0, cnt = 0;
                for (int y = minE.y; y <= maxE.y; ++y) {
                    const rgb_t& p = img[static_cast<size_t>(y) * width + x];
                    sum += brightness(p.r, p.g, p.b);
                    cnt++;
                }

                const float avgBright = cnt > 0 ? static_cast<float>(sum) / static_cast<float>(cnt) : 0.0f;
                if (static_cast<int>(avgBright) >= bright_lim)
                    continue;

                const float avgUp = avgRegion(minE.y - 16, minE.y - 1);
                const float avgDown = avgRegion(maxE.y + 1, maxE.y + 16);

                if (avgUp < 200.0f || avgDown < 200.0f)
                    continue;

                candidates.push_back({
                    minE.y,
                    maxE.y,
                    avgUp,
                    avgDown,
                    avgBright
                });
            }
        }
    }

    if (candidates.empty()) {
        invalidateScan(scan);
        return;
    }

    // 4. Выбор лучшей пары
    const auto best = std::max_element(
        candidates.begin(), candidates.end(),
        [](const PairY& a, const PairY& b) {
            return (a.avgUp + a.avgDown) < (b.avgUp + b.avgDown);
        }
    );

    scan->posDifMin = { static_cast<uint16_t>(x), static_cast<uint16_t>(best->yMin) };
    scan->posDifMax = { static_cast<uint16_t>(x), static_cast<uint16_t>(best->yMax) };
}

// ------------------------------------
// Сканирование строки изображения и поиск перепадов яркости
// ------------------------------------
inline void scanRow(anglemeter_t* am, const rgb_t* img, const int y, const int x_from, const int x_to)
{
    const int height = am->img_height;
    const int width  = am->img_width;
    const int bright_lim = am->bright_lim;
    const int max_pairs  = am->max_pairs;
    scan_t* scan = &am->y_scans[y];

    std::vector<EdgeX> mins, maxs;
    mins.reserve(static_cast<size_t>(max_pairs));
    maxs.reserve(static_cast<size_t>(max_pairs));

    // 1. Поиск локальных минимумов/максимумов вдоль строки
    for (int x = x_from; x < x_to; ++x) {
        const rgb_t& p1 = img[y * width + (static_cast<size_t>(x) - 2u)];
        const rgb_t& p2 = img[y * width + static_cast<size_t>(x)];

        const int br1 = brightness(p1.r, p1.g, p1.b);
        const int br2 = brightness(p2.r, p2.g, p2.b);
        const int dBr = br2 - br1;
        const int xEdge = x - 1;

        if (dBr < -bright_lim) {
            if (mins.size() < static_cast<size_t>(max_pairs))
                mins.push_back({ xEdge, dBr });
            else if (dBr < mins.back().dBr)
                mins.back() = EdgeX{ xEdge, dBr };
        }

        if (dBr > bright_lim) {
            if (maxs.size() < static_cast<size_t>(max_pairs))
                maxs.push_back({ xEdge, dBr });
            else if (dBr > maxs.back().dBr)
                maxs.back() = EdgeX{ xEdge, dBr };
        }
    }

    if (mins.empty() || maxs.empty()) {
        invalidateScan(scan);
        return;
    }

    // 2. Сортировка: mins по возрастанию dBr, maxs по убыванию dBr
    std::sort(mins.begin(), mins.end(),
              [](const EdgeX& a, const EdgeX& b){ return a.dBr < b.dBr; });
    std::sort(maxs.begin(), maxs.end(),
              [](const EdgeX& a, const EdgeX& b){ return a.dBr > b.dBr; });

    // 3. Подбор candidate-пар
    auto avgRegion = [&](const int startX, const int endX) {
        if (startX < 0 || endX >= width)
            return 0.0f;

        int sum = 0;
        int cnt = 0;
        for (int x = startX; x <= endX; ++x) {
            const rgb_t& p = img[y * width + static_cast<size_t>(x)];
            sum += brightness(p.r, p.g, p.b);
            cnt++;
        }
        return cnt > 0 ? static_cast<float>(sum) / static_cast<float>(cnt) : 0.0f;
    };

    std::vector<PairX> candidates;

    for (const auto& minE : mins) {
        for (const auto& maxE : maxs) {

            int dx = maxE.x - minE.x;
            if (dx > 10 && dx < 50) {

                int sum = 0, cnt = 0;
                for (int x = minE.x; x <= maxE.x; ++x) {
                    const rgb_t& p = img[y * width + static_cast<size_t>(x)];
                    sum += brightness(p.r, p.g, p.b);
                    cnt++;
                }

                const float avgBright = cnt > 0 ? static_cast<float>(sum) / static_cast<float>(cnt) : 0.0f;
                if (static_cast<int>(avgBright) >= bright_lim)
                    continue;

                const float avgLeft  = avgRegion(minE.x - 16, minE.x - 1);
                const float avgRight = avgRegion(maxE.x + 1, maxE.x + 16);

                if (avgLeft < 200.0f || avgRight < 200.0f)
                    continue;

                candidates.push_back({
                    minE.x,
                    maxE.x,
                    avgLeft,
                    avgRight,
                    avgBright
                });
            }
        }
    }

    if (candidates.empty()) {
        invalidateScan(scan);
        return;
    }

    // 4. Выбор лучшей пары
    const auto best = std::max_element(
        candidates.begin(), candidates.end(),
        [](const PairX& a, const PairX& b) {
            return (a.avgLeft + a.avgRight) < (b.avgLeft + b.avgRight);
        }
    );

    scan->posDifMin = { static_cast<uint16_t>(best->xMin), static_cast<uint16_t>(y) };
    scan->posDifMax = { static_cast<uint16_t>(best->xMax), static_cast<uint16_t>(y) };
}

// ------------------------------------
// Поиск границы при движении вдоль столбцов в заданном направлении
// ------------------------------------
inline bool scanColsDirectional(
    anglemeter_t* am, const rgb_t* img, int* count,
    int x_start, int x_stop, int x_step,
    int y_from_init, int y_to_init
) {
    const int width  = am->img_width;
    const int height = am->img_height;
    scan_t* scans = am->x_scans.data();

    if (x_step != -1 && x_step != 1)
        return false;

    if (x_start < 0 || x_start >= width)
        return false;

    if (x_stop < 0) x_stop = 0;
    if (x_stop >= width) x_stop = width - 1;

    *count = 0;

    // --- 1. Стартовое сканирование ---
    scanCol(am, img, x_start, y_from_init, y_to_init);
    if (!isValidScan(&scans[x_start]))
        return false;

    uint16_t last_y_from = scans[x_start].posDifMin.y - 8;
    uint16_t last_y_to   = scans[x_start].posDifMax.y + 8;

    // --- Аккумуляторы статистики ---
    double sumX = 0;
    double sumY = 0;
    double sumXY = 0;
    double sumXX = 0;
    double sumY2 = 0;
    int n = 0;

    // Добавляем первую точку
    {
        const int x = x_start;
        const int y = last_y_from;
        sumX  += x;
        sumY  += y;
        sumXX += x * x;
        sumXY += x * y;
        sumY2 += y * y;
        n++;
    }

    int stableFailCount = 0;

    // --- 2. Итерация по направлению x_step ---
    for (int x = x_start + x_step;
         (x_step < 0 ? x >= x_stop : x <= x_stop);
         x += x_step)
    {
        uint16_t y_from = last_y_from > 11 ? last_y_from - 8 : 2;
        uint16_t y_to   =  last_y_to < height - 11 ? last_y_to + 8 : height - 2;

        if (y_to >= height) y_to = height - 1;

        scanCol(am, img, x, y_from, y_to);

        if (!isValidScan(&scans[x])) {
            stableFailCount++;

            if (stableFailCount >= 6) {

                // ≥ 20 точек
                if (n < 20)
                    return false;

                // Линейная регрессия: y = a*x + b
                double denom = n * sumXX - sumX * sumX;
                if (denom == 0)
                    return false;

                double a = (n * sumXY - sumX * sumY) / denom;
                double b = (sumY - a * sumX) / n;

                // MSE
                double mse =
                    ( sumY2
                    + a * a * sumXX
                    + n * b * b
                    + 2 * a * b * sumX
                    - 2 * a * sumXY
                    - 2 * b * sumY ) / n;

                if (mse > 4.0)
                    return false;

                return true;
            }

            continue;
        }

        // Валидное сканирование
        stableFailCount = 0;

        last_y_from = scans[x].posDifMin.y;
        if (last_y_from < 2) last_y_from = 2;
        last_y_to   = scans[x].posDifMax.y;
        if (last_y_to >= height) last_y_to = height - 2;

        // Аккумуляция статистики
        int y = last_y_from;
        int xx = x;

        sumX  += xx;
        sumY  += y;
        sumXX += xx * xx;
        sumXY += xx * y;
        sumY2 += y * y;
        n++;
    }

    // Дошли до конца без 6 фейлов подряд
    *count = n;
    return true;
}


// ------------------------------------
// Поиск границы при движении вдоль строк в заданном направлении
// ------------------------------------
inline bool scanRowsDirectional(
    anglemeter_t* am, const rgb_t* img, int* count,
    int y_start, int y_stop, int y_step,
    int x_from_init, int x_to_init
) {

    const int width  = am->img_width;
    const int height = am->img_height;
    scan_t* scans = am->y_scans.data();

    if (y_step != -1 && y_step != 1)
        return false;

    if (y_start < 0 || y_start >= height)
        return false;

    if (y_stop < 0) y_stop = 0;
    if (y_stop >= height) y_stop = height - 1;

    *count = 0;

    // --- 1. Стартовое сканирование ---
    scanRow(am, img, y_start, x_from_init, x_to_init);
    if (!isValidScan(&scans[y_start]))
        return false;

    int last_x_from = scans[y_start].posDifMin.x - 8;
    if (last_x_from < 2) last_x_from = 2;
    int last_x_to   = scans[y_start].posDifMax.x + 8;
    if (last_x_to >= width) last_x_to = width - 2;

    // --- Аккумуляторы статистики ---
    double sumY = 0;
    double sumX = 0;
    double sumXY = 0;
    double sumYY = 0;
    double sumX2 = 0;
    int n = 0;

    // Добавляем первую точку
    {
        const int y = y_start;
        const int x = last_x_from;
        sumY  += y;
        sumX  += x;
        sumYY += y * y;
        sumXY += y * x;
        sumX2 += x * x;
        n++;
    }

    int stableFailCount = 0;

    // --- 2. Итерация по направлению y_step ---
    for (int y = y_start + y_step;
         (y_step < 0 ? y >= y_stop : y <= y_stop);
         y += y_step)
    {
        int x_from = last_x_from > 11 ? last_x_from - 8 : 2;
        int x_to   = last_x_to < width - 11 ? last_x_to + 8 : width - 2;

        scanRow(am, img, y, x_from, x_to);

        if (!isValidScan(&scans[y])) {
            stableFailCount++;

            if (stableFailCount >= 6) {

                // А) Должно быть ≥ 20 валидных точек
                if (n < 20)
                    return false;

                // Б) Линейная регрессия по накопленным суммам
                double denom = n * sumYY - sumY * sumY;
                if (denom == 0)
                    return false;

                double a = (n * sumXY - sumY * sumX) / denom;
                double b = (sumX - a * sumY) / n;

                // Вычисление MSE без хранения точек
                double mse =
                    ( sumX2
                    + a * a * sumYY
                    + n * b * b
                    + 2 * a * b * sumY
                    - 2 * a * sumXY
                    - 2 * b * sumX ) / n;

                if (mse > 4.0)
                    return false;

                return true;
            }

            continue;
        }

        // --- Валидное сканирование ---
        stableFailCount = 0;

        last_x_from = scans[y].posDifMin.x;
        last_x_to   = scans[y].posDifMax.x;

        // Аккумуляция статистики
        int x = last_x_from;
        int yy = y;

        sumY  += yy;
        sumX  += x;
        sumYY += yy * yy;
        sumXY += yy * x;
        sumX2 += x * x;
        n++;
    }

    // Если дошли до конца без >=6 фейлов
    *count = n;
    return true;
}


// ------------------------------------
// Попытка найти границы через вертикальные сканы
// ------------------------------------
inline bool scanCols(anglemeter_t* am, const rgb_t* img, int* dir)
{
    const int width  = am->img_width;
    const int height = am->img_height;
    scan_t* scans = am->x_scans.data();

    const int center_left  = width / 4;
    const int center_right = (3 * width) / 4;

    auto findStart = [&](int x, uint16_t& ys, uint16_t& ye)->bool {
        if (x < 2 || x >= width - 2)
            return false;

        scanCol(am, img, x, 2, height - 2);
        if (!isValidScan(&scans[x]))
            return false;

        ys = scans[x].posDifMin.y - 8;
        if (ys < 2) ys = 2;
        ye = scans[x].posDifMax.y + 8;
        if (ye >= height) ye = height - 2;
        return true;
    };

    auto tryHalf = [&](int x_start, int x_min, int x_max, int* count)->bool {
        uint16_t ys, ye;
        if (!findStart(x_start, ys, ye))
            return false;

        if (x_start > x_min &&
            !scanColsDirectional(am, img, count, x_start, x_min, -1, ys, ye))
            return false;

        if (x_start < x_max &&
            !scanColsDirectional(am, img, count, x_start, x_max, +1, ys, ye))
            return false;

        return true;
    };

    constexpr int offsets[] = {0, -10, +10};

    for (const int dx : offsets) {

        const int x_left  = center_left  + dx;
        const int x_right = center_right + dx;

        int left_cnt = 0, right_cnt = 0;

        const bool left  = tryHalf(x_left,  0,       width/2,   &left_cnt);
        const bool right = tryHalf(x_right, width/2, width - 1, &right_cnt);

        if (left && right) {
            *dir = (left_cnt >= right_cnt) ? -1 : +1;
            return true;
        }
        if (left)  { *dir = -1; return true; }
        if (right) { *dir = +1; return true; }
    }

    return false;
}



// ------------------------------------
// Попытка найти границы через горизонтальные сканы
// ------------------------------------
inline bool scanRows(anglemeter_t* am, const rgb_t* img, int* dir)
{
    const int width  = am->img_width;
    const int height = am->img_height;
    scan_t* scans = am->y_scans.data();

    const int center_top = height / 4;
    const int center_bot = (3 * height) / 4;

    auto findStart = [&](int y, uint16_t& xs, uint16_t& xe)->bool {
        if (y < 2 || y >= height - 2) return false;

        scanRow(am, img, y, 2, width - 2);
        if (!isValidScan(&scans[y])) return false;

        xs = scans[y].posDifMin.x - 8;
        if (xs < 2) xs = 2;
        xe = scans[y].posDifMax.x + 8;
        if (xe >= width) xe = width - 2;
        return true;
    };

    auto tryHalf = [&](int y_start, int y_min, int y_max, int* count)->bool {
        uint16_t xs, xe;
        if (!findStart(y_start, xs, xe))
            return false;

        if (y_start > y_min &&
            !scanRowsDirectional(am, img, count, y_start, y_min, -1, xs, xe))
            return false;

        if (y_start < y_max &&
            !scanRowsDirectional(am, img, count, y_start, y_max, +1, xs, xe))
            return false;

        return true;
    };

    constexpr int offsets[] = {0, -10, +10};

    for (const int dy : offsets) {

        const int y_top = center_top + dy;
        const int y_bot = center_bot + dy;

        int up_cnt = 0, down_cnt = 0;
        const bool up   = tryHalf(y_top, 0, height/2, &up_cnt);
        const bool down = tryHalf(y_bot, height/2, height-1, &down_cnt);

        if (up && down) {
            *dir = (up_cnt >= down_cnt) ? -1 : +1;
            return true;
        }
        if (up)   { *dir = -1; return true; }
        if (down) { *dir = +1; return true; }
    }

    return false;
}

// dir: 1 - left, 2 - right, 3 - up, 4 - down
// ------------------------------------
// Главный поиск ориентира: выбираем стратегию по последнему углу
// ------------------------------------
inline bool scan(anglemeter_t* am, const rgb_t* img, int* dir) {
    bool c = false, r = false;
    const float a = am->last_angle_deg;

    const bool rows_first =
        (a <= 45.0f) ||
        (a >= 315.0f) ||
        ((a >= 135.0f) && (a <= 225.0f));

    if (rows_first) {
        r = scanRows(am, img, dir);
        if (!r) {
            c = scanCols(am, img, dir);
            (*dir == -1) ? (*dir = 1) : (*dir = 2);     // left/right
        } else {
            (*dir == -1) ? (*dir = 3) : (*dir = 4);     // up/down
        }
    } else {
        c = scanCols(am, img, dir);
        if (!c) {
            r = scanRows(am, img, dir);
            (*dir == -1) ? (*dir = 3) : (*dir = 4);     // up/down
        } else {
            (*dir == -1) ? (*dir = 1) : (*dir = 2);     // left/right
        }
    }

    return c || r;
}


// ------------------------------------
// Формирование набора опорных точек в выбранной половине кадра
// ------------------------------------
inline void selectPoints(anglemeter_t* am, int dir) {
    if (dir == 1) {
        for (int x =  am->img_width / 2 - 1; x > 0; --x) {
            auto* scan = &am->x_scans[x];
            if (isValidScan(scan)) {
                am->points_1.push_back(pos2f(scan->posDifMin));
                am->points_2.push_back(pos2f(scan->posDifMax));
            }
        }
    }
    else if (dir == 2) {
        for (int x = am->img_width / 2 + 1; x < am->img_width; ++x) {
            auto* scan = &am->x_scans[x];
            if (isValidScan(scan)) {
                am->points_1.push_back(pos2f(scan->posDifMin));
                am->points_2.push_back(pos2f(scan->posDifMax));
            }
        }
    }
    else if (dir == 3) {
        for (int y = am->img_height / 2 - 1; y > 0; --y) {
            auto* scan = &am->y_scans[y];
            if (isValidScan(scan)) {
                am->points_1.push_back(pos2f(scan->posDifMin));
                am->points_2.push_back(pos2f(scan->posDifMax));
            }
        }
    }
    else if (dir == 4) {
        for (int y = am->img_height / 2 + 1; y < am->img_height; ++y) {
            auto* scan = &am->y_scans[y];
            if (isValidScan(scan)) {
                am->points_1.push_back(pos2f(scan->posDifMin));
                am->points_2.push_back(pos2f(scan->posDifMax));
            }
        }
    }
}


// ------------------------------------
// Подсчёт количества точек, лежащих в окрестности прямой
// ------------------------------------
inline uint16_t cntOnLine(
    const posf_t* points,
    const int cnt,
    float nx,
    float ny,
    const posf_t ref,
    const float eps
)
{
    const float len = std::sqrt(nx*nx + ny*ny);
    nx /= len;
    ny /= len;

    int count = 0;

    for (int i = 0; i < cnt; i++) {
        const float vx = points[i].x - ref.x;
        const float vy = points[i].y - ref.y;
        const float d = vx*nx + vy*ny;
        if (std::fabs(d) < eps)
            count++;
    }
    return count;
}

// ------------------------------------
// Оценка прямой через PROSAC/RANSAC с финальной PCA-аппроксимацией
// ------------------------------------
inline bool fitLineRANSAC(
    const posf_t* pts,
    const int cnt,
    float* out_nx,
    float* out_ny,
    posf_t* out_ref,
    const float eps,
    const int iterations,
    const float min_inliers_ratio
)
{
    if (cnt < 2) return false;

    const int min_inliers = int(std::ceil(min_inliers_ratio * cnt));

    // ------------------------------------
    // PRECOMPUTE: flat arrays + scores
    // ------------------------------------
    std::vector<float> xs(cnt), ys(cnt), score(cnt);
    for (int i = 0; i < cnt; ++i) {
        xs[i] = pts[i].x;
        ys[i] = pts[i].y;
        // Score для PROSAC: расстояние от глобального центра
        // (или любой другой информативный ранк)
    }

    // Глобальный mean — для score
    float mx = 0, my = 0;
    for (int i = 0; i < cnt; ++i) { mx += xs[i]; my += ys[i]; }
    mx /= cnt; my /= cnt;

    for (int i = 0; i < cnt; ++i) {
        float dx = xs[i] - mx;
        float dy = ys[i] - my;
        score[i] = std::sqrt(dx*dx + dy*dy);
    }

    // ------------------------------------
    // PROSAC: сортируем точки по score
    // «более информативные» точки — первые
    // ------------------------------------
    std::vector<int> order(cnt);
    for (int i = 0; i < cnt; ++i) order[i] = i;

    std::sort(order.begin(), order.end(),
        [&](int a, int b){ return score[a] > score[b]; });

    // ------------------------------------
    // RANSAC с ускорением:
    // (1) быстрый подсчёт inliers,
    // (2) Haar-like reject,
    // (3) PROSAC progressive sampling
    // ------------------------------------

    const float eps_abs = std::fabs(eps);

    std::mt19937 rng(1234567);
    std::uniform_real_distribution<float> uf(0.f, 1.f);
    std::uniform_int_distribution<int> dist(0, cnt - 1);

    int best_inliers = -1;
    int best_i1 = -1, best_i2 = -1;

    // PROSAC: увеличивающееся окно top-K
    int K = 2;
    const int Kmax = cnt;

    for (int it = 0; it < iterations; ++it) {

        // PROSAC: растим K со временем
        if (K < Kmax) {
            float p = float(it) / iterations;
            K = int(2 + p * (Kmax - 2)); // линейное увеличение
            if (K > Kmax) K = Kmax;
        }

        // выбор из top-K
        int i1 = order[int(uf(rng) * K)];
        int i2 = order[int(uf(rng) * K)];
        if (i1 == i2) continue;

        float x1 = xs[i1], y1 = ys[i1];
        float x2 = xs[i2], y2 = ys[i2];

        float dx = x2 - x1;
        float dy = y2 - y1;
        float L2 = dx*dx + dy*dy;
        if (L2 < 1e-12f) continue;

        // нормаль
        float nx = -dy;
        float ny = dx;
        float inv = 1.f / std::sqrt(nx*nx + ny*ny);
        nx *= inv; ny *= inv;

        // ------------------------------------
        // HAAR-like reject:
        // проверяем sign-дот на 8 случайных точках
        // если почти все по одну сторону → плохая гипотеза
        // ------------------------------------
        {
            int bad = 0;
            for (int t = 0; t < 8; ++t) {
                int j = dist(rng) % cnt;
                float d = nx * (xs[j] - x1) + ny * (ys[j] - y1);
                if (d > eps_abs) ++bad;
                if (d < -eps_abs) --bad;
            }
            if (std::abs(bad) > 6) continue; // reject
        }

        // ------------------------------------
        // Быстрый подсчёт inliers
        // ------------------------------------
        int c = 0;

        // минимальная тяжёлая форма, но быстрее обычной
        for (int i = 0; i < cnt; ++i) {
            float d = nx * (xs[i] - x1) + ny * (ys[i] - y1);
            c += (std::fabs(d) <= eps_abs);
        }

        if (c > best_inliers) {
            best_inliers = c;
            best_i1 = i1;
            best_i2 = i2;
        }

        if (best_inliers == cnt) break;
    }

    if (best_inliers < min_inliers)
        return false;

    // ------------------------------------
    // Финальное уточнение (PCA)
    // ------------------------------------
    float x1 = xs[best_i1], y1 = ys[best_i1];
    float x2 = xs[best_i2], y2 = ys[best_i2];

    float dx = x2 - x1, dy = y2 - y1;
    float nx = -dy, ny = dx;
    float inv = 1.f / std::sqrt(nx*nx + ny*ny);
    nx *= inv; ny *= inv;

    // сбор inliers
    std::vector<int> idx;
    idx.reserve(best_inliers);
    for (int i = 0; i < cnt; ++i) {
        float d = nx * (xs[i] - x1) + ny * (ys[i] - y1);
        if (std::fabs(d) <= eps_abs)
            idx.push_back(i);
    }

    const int m = idx.size();
    float mx2 = 0, my2 = 0;
    for (int i : idx) { mx2 += xs[i]; my2 += ys[i]; }
    mx2 /= m; my2 /= m;

    float Sxx=0, Sxy=0, Syy=0;
    for (int i : idx) {
        float ux = xs[i] - mx2;
        float uy = ys[i] - my2;
        Sxx += ux*ux;
        Sxy += ux*uy;
        Syy += uy*uy;
    }

    float theta = 0.5f * std::atan2(2*Sxy, Sxx - Syy);
    float dirx = std::cos(theta), diry = std::sin(theta);

    *out_nx = -diry;
    *out_ny =  dirx;
    out_ref->x = mx2;
    out_ref->y = my2;

    return true;
}




// ------------------------------------
// Приведение нормали к нужному направлению и расчёт угла в градусах
// ------------------------------------
inline float angleOfLine(anglemeter_t* am, float nx, float ny, int dir)
{
    // Тангенциальный вектор (направление линии)
    float tx =  ny;
    float ty = -nx;

    // Фиксация направления по dir
    switch (dir)
    {
        case 1: // left (-X)
            if (tx > 0) { tx = -tx; ty = -ty; }
            break;

        case 2: // right (+X)
            if (tx < 0) { tx = -tx; ty = -ty; }
            break;

        case 3: // up (-Y)
            if (ty > 0) { tx = -tx; ty = -ty; }
            break;

        case 4: // down (+Y)
            if (ty < 0) { tx = -tx; ty = -ty; }
            break;
    }

    // Теперь atan2 всегда даст корректный угол
    float angle = std::atan2(-tx, ty);
    angle = angle * 180.0f / M_PI;

    if (angle < 0.0f)
        angle += 360.0f;

    am->last_angle_deg = angle;
    return angle;
}



#endif //TEST_CAST_ANGLEMETER_H