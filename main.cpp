#include <opencv2/opencv.hpp>
#include "cast_anglemeter.h"
#include <iostream>
#include <fstream>

using namespace cv;

// ------------------------------------
// Контейнер результата подбора прямой
// ------------------------------------
struct LineFitResult {
    bool ok;
    float nx, ny;
    posf_t ref;
    float angle;
};

// ------------------------------------
// Унифицированная функция для RANSAC + отрисовки
// ------------------------------------
static LineFitResult runRansacDraw(
    anglemeter_t* am,
    Mat& frame,
    const std::vector<posf_t>& pts,
    int dir,
    const Scalar& color,
    const std::string& label,
    const Point& text_pos)
{
    LineFitResult r{false,0,0,{0,0},0};

    if (pts.empty())
        return r;

    float nx=0, ny=0;
    posf_t ref{0,0};

    bool ok = fitLineRANSAC(
        pts.data(), pts.size(),
        &nx, &ny, &ref,
        1.5f, 5000, 0.9f
    );

    if (!ok)
        return r;

    r.ok = true;
    r.nx = nx;
    r.ny = ny;
    r.ref = ref;
    r.angle = angleOfLine(am, nx, ny, dir);

    // Построение линии
    float tx =  ny;
    float ty = -nx;
    float t  = 2000.0f;

    Point2f p1(ref.x - t*tx, ref.y - t*ty);
    Point2f p2(ref.x + t*tx, ref.y + t*ty);

    line(frame, p1, p2, color, 2);
    circle(frame, Point((int)ref.x, (int)ref.y), 4, color, FILLED);

    // Подпись угла
    putText(
        frame,
        label + ": " + std::to_string(r.angle),
        text_pos,
        FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2
    );

    return r;
}

// ------------------------------------
// Точка входа: обработка видео и логирование углов
// ------------------------------------
int main() {
    VideoCapture cap("video.avi");
    if (!cap.isOpened())
        return -1;

    // cap.set(CAP_PROP_POS_FRAMES, 3000);

    anglemeter_t* am;
    anglemeterCreate(&am);
    anglemeterSetImageSize(am, 640, 480);

    // ------------------------------------
    // Создаём csv, перезаписывая старый
    // ------------------------------------
    std::ofstream csv("angles.csv", std::ios::trunc);
    csv << "frame,angle1,angle2,compute_ms\n";

    Mat frame;
    TickMeter t_all;

    int frameIndex = 0;

    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        Mat rgb;
        cvtColor(frame, rgb, COLOR_BGR2RGB);
        const rgb_t* img_rgb = reinterpret_cast<const rgb_t*>(rgb.data);

        // ------------------------------------
        // Засекаем ВСЁ вычисление углов
        // ------------------------------------
        t_all.reset();
        t_all.start();

        anglemeterRestoreState(am);

        int dir = 0;
        scan(am, img_rgb, &dir);
        selectPoints(am, dir);

        // ------------------------------------
        // RANSAC + рисование
        // ------------------------------------
        LineFitResult r1 = runRansacDraw(am,
            frame, am->points_1, dir,
            Scalar(255,0,0),
            "Line1", Point(10,30)
        );

        LineFitResult r2 = runRansacDraw(am,
            frame, am->points_2, dir,
            Scalar(0,255,0),
            "Line2", Point(10,60)
        );

        t_all.stop();

        // ------------------------------------
        // Углы для записи
        // ------------------------------------
        float a1 = r1.ok ? r1.angle : 0.0f;
        float a2 = r2.ok ? r2.angle : 0.0f;

        // ------------------------------------
        // Пишем в CSV
        // ------------------------------------
        csv << frameIndex << "," << a1 << "," << a2 << "," << t_all.getTimeMilli() << "\n";

        imshow("Video", frame);
        if (waitKey(1) >= 0)
            break;

        frameIndex++;
    }

    csv.close();
    anglemeterDestroy(am);
    return 0;
}
