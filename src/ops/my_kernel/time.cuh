#include <chrono>

class Timer {
std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
public:
Timer() {}
void tik() {
    start_time = std::chrono::system_clock::now();
}
void tok() {
    end_time = std::chrono::system_clock::now();
}
float get_time() {
    return std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
}
};